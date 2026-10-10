"""Resumable GPU preparation and matched single-candidate feedback comparisons."""
from pathlib import Path
import time
import numpy as np
from .common import (read, write, manifest, save_manifest, load_xyz, annotation, digest,
                     now, error_record, file_hash, motion_hash, code_identity)
from .data import save_motion
from .measure import measure, rule_feedback

CONDITIONS = ('unchanged', 'semantic', 'rule', 'human')

def validate_config(config):
    if config['rounds'] not in (1, 2) or config['model']['candidate_budget'] != 1:
        raise ValueError('Use one candidate and one or two rounds')
    if not set(config['conditions']).issubset(CONDITIONS):
        raise ValueError('Unknown experiment condition')
    if config['model']['dtype'] not in ('auto', 'float16', 'bfloat16'):
        raise ValueError('Model dtype must be auto, float16 or bfloat16')
    if config.get('limit') is not None and config['limit'] < 1:
        raise ValueError('Clip limit must be positive or null')
    for k in ('sliding_speed_m_s', 'penetration_depth_m', 'floating_height_m', 'min_duration_s'):
        if not np.isfinite(config['thresholds'][k]) or config['thresholds'][k] <= 0:
            raise ValueError('Measurement thresholds must be finite and positive')

def measure_all(work, config):
    from .review import review_items
    records = []
    for key, item in review_items(work).items():
        try:
            xyz = load_xyz(Path(work) / item['path'])
            ann = annotation(work, key)
            metrics = measure(xyz, item['fps'], ann, config['thresholds'])
            fingerprint = digest([key, metrics['motion_hash'], ann, config['thresholds']])
            path = Path('metrics') / f'{fingerprint}.json'
            write(Path(work) / path, metrics)
            records.append(dict(key=key, path=path.as_posix(), summary=metrics['summary'],
                                availability=metrics['availability'], annotation_reviewed=ann.get('reviewed', False)))
        except Exception as exc:
            error_record(work, 'measure', key, exc)
            records.append(dict(key=key, error=str(exc)))
    write(Path(work) / 'metrics_index.json', records)
    return records

def gpu_prepare(work, config, mode, backend_factory=None):
    from .backend import IRGBackend
    work = Path(work)
    rows = manifest(work)
    selected = [r for r in rows if
                (mode == 'generate' and r['origin'] == 'generated' and not r['paths']) or
                (mode == 'reconstruct' and 'raw_263' in r['paths'] and 'initial_tokens' not in r['paths'])]
    if config.get('limit'):
        selected = selected[:config['limit']]
    if not selected:
        return {'processed': 0, 'pending': 0}
    backend = None
    done = 0
    try:
        backend = (backend_factory or IRGBackend)(config)
        for row in selected:
            try:
                if mode == 'generate':
                    tokens, features, trace = backend.generate(row['caption'], row['seed'])
                    from .common import recover_xyz
                    candidate_hash = motion_hash(recover_xyz(features))
                    if any(other.get('raw_hash') == candidate_hash for other in rows):
                        raise ValueError('Generated candidate duplicates an existing clip; keep this slot pending and revise the cohort explicitly')
                    save_motion(work, row, features, 'raw', tokens)
                    info = {'origin': 'direct_decoder_output', 'pre_tokenisation_source_available': False}
                else:
                    raw = np.load(work / row['paths']['raw_263'], allow_pickle=False)
                    cache_key = digest([file_hash(work / row['paths']['raw_263']), backend.codec_id])
                    cache = work / 'token_cache' / cache_key
                    if (cache / 'info.json').exists():
                        tokens = np.load(cache / 'tokens.npy', allow_pickle=False)
                        features = np.load(cache / 'features.npy', allow_pickle=False)
                        info = read(cache / 'info.json')
                    else:
                        tokens, features, info = backend.encode(raw)
                        cache.mkdir(parents=True, exist_ok=True)
                        np.save(cache / 'tokens.npy', tokens)
                        np.save(cache / 'features.npy', features)
                        write(cache / 'info.json', info)
                    trace = {'tokenisation': info}
                save_motion(work, row, features, 'initial', tokens)
                row.update(status='ready_for_review', settings_id=digest(backend.identity),
                           model_identity=backend.identity, reconstruction=info,
                           annotation_status='pending_post_tokenisation_review')
                path = work / 'clips' / row['clip_id'] / f'{mode}_trace.json'
                write(path, trace)
                save_manifest(work, rows)  # progress survives an interrupted job
                done += 1
            except Exception as exc:
                error_record(work, mode, row['clip_id'], exc)
                if hasattr(exc, 'generation_trace'):
                    write(work / 'clips' / row['clip_id'] / 'failed_generation_trace.json', exc.generation_trace)
                row['last_error'] = str(exc)
                save_manifest(work, rows)
    except Exception as exc:
        if hasattr(exc, 'gpu_session'):
            write(work / 'gpu_sessions' / f'{digest([now(), str(exc)])}.json', exc.gpu_session)
        raise
    finally:
        if backend:
            record = backend.session_record()
            record.update(command=mode, finished_at=now())
            write(work / 'gpu_sessions' / f'{digest(record)}.json', record)
    return {'processed': done, 'pending': len(selected) - done}

def refine(work, config, run_id, backend_factory=None):
    from .backend import IRGBackend
    validate_config(config)
    if not run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('Run ID must be a simple directory name')
    work = Path(work)
    run = work / 'runs' / run_id
    run.mkdir(parents=True, exist_ok=True)
    snapshot = run / 'config.json'
    if snapshot.exists() and read(snapshot) != config:
        raise ValueError('Run configuration changed. Choose a new run ID to preserve comparability.')
    write(snapshot, config)
    selected = [r for r in manifest(work) if 'initial_tokens' in r['paths'] and r['subset'] == 'comparison']
    if config.get('limit'):
        selected = selected[:config['limit']]
    backend, backend_failure = None, None
    snapshot_values = {'code_identity.json': code_identity(),
                       'cohort.json': [r['clip_id'] for r in selected]}
    for name, value in snapshot_values.items():
        if (run / name).exists() and read(run / name) != value:
            raise ValueError(f'{name} changed; select a new run ID')
        write(run / name, value)
    counts = {'complete': 0, 'blocked': 0, 'errors': 0, 'resumed': 0}
    validated_inputs = set()
    def get_backend():
        nonlocal backend, backend_failure
        if backend_failure:
            raise RuntimeError(backend_failure)
        if backend is None:
            try:
                backend = (backend_factory or IRGBackend)(config)
                identity = run / 'model_identity.json'
                if identity.exists() and read(identity) != backend.identity:
                    raise ValueError('Frozen model/runtime identity changed; use a new run ID')
                write(identity, backend.identity)
            except Exception as exc:
                if hasattr(exc, 'gpu_session'):
                    exc.gpu_session['run_id'] = run_id
                    write(work / 'gpu_sessions' / f'{digest([now(), str(exc)])}.json', exc.gpu_session)
                backend_failure = str(exc)
                raise
        return backend
    def validated_backend(current):
        model = get_backend()
        token_hash = file_hash(work / current['tokens'])
        feature_hash = file_hash(work / current['features'])
        pair = (token_hash, feature_hash)
        if pair not in validated_inputs:
            tokens = np.load(work / current['tokens'], allow_pickle=False)
            decoded = model.decode(tokens)
            saved = np.load(work / current['features'], allow_pickle=False)
            if decoded.shape != saved.shape or not np.allclose(decoded, saved, atol=1e-4, rtol=1e-4):
                raise ValueError('Cached tokens do not reproduce the initial clip with this decoder/statistics')
            validated_inputs.add(pair)
        return model
    try:
        for row in selected:
            for condition in config['conditions']:
                paths = row['paths']
                current = dict(xyz=paths['initial_xyz'], features=paths['initial_263'], tokens=paths['initial_tokens'])
                input_key = row['clip_id'] + '/initial'
                for round_number in range(1, config['rounds'] + 1):
                    folder = run / 'cases' / row['clip_id'] / condition / f'round_{round_number}'
                    folder.mkdir(parents=True, exist_ok=True)
                    key = f'{run_id}/{row["clip_id"]}/{condition}/round_{round_number}'
                    start = time.perf_counter()
                    try:
                        ann = annotation(work, input_key)
                        initial_hash = file_hash(work / current['tokens'])
                        fingerprint = digest([config, row['caption'], current, initial_hash,
                                              file_hash(work / current['features']), ann])
                        output = folder / 'result.json'
                        if output.exists() and read(output).get('status') == 'complete':
                            old = read(output)
                            if old['fingerprint'] != fingerprint:
                                raise ValueError('Inputs/review changed for completed case. Use a new run ID.')
                            counts['resumed'] += 1
                            current = dict(xyz=old['output_xyz'], features=old['output_features'], tokens=old['output_tokens'])
                            input_key = old['output_key']
                            if old['feedback']['decision'] != 'defect' or condition == 'unchanged':
                                break
                            continue
                        xyz = load_xyz(work / current['xyz'])
                        metrics = measure(xyz, row['fps'], ann, config['thresholds'])
                        tokens = np.load(work / current['tokens'], allow_pickle=False)
                        feedback, assessor_trace, editor_trace = None, None, None
                        if condition == 'unchanged':
                            feedback = dict(decision='not_assessed', instruction='', defects=[], reason='Reference condition; no assessment performed')
                        elif condition == 'rule':
                            if round_number > 1 and not all(ann.get(k) for k in ('reviewed', 'units_confirmed', 'floor_verified', 'support_reviewed')):
                                write(output, dict(status='blocked', reason='Review support and geometry on the edited motion before rule round two',
                                                   input_key=input_key, output_key=key, clip_id=row['clip_id'], condition=condition))
                                counts['blocked'] += 1
                                break
                            feedback = rule_feedback(metrics)
                        elif condition == 'human':
                            feedback = ann.get('feedback', {})
                            if not ann.get('reviewed') or not feedback.get('approved') or ann.get('motion_hash') != motion_hash(xyz):
                                write(output, dict(status='blocked', reason='Explicit human-verified feedback required for this exact motion',
                                                   input_key=input_key, output_key=key, clip_id=row['clip_id'], condition=condition))
                                counts['blocked'] += 1
                                break
                        else:
                            feedback, assessor_trace = validated_backend(current).semantic(row['caption'], tokens, config['seed'] + round_number - 1)
                        # Save the actual feedback and measurement evidence before a potentially failing generation.
                        write(folder / 'feedback.json', dict(input_key=input_key, feedback=feedback,
                              assessor_trace=assessor_trace, annotation_snapshot=ann if condition == 'human' else None))
                        write(folder / 'before_metrics.json', metrics)
                        if condition != 'unchanged' and feedback['decision'] == 'defect':
                            tokens, features, editor_trace = validated_backend(current).edit(row['caption'], tokens, feedback, config['seed'] + round_number - 1)
                        else:
                            features = np.load(work / current['features'], allow_pickle=False)
                        from .common import recover_xyz
                        after = recover_xyz(features)
                        np.save(folder / 'motion_263.npy', features)
                        np.save(folder / 'motion_xyz.npy', after)
                        np.save(folder / 'tokens.npy', tokens)
                        if editor_trace:
                            write(folder / 'editor_trace.json', editor_trace)
                        record = dict(status='complete', fingerprint=fingerprint, clip_id=row['clip_id'],
                                      source_recording=row['source_recording'], subset=row['subset'],
                                      condition=condition, round=round_number, caption=row['caption'], fps=row['fps'],
                                      input_key=input_key, output_key=key, input_xyz=current['xyz'],
                                      input_tokens_sha256=initial_hash, output_xyz=(folder / 'motion_xyz.npy').relative_to(work).as_posix(),
                                      output_features=(folder / 'motion_263.npy').relative_to(work).as_posix(),
                                      output_tokens=(folder / 'tokens.npy').relative_to(work).as_posix(),
                                      feedback=feedback, unchanged=bool(np.array_equal(xyz, after)),
                                      elapsed_seconds=time.perf_counter() - start,
                                      assessor_seconds=assessor_trace['seconds'] if assessor_trace else 0,
                                      editor_seconds=editor_trace['seconds'] if editor_trace else 0,
                                      peak_allocated_gib=max([t.get('peak_allocated_gib', 0) for t in (assessor_trace, editor_trace) if t] or [0]),
                                      completed_at=now())
                        write(output, record)
                        counts['complete'] += 1
                        current = dict(xyz=record['output_xyz'], features=record['output_features'], tokens=record['output_tokens'])
                        input_key=key
                        if feedback['decision'] != 'defect' or condition == 'unchanged':
                            break
                    except Exception as exc:
                        error_record(work, 'refine', key, exc)
                        if hasattr(exc, 'generation_trace'):
                            write(folder / 'failed_generation_trace.json', exc.generation_trace)
                        # Preserve completed records even when a resume conflict is detected.
                        target = folder / ('resume_error.json' if (folder / 'result.json').exists() and read(folder / 'result.json').get('status') == 'complete' else 'result.json')
                        write(target, dict(status='error', error=str(exc), clip_id=row['clip_id'], condition=condition,
                                           input_key=input_key, output_key=key, attempted_at=now()))
                        counts['errors'] += 1
                        break
    finally:
        if backend:
            record = backend.session_record()
            record.update(command='refine', run_id=run_id, finished_at=now())
            write(work / 'gpu_sessions' / f'{digest(record)}.json', record)
        write(run / 'last_invocation.json', dict(counts=counts, completed_at=now()))
    return counts
