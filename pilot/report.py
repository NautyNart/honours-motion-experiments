"""Exploratory paired reports, with explicit review coverage and no invented outcomes."""
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .common import (read, write, manifest, annotation, csv_write, DEFECTS)
from .experiment import measure_all
from .measure import rule_feedback

def temporal_iou(a, b):
    intersection = max(0, min(a['end'], b['end']) - max(a['start'], b['start']))
    union = max(a['end'], b['end']) - min(a['start'], b['start'])
    return intersection / union if union else 0

def cluster_interval(rows, seed, repeats):
    groups = defaultdict(list)
    for family, value in rows:
        groups[family].append(value)
    if len(groups) < 5:
        return None
    arrays = [np.asarray(v) for v in groups.values()]
    rng = np.random.default_rng(seed)
    estimates = [np.concatenate([arrays[i] for i in rng.integers(len(arrays), size=len(arrays))]).mean() for _ in range(repeats)]
    return [float(v) for v in np.quantile(estimates, [.025, .975])]

def report(work, config, run_id=None):
    work = Path(work)
    if run_id and (work / 'runs' / run_id / 'config.json').exists():
        config = read(work / 'runs' / run_id / 'config.json')
    directory = work / 'reports' / (run_id or 'inventory')
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / 'measurement_config.json', config)
    rows = manifest(work)
    by_id = {r['clip_id']: r for r in rows}
    index = {m['key']: m for m in measure_all(work, config) if 'error' not in m}
    measurements = []
    for key, value in index.items():
        measurements.append(dict(key=key, reviewed=value['annotation_reviewed'], **value['availability'], **value['summary']))
    csv_write(directory / 'measurements.csv', measurements)
    reconstruction_rows = []
    for row in rows:
        raw, reconstructed = index.get(row['clip_id'] + '/raw'), index.get(row['clip_id'] + '/initial')
        if raw and reconstructed:
            item = dict(clip_id=row['clip_id'], source_recording=row['source_recording'], origin=row['origin'],
                        reconstruction_status=row.get('reconstruction_status', row.get('reconstruction', {}).get('codec_id', 'unknown')))
            for metric in raw['summary']:
                a, b = raw['summary'][metric], reconstructed['summary'][metric]
                item['raw_' + metric], item['reconstructed_' + metric] = a, b
                item['delta_' + metric] = b - a if a is not None and b is not None else None
            reconstruction_rows.append(item)
    csv_write(directory / 'raw_reconstruction.csv', reconstruction_rows,
              None if reconstruction_rows else ['clip_id', 'origin', 'reconstruction_status'])
    detection, localisation = [], []
    counts = defaultdict(Counter)
    for row in rows:
        key = row['clip_id'] + '/initial'
        ann = annotation(work, key)
        if key not in index or not ann.get('reviewed') or not ann.get('labels_complete'):
            continue
        metrics = read(work / index[key]['path'])
        proposals = rule_feedback(metrics)['defects']
        for kind in DEFECTS:
            count = counts[(row['subset'], row['origin'], kind)]
            if not metrics['availability'][kind]:
                count['unknown'] += 1
                continue
            truth = [e for e in ann['labels'] if e['defect'] == kind]
            pred = [e for e in proposals if e['defect'] == kind]
            category = ('tp' if truth else 'fp') if pred else ('fn' if truth else 'tn')
            count[category] += 1
            detection.append(dict(clip_id=row['clip_id'], subset=row['subset'], defect=kind,
                                  category=category, origin=row['origin'], source_recording=row['source_recording']))
            # One-to-one greedy event matching by temporal overlap. Report body separately.
            remaining = list(pred)
            for target in truth:
                best = max(remaining, key=lambda e: temporal_iou(e, target), default=None)
                overlap = temporal_iou(best, target) if best else 0
                if best and overlap > 0:
                    remaining.remove(best)
                localisation.append(dict(clip_id=row['clip_id'], defect=kind, origin=row['origin'], subset=row['subset'],
                    target_body=target['body_part'], predicted_body=best['body_part'] if overlap else '',
                    temporal_iou=overlap, body_correct=bool(overlap and best['body_part'] == target['body_part'])))
    csv_write(directory / 'detection_per_clip.csv', detection,
              ['clip_id', 'subset', 'defect', 'category', 'origin', 'source_recording'])
    csv_write(directory / 'localisation.csv', localisation,
              ['clip_id', 'defect', 'origin', 'subset', 'target_body', 'predicted_body', 'temporal_iou', 'body_correct'])
    detection_summary = []
    for (subset, origin, kind), c in sorted(counts.items()):
        detection_summary.append(dict(subset=subset, origin=origin, defect=kind, **{k: c[k] for k in ('tp', 'fp', 'fn', 'tn', 'unknown')},
            precision=c['tp'] / (c['tp'] + c['fp']) if c['tp'] + c['fp'] else None,
            recall=c['tp'] / (c['tp'] + c['fn']) if c['tp'] + c['fn'] else None,
            false_alarm_rate=c['fp'] / (c['fp'] + c['tn']) if c['fp'] + c['tn'] else None))
    write(directory / 'detection_summary.json', detection_summary)
    results = []
    case_paths = sorted((work / 'runs' / run_id).glob('cases/*/*/round_*/result.json')) if run_id else []
    for path in case_paths:
        result = read(path)
        entry = dict(clip_id=result['clip_id'], condition=result['condition'], status=result['status'],
                     round=result.get('round', int(path.parent.name.split('_')[-1])), reason=result.get('reason', result.get('error', '')))
        if result['status'] == 'complete':
            key = result['output_key']
            before, after = index.get(result['input_key']), index.get(key)
            reviewed = annotation(work, key)
            outcome = reviewed.get('outcome', {})
            initial_ann = annotation(work, result['clip_id'] + '/initial')
            target_known = bool(initial_ann.get('reviewed') and initial_ann.get('labels_complete') and initial_ann.get('labels'))
            success = None
            if target_known and reviewed.get('reviewed') and outcome.get('reviewed'):
                required = [outcome.get('target_resolved'), outcome.get('new_defect'), outcome.get('action_preserved')]
                if all(isinstance(x, bool) for x in required):
                    success = required == [True, False, True]
            entry.update(source_recording=result['source_recording'],
                         unchanged=result['unchanged'], decision=result['feedback']['decision'],
                         target_known=target_known, reviewed_success=success,
                         action_preserved=outcome.get('action_preserved'), new_defect=outcome.get('new_defect'),
                         unnecessary_edit=outcome.get('unnecessary_edit'),
                         elapsed_seconds=result['elapsed_seconds'], editor_seconds=result['editor_seconds'],
                         assessor_seconds=result['assessor_seconds'], peak_allocated_gib=result['peak_allocated_gib'])
            for metric, defect in [('penetration_max_m', 'penetration'), ('sliding_mean_m_s', 'sliding'), ('floating_duration_s', 'floating')]:
                a = before['summary'][metric] if before else None
                b = after['summary'][metric] if after else None
                entry['before_' + metric], entry['after_' + metric] = a, b
                entry['delta_' + metric] = b - a if a is not None and b is not None else None
                entry['reviewed_' + metric] = bool(before and after and before['availability'][defect] and after['availability'][defect])
        results.append(entry)
    csv_write(directory / 'comparison_per_clip.csv', results,
              None if results else ['clip_id', 'condition', 'status', 'round', 'reviewed_success'])
    summary = []
    unfinished = {(r['clip_id'], r['condition']) for r in results if r['status'] != 'complete'}
    for condition in config['conditions']:
        cases = [r for r in results if r['condition'] == condition]
        # Final completed round per clip; round-level records remain in the CSV.
        final = {}
        for case in cases:
            if case['status'] == 'complete':
                if case['clip_id'] not in final or case['round'] > final[case['clip_id']]['round']:
                    final[case['clip_id']] = case
        reviewed = [r for r in final.values() if r.get('reviewed_success') is not None
                    and (r['clip_id'], condition) not in unfinished]
        success_rows = [(r['source_recording'], float(r['reviewed_success'])) for r in reviewed]
        summary.append(dict(condition=condition, completed_clips=len(final),
                            blocked_or_error_records=sum(r['status'] != 'complete' for r in cases),
                            reviewed_defective_clips=len(reviewed), successful_corrections=sum(r['reviewed_success'] for r in reviewed),
                            unchanged_clips=sum(r['unchanged'] for r in final.values()),
                            success_fraction=np.mean([v for _, v in success_rows]).item() if success_rows else None,
                            descriptive_cluster_bootstrap_95=cluster_interval(success_rows, config['seed'], config['bootstrap_replicates'])))
    write(directory / 'comparison_summary.json', summary)
    final_reviewed = {}
    for r in results:
        if r['status'] == 'complete' and r.get('reviewed_success') is not None and (r['clip_id'], r['condition']) not in unfinished:
            key = (r['clip_id'], r['condition'])
            if key not in final_reviewed or r['round'] > final_reviewed[key]['round']:
                final_reviewed[key] = r
    paired, paired_summary = [], []
    for alternative, reference in [('rule', 'semantic'), ('human', 'semantic'), ('human', 'rule')]:
        values = []
        for clip_id in by_id:
            a, b = final_reviewed.get((clip_id, alternative)), final_reviewed.get((clip_id, reference))
            if a and b:
                change = int(a['reviewed_success']) - int(b['reviewed_success'])
                paired.append(dict(clip_id=clip_id, source_recording=a['source_recording'],
                                   alternative=alternative, reference=reference, success_difference=change))
                values.append((a['source_recording'], change))
        paired_summary.append(dict(alternative=alternative, reference=reference, matched_reviewed_clips=len(values),
            mean_success_difference=float(np.mean([v for _, v in values])) if values else None,
            descriptive_cluster_bootstrap_95=cluster_interval(values, config['seed'], config['bootstrap_replicates'])))
    csv_write(directory / 'matched_feedback_comparisons.csv', paired,
              ['clip_id', 'source_recording', 'alternative', 'reference', 'success_difference'])
    write(directory / 'matched_feedback_summary.json', paired_summary)
    sessions = [read(p) for p in (work / 'gpu_sessions').glob('*.json')]
    if run_id:
        sessions = [s for s in sessions if s.get('run_id') == run_id]
    gpu_hours = sum(s['allocated_process_gpu_hours'] for s in sessions)
    available = sum('raw_xyz' in r['paths'] for r in rows)
    reviewed_count = sum(annotation(work, r['clip_id'] + '/initial').get('labels_complete', False) for r in rows)
    lines = ['# Pilot execution report', '', f'Available unique clip slots: **{available}/50**. Missing: **{50-available}**.',
             f'Initial clips with completed label review: **{reviewed_count}**.',
             f'Recorded GPU process allocation: **{gpu_hours:.6f} GPU-hours** across {len(sessions)} sessions.',
             'This excludes notebook reservation time outside recorded processes. No simulator is used.', '',
             '## Comparison coverage', '', '| Condition | Completed clips | Pending/error records | Reviewed defective clips | Successes | Unchanged |',
             '|---|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['condition']} | {s['completed_clips']} | {s['blocked_or_error_records']} | {s['reviewed_defective_clips']} | {s['successful_corrections']} | {s['unchanged_clips']} |")
    lines += ['', 'Success means the target defect is resolved, no new defect appears, and the requested action is preserved.',
              'Unreviewed or uncertain cases do not count as successes. Missing support measurements are null, not zero.',
              'Cases awaiting another configured round are excluded from final success summaries. Completed clips counts include available intermediate outputs.',
              'Detection summary uses reviewed labels only; per-clip CSV separates calibration/comparison and origin.',
              'Confidence intervals, when at least five source groups exist, are exploratory source-cluster bootstrap summaries; no significance claim.',
              '', '## Dependencies and observed execution problems', '']
    issues = Counter(r['reason'] for r in results if r['status'] != 'complete')
    lines += [f'- {count} record(s): {reason}' for reason, count in issues.items()] or ['- No editing attempts recorded, or no execution errors in selected results.']
    if not reviewed_count:
        lines.append('- Physical failure labels are unreviewed; detection accuracy and correction efficacy cannot yet be estimated.')
    if available < 50:
        lines.append('- Missing HumanML3D inputs/reviewed corruptions and/or generated candidates remain explicit manifest slots.')
    lines += ['', '## Measurement plots', '', '![Initial penetration proxy](initial_penetration.svg)',
              '', 'These are joint proxies with the configured floor. Unverified geometry is exploratory.',
              '', 'CSV files retain per-clip measurements, paired changes, timing and review availability.']
    if results:
        lines += ['', '![Paired penetration proxy](paired_penetration.svg)']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    plot(directory, rows, index, results)
    return dict(available_clips=available, reviewed_initial_clips=reviewed_count, gpu_hours=gpu_hours,
                report=str(directory / 'report.md'))

def plot(directory, rows, index, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    pairs = [(r['clip_id'], index[r['clip_id'] + '/initial']['summary']['penetration_max_m'])
             for r in rows if r['clip_id'] + '/initial' in index]
    fig, ax = plt.subplots(figsize=(9, 4))
    if pairs:
        ax.bar([p[0] for p in pairs], [p[1] * 1000 for p in pairs])
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(.5, .5, 'No decoded initial motions', ha='center', transform=ax.transAxes)
    ax.set_ylabel('Maximum below-floor joint depth (mm)')
    ax.set_title('Exploratory joint penetration proxy; labels/ground may be unreviewed')
    fig.tight_layout()
    fig.savefig(directory / 'initial_penetration.svg')
    plt.close(fig)
    if results:
        fig, ax = plt.subplots(figsize=(7, 4))
        for row in results:
            a, b = row.get('before_penetration_max_m'), row.get('after_penetration_max_m')
            if a is not None and b is not None:
                ax.plot([0, 1], [1000*a, 1000*b], marker='o', alpha=.5, label=f"{row['clip_id']} / {row['condition']}")
        ax.set_xticks([0, 1], ['Before', 'After'])
        ax.set_ylabel('Maximum below-floor joint depth (mm)')
        ax.set_title('Paired proxies; inspect review coverage before interpreting')
        fig.tight_layout()
        fig.savefig(directory / 'paired_penetration.svg')
        plt.close(fig)
