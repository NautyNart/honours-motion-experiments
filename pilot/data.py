"""A 50-slot plan with honest missing-data states; original assets are never modified."""
import csv
from pathlib import Path
import shutil
import numpy as np
from .common import (read, write, manifest, save_manifest, digest, motion_hash, recover_xyz, file_hash,
                     load_xyz, annotation)

PROMPTS = [
    'A person walks forward, turns left, and stops.',
    'A person raises their right arm and waves.',
    'A person squats down and then stands up.',
    'A person takes two steps backwards.',
    'A person jumps once and lands on both feet.',
    'A person stands still with both feet on the ground.',
    'A person walks slowly forward and stops.',
    'A person runs forward with natural flight phases between steps.',
    'A person hops twice on the left foot and stops.',
    'A person hops twice on the right foot and stops.',
    'A person steps sideways to the left and brings their feet together.',
    'A person steps sideways to the right and brings their feet together.',
    'A person bends forward while keeping both feet planted.',
    'A person jumps forward and lands before standing still.',
    'A person walks forward, turns right, and stops.'
]

def plan(seed):
    rows = []
    def record(clip_id, origin, family, subset, **extra):
        return dict(clip_id=clip_id, origin=origin, source_recording=family,
                    source_partition='pending' if origin != 'generated' else 'not_applicable',
                    subset=subset, caption='', fps=20, frames=None, representation='humanml3d_263_and_xyz22',
                    units='nominal_metres_uniform_skeleton', seed=seed, status='pending_source',
                    defect_labels=[], annotation_status='pending', paths={}, settings_id=None, **extra)
    for i in range(15):
        subset = 'calibration' if i in (0, 4, 10) else 'comparison'
        base = f'original_{i:02d}'
        rows.append(record(base, 'original', base, subset, intended_role='acceptable_candidate'))
        types = ['sliding', 'penetration'] if i < 4 else ['sliding', 'floating'] if i < 8 else ['penetration', 'floating'] if i < 10 else []
        for defect in types:
            rows.append(record(f'{base}_{defect}', 'controlled_corruption', base, subset,
                               parent_id=base, intended_defect=defect, intended_role='controlled_defect'))
    for i, prompt in enumerate(PROMPTS):
        rows.append(record(f'generated_{i:02d}', 'generated', f'prompt_{digest(prompt)[:12]}',
                           'calibration' if i < 3 else 'comparison', intended_role='generated_candidate'))
        rows[-1]['caption'] = prompt
    assert len(rows) == 50
    return rows

def save_motion(work, row, features, phase, tokens=None):
    folder = Path(work) / 'clips' / row['clip_id']
    folder.mkdir(parents=True, exist_ok=True)
    features = np.asarray(features, dtype=np.float32)
    xyz = recover_xyz(features)
    for suffix, array in [('263', features), ('xyz', xyz)]:
        path = folder / f'{phase}_{suffix}.npy'
        np.save(path, array)
        row['paths'][f'{phase}_{suffix}'] = path.relative_to(work).as_posix()
    if tokens is not None:
        path = folder / f'{phase}_tokens.npy'
        np.save(path, np.asarray(tokens, dtype=np.int64).reshape(-1))
        row['paths'][f'{phase}_tokens'] = path.relative_to(work).as_posix()
    row['frames'] = len(xyz)
    row[f'{phase}_hash'] = motion_hash(xyz)
    return xyz

def prepare(work, config, baseline=None, source_index=None):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    rows = manifest(work) if (work / 'manifest.json').exists() else plan(config['seed'])
    by_id = {r['clip_id']: r for r in rows}
    if baseline:
        baseline = Path(baseline)
        run = read(baseline / 'run.json')
        revision_path = baseline.parent / 'asset_revisions.json'
        if revision_path.exists() and not (work / 'asset_revisions.json').exists():
            write(work / 'asset_revisions.json', read(revision_path))
        provenance_path = work / 'baseline_source_hashes.json'
        if not provenance_path.exists():
            write(provenance_path, {p.relative_to(baseline).as_posix(): file_hash(p)
                                    for p in sorted(baseline.rglob('*')) if p.is_file()})
        used = {r.get('raw_hash') for r in rows if r.get('raw_hash')}
        for folder in sorted(baseline.iterdir()):
            if not (folder / 'first_263.npy').exists():
                continue
            trace = read(folder / 'trace.json')
            candidates = [r for r in rows if r['origin'] == 'generated' and r['caption'] == trace['prompt']]
            if not candidates:
                continue
            row = candidates[0]
            if row['paths']:
                continue
            features = np.load(folder / 'first_263.npy', allow_pickle=False)
            xyz = recover_xyz(features)
            saved = load_xyz(folder / 'first_xyz.npy')
            if not np.allclose(xyz, saved, atol=5e-5):
                raise ValueError(f'Coordinate recovery does not match existing decoded output: {folder}')
            if motion_hash(xyz) in used:
                continue
            tokens = trace['intermediate_tokens'][0]
            save_motion(work, row, features, 'raw', tokens)
            save_motion(work, row, features, 'initial', tokens)
            used.add(row['raw_hash'])
            row.update(status='ready_for_review', source_recording=f'prompt_{digest(row["caption"])[:12]}',
                       seed=run['seed'], settings_id=digest(run), source_run=run,
                       reconstruction_status='already_decoded_from_saved_tokens_not_reencoded',
                       pre_tokenisation_source_available=False)
            target = work / 'clips' / row['clip_id'] / 'source_trace.json'
            shutil.copy2(folder / 'trace.json', target)
            row['paths']['source_trace'] = target.relative_to(work).as_posix()
    if source_index:
        source_index = Path(source_index).resolve()
        with source_index.open(encoding='utf-8-sig', newline='') as stream:
            sources = list(csv.DictReader(stream))
        families = {}
        for src in sources:
            row = by_id[src['clip_id']]
            if row['origin'] != 'original' or not src['source_recording'].strip():
                raise ValueError('Supply original slots and recording-family IDs, including mirror/segment relatives')
            prior = families.setdefault(src['source_recording'], (row['subset'], src['source_partition']))
            if prior != (row['subset'], src['source_partition']):
                raise ValueError('A recording family crosses partitions or calibration/comparison subsets')
            if row['paths']:
                continue
            if src['source_partition'] not in ('train', 'val', 'test') or float(src['fps']) != 20:
                raise ValueError('Use official HumanML3D partition labels and 20 fps features')
            path = Path(src['features_path'])
            if not path.is_absolute():
                path = source_index.parent / path
            features = np.load(path, allow_pickle=False)
            save_motion(work, row, features, 'raw')
            row.update(caption=src['caption'], source_recording=src['source_recording'],
                       source_partition=src['source_partition'], status='pending_reconstruction',
                       source_clip_id=src['source_clip_id'], source_file_name=path.name)
            for child in rows:
                if child.get('parent_id') == row['clip_id']:
                    child.update(source_recording=row['source_recording'], source_partition=row['source_partition'],
                                 caption=row['caption'])
    # Enforce grouping against already imported rows as well.
    groups = {}
    contents = {}
    for row in rows:
        if row['paths']:
            assignment = (row['subset'], row['source_partition'])
            if groups.setdefault(row['source_recording'], assignment) != assignment:
                raise ValueError('Recording family leakage across subsets/partitions')
            if row.get('raw_hash'):
                prior = contents.setdefault(row['raw_hash'], row['clip_id'])
                if prior != row['clip_id']:
                    raise ValueError(f'Duplicate raw motion in slots {prior} and {row["clip_id"]}')
    save_manifest(work, rows)
    if not (work / 'preparation_config.json').exists():
        write(work / 'preparation_config.json', config)
    return rows

def translate_features(features, displacement):
    """Coherent global translation in the redundant 263-D representation; preserve rotations.

    Updates root velocities, all joint heights and local velocities. Contact channels
    follow the upstream speed heuristic for encoder compatibility, NOT evaluation.
    """
    f = np.array(features, dtype=np.float32, copy=True)
    d = np.asarray(displacement, dtype=np.float32)
    if d.shape != (len(f), 3) or not np.allclose(d[0, [0, 2]], 0):
        raise ValueError('Translation must be [T,3], with zero initial horizontal offset')
    angle = np.cumsum(np.r_[0, f[:-1, 0]])
    delta = np.diff(d, axis=0)
    def forward(v, a):
        c, s = np.cos(2 * a), np.sin(2 * a)
        out = v.copy()
        out[:, 0], out[:, 2] = c * v[:, 0] + s * v[:, 2], -s * v[:, 0] + c * v[:, 2]
        return out
    f[:-1, 1:3] += forward(delta, angle[1:])[:, [0, 2]]
    f[:, 3] += d[:, 1]
    f[:, 4:67].reshape(-1, 21, 3)[:, :, 1] += d[:, None, 1]
    f[:-1, 193:259].reshape(-1, 22, 3)[:] += forward(delta, angle[:-1])[:, None, :]
    xyz = recover_xyz(f)
    for k, joint in enumerate([7, 10, 8, 11]):
        f[:-1, 259 + k] = (np.sum(np.diff(xyz[:, joint], axis=0) ** 2, axis=1) < 0.002)
    return f

def corrupt(work):
    rows = manifest(work)
    by_id = {r['clip_id']: r for r in rows}
    for row in rows:
        if row['origin'] != 'controlled_corruption' or row['paths']:
            continue
        parent = by_id[row['parent_id']]
        if 'raw_263' not in parent['paths']:
            continue
        ann = annotation(work, parent['clip_id'] + '/raw')
        if not all(ann.get(k) for k in ('reviewed', 'units_confirmed', 'floor_verified', 'support_reviewed')):
            row['status'] = 'pending_parent_raw_support_review'
            continue
        if not ann.get('labels_complete') or ann.get('labels'):
            row['status'] = 'pending_acceptable_parent_review'
            continue
        features = np.load(Path(work) / parent['paths']['raw_263'], allow_pickle=False)
        xyz = recover_xyz(features)
        if ann.get('motion_hash') != motion_hash(xyz):
            raise ValueError('Parent review is stale')
        supports = [s for s in ann['support'] if s['end'] - s['start'] >= 12]
        if not supports:
            row['status'] = 'pending_suitable_support_interval'
            continue
        interval = max(supports, key=lambda s: s['end'] - s['start'])
        a, b = interval['start'], min(interval['end'], interval['start'] + 24)
        profile = np.sin(np.linspace(0, np.pi, b - a)) ** 2
        d = np.zeros((len(xyz), 3), dtype=np.float32)
        defect = row['intended_defect']
        if defect == 'sliding':
            amplitude, axis = 0.12, 0
        elif defect == 'penetration':
            amplitude, axis = -(max(0, float(xyz[a:b, [7, 8, 10, 11], 1].min() - ann['floor_m'])) + 0.08), 1
        else:
            amplitude, axis = 0.18, 1
        d[a:b, axis] = amplitude * profile
        altered = translate_features(features, d)
        save_motion(work, row, altered, 'raw')
        row.update(status='pending_reconstruction', corruption=dict(kind=defect,
                   transform='smooth_global_translation', start=a, end=b, amplitude_m=amplitude,
                   axis=axis, support_source=parent['clip_id'] + '/raw'),
                   intended_locations=[dict(defect=defect, body_part=interval['body_part'] if defect == 'sliding' else 'both_feet',
                                            start=a, end=b, severity=abs(amplitude), severity_unit='translation_m',
                                            annotation_status='construction_interval_only')],
                   annotation_status='intended_defect_only_reconstruction_requires_review')
    save_manifest(work, rows)

def make_source_template(path):
    from .common import csv_write
    fields = ['clip_id', 'source_clip_id', 'source_recording', 'source_partition', 'caption', 'fps', 'features_path']
    csv_write(path, [dict(zip(fields, [f'original_{i:02d}', '', '', '', '', 20, ''])) for i in range(15)], fields)
