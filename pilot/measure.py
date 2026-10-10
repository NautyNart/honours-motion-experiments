"""Contact-conditioned kinematic proxies, never a complete physical validity test."""
import math
import numpy as np
from .common import FEET, JOINTS, motion_hash

def intervals(mask, minimum=1):
    padded = np.r_[False, mask, False].astype(int)
    starts, ends = np.where(np.diff(padded) == 1)[0], np.where(np.diff(padded) == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= minimum]

def mask_intervals(items, length):
    mask = np.zeros(length, dtype=bool)
    for item in items:
        a, b = item['start'], item['end']
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= length:
            raise ValueError(f'Invalid half-open frame interval [{a},{b}) for {length} frames')
        mask[a:b] = True
    return mask

def measure(xyz, fps, ann, thresholds):
    xyz = np.asarray(xyz)
    n = len(xyz)
    if fps <= 0 or xyz.shape != (n, 22, 3) or n < 2 or not np.isfinite(xyz).all():
        raise ValueError('Invalid positions or frame rate')
    if ann.get('motion_hash') and ann['motion_hash'] != motion_hash(xyz):
        raise ValueError('Annotation belongs to a different motion; re-export review')
    floor = float(ann.get('floor_m', 0))
    if not math.isfinite(floor):
        raise ValueError('Ground height must be finite')
    minimum = max(1, math.ceil(thresholds['min_duration_s'] * fps))
    height = xyz[:, :, 1] - floor
    depth = np.maximum(-height, 0)
    valid_geometry = bool(ann.get('reviewed') and ann.get('units_confirmed') and ann.get('floor_verified'))
    support_known = valid_geometry and bool(ann.get('support_reviewed'))
    events = []
    def add_events(mask, kind, body, values, units, minimum_frames=minimum):
        for a, b in intervals(mask, minimum_frames):
            events.append(dict(defect=kind, body_part=body, start=a, end=b,
                               start_s=a / fps, end_s=b / fps,
                               severity=float(np.max(values[a:b])), severity_unit=units,
                               status='candidate'))
    for j, name in enumerate(JOINTS):
        body = next((foot for foot, indices in FEET.items() if j in indices), name)
        add_events(depth[:, j] > thresholds['penetration_depth_m'], 'penetration', body, depth[:, j], 'm')
    summary = dict(penetration_max_m=float(depth.max()),
                   penetration_mean_m=float(depth.mean()),
                   penetration_frame_fraction=float((depth.max(axis=1) > thresholds['penetration_depth_m']).mean()),
                   sliding_mean_m_s=None, sliding_path_m=None, floating_max_m=None,
                   floating_duration_s=None, support_frame_fraction=None)
    foot_series = {}
    speeds, paths = [], []
    required = np.zeros(n, dtype=bool)
    if support_known:
        required = mask_intervals(ann.get('required_support', []), n)
        airborne = mask_intervals(ann.get('airborne', []), n)
        annotated_anchors = {}
        if np.any(required & airborne):
            raise ValueError('Required-support and legitimate-airborne intervals overlap')
        for support in ann.get('support', []):
            foot = support['body_part']
            if foot not in FEET or support.get('anchor') not in ('ankle', 'toe', 'both'):
                raise ValueError('Support must identify left/right foot and ankle/toe/both anchor')
            mask = mask_intervals([support], n)
            if np.any(mask & airborne):
                raise ValueError('Support-foot and legitimate-airborne intervals overlap')
            required |= mask
            ids = FEET[foot] if support['anchor'] == 'both' else [FEET[foot][support['anchor'] == 'toe']]
            for j in ids:
                previous = annotated_anchors.setdefault(j, np.zeros(n, dtype=bool))
                if np.any(previous & mask):
                    raise ValueError('Overlapping support intervals double-count the same anchor')
                previous |= mask
                step = np.linalg.norm(np.diff(xyz[:, j][:, [0, 2]], axis=0), axis=1)
                speed = np.r_[step * fps, 0.0]
                valid = mask & np.r_[mask[1:], False]
                speeds.extend(speed[valid].tolist())
                paths.extend(step[valid[:-1]].tolist())
                foot_series.setdefault(JOINTS[j], speed.tolist())
                add_events(valid & (speed > thresholds['sliding_speed_m_s']), 'sliding', foot, speed, 'm/s')
        # Floating requires an explicit expected-support interval. A swing foot alone is never sufficient.
        lowest_foot = height[:, [7, 8, 10, 11]].min(axis=1)
        floating = required & (lowest_foot > thresholds['floating_height_m'])
        add_events(floating, 'floating', 'both_feet', lowest_foot, 'm')
        summary.update(sliding_mean_m_s=float(np.mean(speeds)) if speeds else None,
                       sliding_path_m=float(np.sum(paths)) if paths else None,
                       floating_max_m=float(np.maximum(lowest_foot[required], 0).max()) if required.any() else None,
                       floating_duration_s=float(floating.sum() / fps) if required.any() else None,
                       support_frame_fraction=float(required.mean()))
    # Merge duplicate joint events within a foot, preserving the maximum severity.
    merged = {}
    for e in events:
        key = (e['defect'], e['body_part'], e['start'], e['end'])
        if key not in merged or e['severity'] > merged[key]['severity']:
            merged[key] = e
    availability = {'penetration': valid_geometry,
                    'sliding': support_known and bool(ann.get('support')),
                    'floating': support_known and bool(required.any())}
    return dict(motion_hash=motion_hash(xyz), frames=n, fps=fps, floor_m=floor,
                thresholds=thresholds, availability=availability,
                summary=summary, events=list(merged.values()),
                series={'min_joint_height_m': height.min(axis=1).tolist(),
                        'min_foot_height_m': height[:, [7, 8, 10, 11]].min(axis=1).tolist(),
                        'support_foot_speed_m_s': foot_series},
                caveat='Joint proxies. Unverified geometry produces exploratory penetration candidates only. Missing support is unknown, not zero.')

def rule_feedback(metrics):
    defects = [e for e in metrics['events'] if metrics['availability'][e['defect']]]
    if not defects:
        complete = all(metrics['availability'].values())
        return dict(decision='no_defect' if complete else 'uncertain', defects=[], instruction='',
                    reason='No threshold crossings' if complete else 'Reviewed geometry/support is incomplete')
    actions = {'sliding': 'keep the supporting foot planted without sliding',
               'penetration': 'keep the affected joints above the floor',
               'floating': 'restore the required ground contact without removing legitimate airborne phases'}
    phrases = [f"{d['body_part']} from {d['start_s']:.2f} to {d['end_s']:.2f} seconds: {actions[d['defect']]}"
               for d in defects]
    return dict(decision='defect', defects=defects,
                instruction='; '.join(phrases) + '. Preserve the requested action, direction and timing elsewhere.')
