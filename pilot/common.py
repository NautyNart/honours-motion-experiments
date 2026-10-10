"""Portable, atomic records. All manifest paths are relative to the pilot workspace."""
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
JOINTS = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
          'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_toe', 'right_toe',
          'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
          'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
FEET = {'left_foot': [7, 10], 'right_foot': [8, 11]}
CHAINS = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
          [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
DEFECTS = ('sliding', 'penetration', 'floating')

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def motion_hash(xyz):
    return hashlib.sha256(np.ascontiguousarray(xyz, dtype='<f4').tobytes()).hexdigest()

def csv_write(path, rows, fields=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def manifest(work):
    return read(Path(work) / 'manifest.json')

def save_manifest(work, records):
    write(Path(work) / 'manifest.json', records)

def load_xyz(path):
    xyz = np.load(path, allow_pickle=False).astype(np.float32)
    if xyz.ndim != 3 or xyz.shape[1:] != (22, 3) or len(xyz) < 2 or not np.isfinite(xyz).all():
        raise ValueError(f'Expected finite [T>=2,22,3] joint positions: {path}')
    return xyz

def annotation(work, key):
    path = Path(work) / 'annotations.json'
    return read(path).get(key, {}) if path.exists() else {}

def blank_annotation(key, xyz, fps, caption):
    return dict(key=key, motion_hash=motion_hash(xyz), fps=fps, frames=len(xyz), caption=caption,
                reviewer='', reviewed=False, units_confirmed=False,
                floor_m=0.0, floor_verified=False, support_reviewed=False,
                support=[], required_support=[], airborne=[], labels_complete=False,
                labels=[], severity_notes='', feedback={'approved': False, 'decision': 'uncertain',
                'instruction': '', 'defects': []})

def error_record(work, operation, clip_id, exc):
    path = Path(work) / 'errors.jsonl'
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(time=now(), operation=operation, clip_id=clip_id,
                                     error_type=type(exc).__name__, error=str(exc))) + '\n')

def code_identity():
    paths = list((REPO / 'pilot').glob('*.py')) + [
        REPO / 'SFT/HumanML3D/models/mllm_single_lora.py',
        REPO / 'SFT/HumanML3D/dataset/data_processor.py',
        REPO / 'SFT/HumanML3D/models/vqvae.py']
    # Git may use CRLF on Windows and LF in Kaggle; hash logical source text.
    return {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
            for p in sorted(paths)}

def recover_xyz(features):
    """NumPy equivalent of upstream recover_from_ric; no re-flooring or rescaling."""
    f = np.asarray(features, dtype=np.float32)
    if f.ndim != 2 or f.shape[1] != 263 or len(f) < 2 or not np.isfinite(f).all():
        raise ValueError('Expected finite, unnormalised HumanML3D [T,263] features')
    angle = np.cumsum(np.r_[np.float32(0), f[:-1, 0]])
    c, s = np.cos(2 * angle), np.sin(2 * angle)
    def inverse_rotate(v):
        out = v.copy()
        cc, ss = c.reshape((-1,) + (1,) * (v.ndim - 2)), s.reshape((-1,) + (1,) * (v.ndim - 2))
        out[..., 0] = cc * v[..., 0] - ss * v[..., 2]
        out[..., 2] = ss * v[..., 0] + cc * v[..., 2]
        return out
    velocity = np.zeros((len(f), 3), dtype=np.float32)
    velocity[1:, 0], velocity[1:, 2] = f[:-1, 1], f[:-1, 2]
    root = np.cumsum(inverse_rotate(velocity), axis=0)
    root[:, 1] = f[:, 3]
    xyz = inverse_rotate(f[:, 4:67].reshape(-1, 21, 3))
    xyz[:, :, 0] += root[:, None, 0]
    xyz[:, :, 2] += root[:, None, 2]
    return np.concatenate([root[:, None], xyz], axis=1)
