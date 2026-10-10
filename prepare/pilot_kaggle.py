"""Reusable Kaggle setup. Run from the writable repository copy; no credentials saved."""
import argparse
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def setup(environment):
    python = environment / 'bin/python'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'uv'], check=True)
    uv = [sys.executable, '-m', 'uv']
    if not python.exists():
        subprocess.run(uv + ['venv', '--python', '3.11', str(environment)], check=True)
    subprocess.run(uv + ['pip', 'install', '--python', str(python), 'torch==2.5.1',
                        '--index-url', 'https://download.pytorch.org/whl/cu121'], check=True)
    subprocess.run(uv + ['pip', 'install', '--python', str(python), '-r', str(ROOT / 'requirements-baseline.txt'),
                        '-r', str(ROOT / 'requirements-pilot.txt')], check=True)
    subprocess.run(uv + ['pip', 'check', '--python', str(python)], check=True)

def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError('Unsafe archive path')
            if not target.exists():
                z.extract(entry, destination)

def assets(work):
    from huggingface_hub import HfApi, snapshot_download
    import gdown
    from pilot.common import read, write, file_hash
    work.mkdir(parents=True, exist_ok=True)
    revisions_path = work / 'asset_revisions.json'
    revisions = read(revisions_path) if revisions_path.exists() else {}
    sources = {
        'gemma': ('google/gemma-2-2b-it', ROOT / 'models/gemma-2-2b-it', ['*.json', '*.model', '*.safetensors']),
        'stage2': ('Lymann/IRG-MotionLLM-HumanML3D', ROOT / 'models/IRG-MotionLLM-HumanML3D', ['Stage-2/*'])
    }
    for label, (repo, destination, patterns) in sources.items():
        revision = revisions.get(label, {}).get('revision') or HfApi().model_info(repo).sha
        revisions[label] = dict(repo=repo, revision=revision)
        write(revisions_path, revisions)  # pin before download, including interrupted attempts
        snapshot_download(repo, revision=revision, local_dir=destination, allow_patterns=patterns, max_workers=1)
    vq = ROOT / 'models/motion-agent/ckpt/vqvae.pth'
    meta = ROOT / 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
    # Same public archives as the existing baseline; omit the unnecessary KIT archive.
    jobs = [
        (not vq.exists(), '1Tagt2xUwv_h0JNMtrM_Ty1rWemkLF5jH', ROOT / 'models/motion-agent/motion_agent.zip', ROOT / 'models/motion-agent'),
        (not all((meta / name).exists() for name in ('mean.npy', 'std.npy')), '1FIiqtkt4F-GVWmnBgtZnv9W3cPWS-oM-', ROOT / 'checkpoints/t2m.zip', ROOT / 'checkpoints')]
    for needed, identifier, archive, destination in jobs:
        if needed:
            archive.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(id=identifier, output=str(archive), resume=True, use_cookies=False)
            extract(archive, destination)
            revisions[archive.name] = {'google_drive_id': identifier, 'sha256': file_hash(archive)}
            write(revisions_path, revisions)
    for link, target in [(ROOT / 'SFT/HumanML3D/ckpt', vq.parent),
                         (ROOT / 'SFT/HumanML3D/checkpoints', ROOT / 'checkpoints')]:
        if link.exists() or link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f'Existing asset path points elsewhere: {link}')
        else:
            link.symlink_to(target, target_is_directory=True)
    for path in (vq, meta / 'mean.npy', meta / 'std.npy'):
        if not path.is_file():
            raise FileNotFoundError(path)
    print('Model assets ready. Revisions recorded; no secret values saved.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['setup', 'assets'])
    parser.add_argument('--environment', type=Path, default=Path('/tmp/irg-pilot-env'))
    parser.add_argument('--work', type=Path, default=Path('/kaggle/working/physical-pilot'))
    args = parser.parse_args()
    if args.command == 'setup':
        setup(args.environment)
    else:
        sys.path.insert(0, str(ROOT))
        assets(args.work)

if __name__ == '__main__':
    main()
