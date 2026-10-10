"""Record local CPU verification without claiming pretrained GPU inference."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pilot.common import read, write, file_hash, recover_xyz, now

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=ROOT / 'outputs/physical-pilot')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'outputs/irg-baseline-results/five-prompts-stage2-seed42')
    args = parser.parse_args()
    destination = args.work / 'verification'
    destination.mkdir(parents=True, exist_ok=True)
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'pilot/tests', '-v'],
                           cwd=ROOT, capture_output=True, text=True)
    (destination / 'unit_tests.txt').write_text(tests.stdout + tests.stderr, encoding='utf-8')
    if tests.returncode:
        print(tests.stdout + tests.stderr)
        raise SystemExit(tests.returncode)
    notebook = read(ROOT / 'notebooks/physical_pilot.ipynb')
    code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    for cell in code:
        ast.parse(''.join(cell['source']))
    baseline = []
    for folder in sorted(args.baseline.iterdir()):
        if not (folder / 'first_263.npy').exists():
            continue
        recovered = recover_xyz(np.load(folder / 'first_263.npy', allow_pickle=False))
        existing = np.load(folder / 'first_xyz.npy', allow_pickle=False)
        assert np.allclose(recovered, existing, atol=5e-5)
        trace = read(folder / 'trace.json')
        baseline.append(dict(clip=folder.name, frames=len(existing),
                             num_motion_blocks=trace['num_motion_blocks'],
                             first_final_identical=file_hash(folder / 'first_xyz.npy') == file_hash(folder / 'final_xyz.npy'),
                             recovery_max_abs_error=float(np.max(np.abs(recovered-existing)))))
    snapshot = args.work / 'baseline_source_hashes.json'
    unchanged = None
    if snapshot.exists():
        unchanged = all(file_hash(args.baseline / path) == checksum for path, checksum in read(snapshot).items())
        assert unchanged, 'Existing source baseline changed'
    hardware = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                              capture_output=True, text=True)
    result = dict(verified_at=now(), python=sys.version, tests_passed=True,
                  notebook_cells=len(notebook['cells']), notebook_code_cells_parsed=len(code),
                  baseline=baseline, baseline_source_files_unchanged=unchanged,
                  local_gpu=hardware.stdout.strip(), numpy=np.__version__,
                  torch_installed=importlib.util.find_spec('torch') is not None,
                  pretrained_gpu_inference_executed=False,
                  human_approvals_supplied=False)
    write(destination / 'verification.json', result)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
