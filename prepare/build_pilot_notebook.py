"""Generate the thin Kaggle notebook; modules contain the implementation."""
import json
from pathlib import Path
import textwrap

cells = []
def cell(kind, source):
    entry = {'cell_type': kind, 'metadata': {}, 'source': textwrap.dedent(source).strip() + '\n'}
    if kind == 'code':
        entry.update(execution_count=None, outputs=[])
    cells.append(entry)

cell('markdown', '''
# Physical-defect correction pilot

Run cells in order. Start with **SMOKE = True**, one candidate and one round.
The code lives in `pilot/`; this notebook only configures and runs it.
See `pilot/README.md` and `pilot/MEASUREMENTS.md` for definitions and review instructions.

Attach the updated repository and previous baseline results as Kaggle inputs.
HumanML3D is optional for the initial five-clip workflow, required for the proposed original/corruption allocation.
Enable a GPU and Internet for model installation/downloads. Inputs under `/kaggle/input` are read-only;
the repository copy and environment use `/tmp`, results use `/kaggle/working`.
Keep a private/exported copy of every completed session before it expires.
''')
cell('markdown', '## 1. Locate the repository and inspect the actual GPU')
cell('code', '''
from pathlib import Path
import os, sys, json, shutil, subprocess

SOURCE = None  # Or Path('/kaggle/input/YOUR-REPO-DATASET/honours-motion-experiments')
if SOURCE is None:
    candidates = sorted({p.parent.parent for p in Path('/kaggle/input').rglob('pilot/__main__.py')})
    if len(candidates) != 1:
        raise RuntimeError(f'Set SOURCE to the updated repository root. Found: {candidates}')
    SOURCE = candidates[0]
REPO = Path('/tmp/irg-pilot-repo')
if not REPO.exists():
    shutil.copytree(SOURCE, REPO, ignore=shutil.ignore_patterns('.git', '.venv*', '__pycache__', 'outputs'))
else:
    for source_file in (SOURCE / 'pilot').rglob('*.py'):
        target = REPO / source_file.relative_to(SOURCE)
        if not target.exists() or target.read_bytes() != source_file.read_bytes():
            raise RuntimeError('Scratch code differs from attached source. Use a new scratch directory/session.')
WORK = Path('/kaggle/working/physical-pilot')
WORK.mkdir(parents=True, exist_ok=True)
ENV = os.environ.copy()
ENV.update(HF_HOME='/tmp/irg-pilot-hf', HF_HUB_DISABLE_XET='1', UV_CACHE_DIR='/tmp/irg-pilot-uv',
           TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8')
subprocess.run(['nvidia-smi'], check=True)
print('Scratch free GiB:', round(shutil.disk_usage('/tmp').free / 2**30, 1))
''')
cell('markdown', '## 2. Install the isolated Python environment')
cell('code', '''
PYTHON = Path('/tmp/irg-pilot-env/bin/python')
subprocess.run([sys.executable, str(REPO / 'prepare/pilot_kaggle.py'), 'setup'], cwd=REPO, env=ENV, check=True)
ENV['PATH'] = str(PYTHON.parent) + os.pathsep + ENV['PATH']
subprocess.run([str(PYTHON), '-c', "import torch; print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name() if torch.cuda.is_available() else None)"], env=ENV, check=True)
''')
cell('markdown', '''
## 3. Configure inputs and optional session restoration

Set `BASELINE` to a directory containing `run.json` and `000/first_263.npy`, etc.
For HumanML3D, fill the generated source-index CSV using official IDs/partitions and source recording families.
Feature paths in that CSV are relative to the CSV or absolute Kaggle input paths.
`RESUME_BUNDLE` can restore a prior exported pilot into an empty WORK directory.
Do not include a ZIP of the whole repository here.
''')
cell('code', '''
SMOKE = True
BASELINE = None  # Path('/kaggle/input/YOUR-RESULTS/five-prompts-stage2-seed42')
SOURCE_INDEX = None  # Path('/kaggle/input/YOUR-HUMANML3D/source_index.csv')
RESUME_BUNDLE = None  # Path('/kaggle/input/YOUR-PILOT/physical-pilot.zip')
if RESUME_BUNDLE:
    if any(WORK.iterdir()):
        raise RuntimeError('Restore only into an empty WORK directory.')
    sys.path.insert(0, str(REPO))
    from prepare.pilot_kaggle import extract
    extract(RESUME_BUNDLE, WORK)
if BASELINE is None and not (WORK / 'manifest.json').exists():
    matches = list(Path('/kaggle/input').rglob('five-prompts-stage2-seed42/run.json'))
    if len(matches) == 1:
        BASELINE = matches[0].parent
config = json.loads((REPO / ('pilot/configs/smoke.json' if SMOKE else 'pilot/configs/pilot.json')).read_text())
CONFIG = WORK / ('kaggle_smoke.json' if SMOKE else 'kaggle_full.json')
CONFIG.write_text(json.dumps(config, indent=2))
def pilot(*arguments, check=True):
    return subprocess.run([str(PYTHON), '-m', 'pilot', '--work', str(WORK), '--config', str(CONFIG),
                           *map(str, arguments)], cwd=REPO, env=ENV, check=check)
arguments = ['prepare']
if BASELINE: arguments += ['--baseline', BASELINE]
if SOURCE_INDEX: arguments += ['--source-index', SOURCE_INDEX]
pilot(*arguments)
''')
cell('markdown', '''
## 4. Obtain model assets if needed

Existing baseline assets in the attached repository are reused. Otherwise set `DOWNLOAD_MODELS=True`.
Enable the Kaggle Secret **HF_TOKEN**, whose value is a read token for an account with Gemma access.
The secret is passed in memory only. Downloads include Gemma, IRG Stage-2, the motion decoder and
pretrained normalisation statistics. No simulator or dataset is downloaded by this cell.
The setup helper records revisions; a restored pilot reuses those revisions.
''')
cell('code', '''
DOWNLOAD_MODELS = False
if DOWNLOAD_MODELS:
    from kaggle_secrets import UserSecretsClient
    ENV['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
    subprocess.run([str(PYTHON), str(REPO / 'prepare/pilot_kaggle.py'), 'assets', '--work', str(WORK)],
                   cwd=REPO, env=ENV, check=True)
''')
cell('markdown', '''
## 5. Reconstruct available original clips; optionally generate more candidates

`reconstruct` caches VQ tokens and saves decoded features/joints. Saved baseline clips already
have decoded tokens. `GENERATE_MORE=True` fills missing generated slots (two at a time in smoke mode).
Generation origin does not imply a physical failure. Review every candidate.
''')
cell('code', '''
pilot('reconstruct')
GENERATE_MORE = False
if GENERATE_MORE:
    pilot('generate')
pilot('measure')
pilot('review-export', '--output', WORK / 'review_initial')
pilot('report')
from IPython.display import FileLink, display
display(FileLink(str(WORK / 'review_initial/index.html')))
''')
cell('markdown', '''
## 6. Review locally, then import your decisions

Download/open the HTML page locally. It embeds all joint frames and has editable JSON and a download button.
Review raw originals before creating corruptions; review reconstructed `initial` clips for comparisons.
Record support, floor, labels, and explicitly approve/correct feedback for condition D.
Upload the downloaded JSON as a private Kaggle input, set REVIEW_JSON, and rerun this cell.
Drafts never become approvals automatically. For a later session, restore the exported bundle first.
''')
cell('code', '''
REVIEW_JSON = None  # Path('/kaggle/input/YOUR-REVIEW/reviewed_annotations.json')
if REVIEW_JSON:
    pilot('review-import', REVIEW_JSON)
pilot('corrupt')
pilot('reconstruct')
pilot('measure')
pilot('review-export', '--output', WORK / 'review_initial')
# Newly reconstructed corruptions need their own review before verified comparisons.
''')
cell('markdown', '''
## 7. Run the matched comparison

Run only after verifying model assets and reviewing the initial clips. Each condition starts with the same
initial tokens. Semantic feedback uses IRG's existing caption-alignment template. Rule feedback abstains when
required evidence is absent. Human feedback is blocked unless explicitly approved for that exact motion.
Use a new RUN_ID after changing configuration or previously completed input reviews.
Inspect `errors.jsonl` and `runs/RUN_ID/last_invocation.json` if a command reports errors.
''')
cell('code', '''
RUN_COMPARISON = False
RUN_ID = 'smoke-r1' if SMOKE else 'pilot-r1'
if RUN_COMPARISON:
    pilot('refine', '--run-id', RUN_ID)
''')
cell('markdown', '''
## 8. Export condition-hidden pairs and import outcome review

The pair page hides condition names and feedback. Left is before, right is after. Review whether the target
defect is resolved, new defects appear, and the requested action remains. Unknown judgments stay null.
Check after-motion support intervals again. Keep `private/*_mapping.json` away from the reviewer until import.
For an optional second round, set rounds=2 in a new configuration/run ID; human feedback on round-one
outputs must be reviewed separately. Refer to README for the staged workflow.
''')
cell('code', '''
PAIR_REVIEW = WORK / ('pairs_' + RUN_ID)
pilot('review-export', '--output', PAIR_REVIEW, '--mode', 'pairs', '--run-id', RUN_ID)
display(FileLink(str(PAIR_REVIEW / 'index.html')))
PAIR_JSON = None  # Path('/kaggle/input/YOUR-OUTCOME-REVIEW/reviewed_annotations.json')
if PAIR_JSON:
    pilot('review-import', PAIR_JSON, '--mapping', WORK / 'private' / (PAIR_REVIEW.name + '_mapping.json'))
''')
cell('markdown', '## 9. Report and export before ending the Kaggle session')
cell('code', '''
pilot('report', '--run-id', RUN_ID)
from datetime import datetime, timezone
EXPORT = Path('/kaggle/working') / ('physical-pilot-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '.zip')
pilot('bundle', '--output', EXPORT)
display(FileLink(str(EXPORT)))
display(FileLink(str(WORK / 'reports' / RUN_ID / 'report.md')))
''')

notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                     'language_info': {'name': 'python', 'version': '3.11'}}, nbformat=4, nbformat_minor=5)
for i, entry in enumerate(cells):
    entry['id'] = f'physical-pilot-{i:02d}'
target = Path(__file__).resolve().parents[1] / 'notebooks/physical_pilot.ipynb'
target.write_text(json.dumps(notebook, indent=1) + '\n', encoding='utf-8')
print(f'Wrote {len(cells)} cells to {target}')
