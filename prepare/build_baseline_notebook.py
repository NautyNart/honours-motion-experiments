"""Build and statically validate the Kaggle starter notebook using stdlib only."""
import ast
import json
from pathlib import Path
from textwrap import dedent

cells = []


def cell(kind, source):
    item = {
        "cell_type": kind,
        "id": f"baseline-{len(cells):02d}",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }
    if kind == "code":
        ast.parse("".join(item["source"]))
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell("markdown", r'''
# IRG-MotionLLM: Kaggle prompt baseline

Run your chosen text prompts through a pretrained **Stage-2** model and compare its
first and final generated motions. Start with **one prompt** to verify model
loading, FP16 generation, motion decoding, and GIF rendering.

Run this notebook on **Kaggle**, from top to bottom, one cell at a time.
No HumanML3D reference motions or training annotations are needed for this pilot.
The motion decoder and pretrained normalization statistics are required.

Before running:

1. Import this `.ipynb` into Kaggle and enable a T4 GPU and Internet.
2. Attach the **updated repository source** as a private Kaggle Dataset using
   **Add Input**, or set a public Git clone URL in step 2. Include `SFT/`,
   `prepare/`, and `requirements-baseline.txt`. Importing this notebook alone
   does not upload the rest of your local repo. Do not upload a local virtual environment.
3. Accept access to [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it).
4. Add a Kaggle Secret named **HF_TOKEN**, holding a Hugging Face read token from
   that account, and enable it for this notebook. Keep the token out of cells.

This notebook uses **one T4**, with the LLM in **FP16** and the VQ decoder in
**FP32**. This is a compatibility configuration, not a verified reproduction of
the original BF16 experiment. T4 memory fit and numerical behavior must be
checked by the smoke test; a second T4 does not automatically increase memory
available to this process. FlashAttention is disabled.

Model files and the isolated Python environment are placed in `/tmp` for this
session. Results are saved under `/kaggle/working` and archived at the end.
Scratch files must be downloaded again after a new session.

References: [Kaggle notebooks](https://www.kaggle.com/docs/notebooks),
[GPU availability](https://www.kaggle.com/product-announcements/735239),
[IRG model weights](https://huggingface.co/Lymann/IRG-MotionLLM-HumanML3D).
''')

cell("markdown", r'''
## 1. Check the Kaggle runtime

Expect T4 GPUs in the output and CUDA available. This checks the notebook's
preinstalled software; inference will use its own Python 3.11 environment.
''')

cell("code", r'''
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

if not Path("/kaggle/working").is_dir():
    raise RuntimeError("Run this notebook in Kaggle with a GPU accelerator enabled.")
subprocess.run(["nvidia-smi"], check=True)
import torch
print("Notebook Python:", sys.version.split()[0])
print("Notebook PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU accelerator in Kaggle settings, then rerun.")
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index),
          round(torch.cuda.get_device_properties(index).total_memory / 2**30, 1), "GiB")
print("Scratch free space:", round(shutil.disk_usage("/tmp").free / 2**30, 1), "GiB")
''')

cell("markdown", r'''
## 2. Load your repository source

Prefer attaching your updated source folder as a private Kaggle Dataset. The
cell will locate it under `/kaggle/input` and copy it to writable scratch space.
If there are multiple copies, set `SOURCE_REPO` to the correct folder.

Alternatively, set `REPO_URL` to your public repo's HTTPS clone URL. Commit and
push these notebook/precision changes first. Keep private Git credentials out
of URLs. To refresh an existing scratch copy, choose a new `REPO` directory.
Reimporting the notebook does not replace an existing `/tmp` repository.
The cell automatically applies the FP32 decoding fix to recognised older
scratch runners and verifies it again before inference. Existing model files
are retained. The attached source dataset remains read only.
''')

cell("code", r'''
REPO_URL = ""  # Optional: public HTTPS clone URL of YOUR updated repo.
SOURCE_REPO = None  # Optional: Path("/kaggle/input/your-dataset/your-repo-folder")
REPO = Path("/tmp/honours-motion-experiments")
RESULTS_ROOT = Path("/kaggle/working/irg-baseline")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

if not REPO.exists():
    if REPO_URL:
        subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)
    else:
        if SOURCE_REPO is None:
            candidates = sorted({p.parents[2] for p in Path("/kaggle/input").rglob("run_prompt_baseline.py")
                                 if p.parent.name == "HumanML3D" and p.parent.parent.name == "SFT"})
            if len(candidates) != 1:
                raise RuntimeError(f"Attach your updated source repo, or set SOURCE_REPO. Candidates: {candidates}")
            SOURCE_REPO = candidates[0]
        shutil.copytree(SOURCE_REPO, REPO,
                        ignore=shutil.ignore_patterns(".git", ".venv*", "__pycache__", "outputs"))
RUNNER = REPO / "SFT/HumanML3D/run_prompt_baseline.py"
for path in (RUNNER, REPO / "requirements-baseline.txt"):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Upload the complete updated source.")
if "--llm-dtype" not in RUNNER.read_text():
    raise RuntimeError("This source copy is outdated; upload the version with FP16 support.")
def ensure_runner_fp32():
    # Patch only the writable scratch runner, including a cached /tmp copy.
    # NumPy statistics can promote decoded features to Float64. Cast both the
    # saved features (for known older runners) and the joint-recovery input.
    import ast

    source = RUNNER.read_text(encoding="utf-8")
    old_call = "recover_from_ric(features, 22)"
    fixed_call = "recover_from_ric(features.float(), 22)"
    if source.count(old_call) + source.count(fixed_call) != 1:
        raise RuntimeError(
            "Unrecognised joint-recovery code; attach the updated repository "
            "and choose a fresh REPO path. No automatic edit was made."
        )
    patched = source.replace(old_call, fixed_call)
    for old_features in (
        "features = model.motion_denorm(normalized.float()).cpu()",
        "features = model.motion_denorm(normalized).cpu()",
    ):
        patched = patched.replace(
            old_features, "features = model.motion_denorm(normalized.float()).float().cpu()"
        )
    ast.parse(patched)  # Validate before writing or launching the GPU process.
    if patched != source:
        RUNNER.write_text(patched, encoding="utf-8", newline="\n")
        print("Applied Float32 decoding compatibility fix to:", RUNNER)
    else:
        print("Float32 decoding compatibility fix verified.")

ensure_runner_fp32()
print("Repository:", REPO)
print("Results:", RESULTS_ROOT)
''')

cell("markdown", r'''
## 3. Install an isolated inference environment

This can take several minutes. `uv` provides Python 3.11 and a separate
environment; the pinned NumPy version does not support Python 3.12+.
The notebook kernel's PyTorch/NumPy packages are left in place.
Inference is launched as a subprocess using the new environment.
The pinned CUDA 12.1 PyTorch wheel follows the
[official version instructions](https://pytorch.org/get-started/previous-versions/#v251).
''')

cell("code", r'''
ENV_DIR = Path("/tmp/irg-baseline-env")
PYTHON = ENV_DIR / "bin/python"
RUN_ENV = os.environ.copy()
RUN_ENV.update({"UV_CACHE_DIR": "/tmp/irg-uv-cache", "UV_NO_CACHE": "1",
                "UV_PYTHON_INSTALL_DIR": "/tmp/irg-uv-python",
                "HF_HOME": "/tmp/irg-hf-cache", "HF_HUB_DISABLE_XET": "1",
                "CUDA_VISIBLE_DEVICES": "0", "TOKENIZERS_PARALLELISM": "false"})

subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "uv"], check=True)
uv = [sys.executable, "-m", "uv"]
if not PYTHON.exists():
    subprocess.run(uv + ["venv", "--python", "3.11", str(ENV_DIR)], env=RUN_ENV, check=True)
subprocess.run(uv + ["pip", "install", "--python", str(PYTHON), "torch==2.5.1",
                     "--index-url", "https://download.pytorch.org/whl/cu121"], env=RUN_ENV, check=True)
subprocess.run(uv + ["pip", "install", "--python", str(PYTHON), "-r",
                     str(REPO / "requirements-baseline.txt")], env=RUN_ENV, check=True)
subprocess.run(uv + ["pip", "check", "--python", str(PYTHON)], env=RUN_ENV, check=True)
RUN_ENV["PATH"] = str(PYTHON.parent) + os.pathsep + RUN_ENV["PATH"]
subprocess.run([str(PYTHON), str(RUNNER), "--help"], env=RUN_ENV, check=True)
gpu_check = "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name()); print('native bf16:', torch.cuda.is_bf16_supported(including_emulation=False))"
subprocess.run([str(PYTHON), "-c", gpu_check], env=RUN_ENV, check=True)
''')

cell("markdown", r'''
## 4. Enable Hugging Face access

Create and enable `HF_TOKEN` in Kaggle Secrets. Accept Gemma access with the
same account. The token is passed through the subprocess environment and is
not printed or saved to the notebook. Downloads will fail if Internet is off
or Gemma access has not been granted.
''')

cell("code", r'''
from kaggle_secrets import UserSecretsClient
RUN_ENV["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
if not RUN_ENV["HF_TOKEN"]:
    raise RuntimeError("Enable a Kaggle Secret named HF_TOKEN.")
print("Hugging Face secret loaded.")
''')

cell("markdown", r'''
## 5. Download Gemma and Stage-2

These are large downloads. Only Stage-2 is downloaded from the IRG repository.
The cell records model repository revisions alongside the results. Downloads
resume within the same session. Hugging Face's
[download documentation](https://huggingface.co/docs/huggingface_hub/guides/download)
explains the file filtering and local folders.
''')

cell("code", r'''
download_script = r"""
import json
from pathlib import Path
import sys
from huggingface_hub import HfApi, snapshot_download

repo, results = map(Path, sys.argv[1:])
sources = {
    "gemma": ("google/gemma-2-2b-it", repo / "models/gemma-2-2b-it", ["*.json", "*.model", "*.safetensors"]),
    "stage2": ("Lymann/IRG-MotionLLM-HumanML3D", repo / "models/IRG-MotionLLM-HumanML3D", ["Stage-2/*"]),
}
revisions = {}
for label, (name, destination, patterns) in sources.items():
    revision = HfApi().model_info(name).sha
    snapshot_download(name, revision=revision, local_dir=destination,
                      allow_patterns=patterns, max_workers=1)
    revisions[label] = {"repo": name, "revision": revision}
(results / "asset_revisions.json").write_text(json.dumps(revisions, indent=2))
"""
subprocess.run([str(PYTHON), "-c", download_script, str(REPO), str(RESULTS_ROOT)],
               env=RUN_ENV, check=True)
''')

cell("markdown", r'''
## 6. Download the motion decoder/statistics and create asset links

The existing helpers download the Motion-Agent archive and evaluator archive.
Only the VQ decoder and pretrained statistics are loaded for this experiment.
The extractor helper also downloads KIT files; no KIT evaluation is run.
Scripts are normalized to LF in the scratch copy so Windows line endings do
not break Bash. Existing assets and links are preserved.
''')

cell("code", r'''
VQ = REPO / "models/motion-agent/ckpt/vqvae.pth"
META = REPO / "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
jobs = [
    (not VQ.is_file(), "prepare/download_base_model.sh"),
    (not all((META / f).is_file() for f in ("mean.npy", "std.npy")), "prepare/download_extractor.sh"),
]
for needed, relative_script in jobs:
    if needed:
        script = REPO / relative_script
        script.write_text(script.read_text(), newline="\n")
        subprocess.run(["bash", str(script)], cwd=REPO, env=RUN_ENV, check=True)
for link, target in [
    (REPO / "SFT/HumanML3D/ckpt", VQ.parent),
    (REPO / "SFT/HumanML3D/checkpoints", REPO / "checkpoints"),
]:
    if not target.is_dir():
        raise FileNotFoundError(target)
    if link.exists() or link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"Inspect the existing path before proceeding: {link}")
    else:
        link.symlink_to(target, target_is_directory=True)
for file in (VQ, META / "mean.npy", META / "std.npy"):
    if not file.is_file():
        raise FileNotFoundError(file)
stats_check = "import numpy as np, sys; from pathlib import Path; p=Path(sys.argv[1]); m=np.load(p/'mean.npy'); s=np.load(p/'std.npy'); assert m.shape == s.shape == (263,); assert np.isfinite(m).all() and np.isfinite(s).all() and (s>0).all(); print('Statistics: OK')"
subprocess.run([str(PYTHON), "-c", stats_check, str(META)], env=RUN_ENV, check=True)
''')

cell("markdown", r'''
## 7. Configure the experiment and prompts

Edit `PROMPTS`, `RUN_LABEL`, and `SEED` here. Any positive number of single-line
prompts is supported. Seed 67 is the default; keep settings and prompt order
fixed when making comparisons. Greedy decoding can still produce the same
motion for different seeds.

Running this cell creates a new experiment ID with the label, seed, prompt
count, UTC time, and a short unique suffix. Rerun it to start a fresh experiment.
Do not change its settings between the smoke and batch runs. If multiple
checkpoints are listed, set `CHECKPOINT_CHOICE` explicitly.
''')

cell("code", r'''
CHECKPOINT_CHOICE = ""  # Only needed if multiple Stage-2 files are listed.
checkpoints = sorted((REPO / "models/IRG-MotionLLM-HumanML3D/Stage-2").rglob("*.pth"))
print("Stage-2 checkpoint files:")
for path in checkpoints:
    print(path)
if CHECKPOINT_CHOICE:
    CHECKPOINT = Path(CHECKPOINT_CHOICE)
elif len(checkpoints) == 1:
    CHECKPOINT = checkpoints[0]
else:
    raise RuntimeError("Set CHECKPOINT_CHOICE to a listed file; verify downloads if none are listed.")
if not CHECKPOINT.is_file():
    raise FileNotFoundError(CHECKPOINT)

PROMPTS = [
    "A person walks forward 2 steps, does a back-flip, then continues walking.",
    "A person lying flat on their back, holding their legs straight out wide",
    "A person doing Cristiano Ronaldo's iconic celebration (the SIUUU)",
    "A person walking backwards with their knees on the ground.",
    "A person hopping around like a bunny to the left first, then to the right and stops at one place but keeps hopping.",
]

from datetime import datetime, timezone
from uuid import uuid4
import re

RUN_LABEL = "prompt-baseline"  # Letters, digits, hyphens, or underscores.
SEED = 67
LLM_DTYPE = "float16"  # T4: FP16 language model; FP32 motion decoder.
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", RUN_LABEL):
    raise ValueError("Use a short RUN_LABEL with letters, digits, hyphens, or underscores.")
if not PROMPTS or any(not isinstance(p, str) or not p.strip() or "\n" in p or "\r" in p for p in PROMPTS):
    raise ValueError("PROMPTS must contain nonempty, single-line strings.")
PROMPTS = [p.strip() for p in PROMPTS]
RUN_ID = f"{RUN_LABEL}-seed{SEED}-n{len(PROMPTS)}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
EXPERIMENT_DIR = RESULTS_ROOT / RUN_ID
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)
PROMPT_FILE = EXPERIMENT_DIR / "prompts.txt"
PROMPT_FILE.write_text("\n".join(PROMPTS) + "\n", encoding="utf-8")
RUN_CONFIG = {"run_id": RUN_ID, "seed": SEED, "prompts": PROMPTS,
              "checkpoint": str(CHECKPOINT), "llm_dtype": LLM_DTYPE,
              "vq_dtype": "float32", "do_sample": False}
(EXPERIMENT_DIR / "experiment.json").write_text(json.dumps(RUN_CONFIG, indent=2), encoding="utf-8")
if (RESULTS_ROOT / "asset_revisions.json").is_file():
    shutil.copy2(RESULTS_ROOT / "asset_revisions.json", EXPERIMENT_DIR / "asset_revisions.json")
SMOKE_DIR = EXPERIMENT_DIR / "smoke"
BATCH_DIR = EXPERIMENT_DIR / "batch"
print("Experiment:", RUN_ID)
print("Selected checkpoint:", CHECKPOINT)
print("Prompts:", len(PROMPTS))
print("Results:", EXPERIMENT_DIR)
print(PROMPT_FILE.read_text())
''')

cell("markdown", r'''
## 8. Run one prompt first

The smoke test uses the first prompt, the configured seed and precision, and
saves into this experiment's `smoke/` folder. Each run includes its own prompt
file, runner metadata, generation log and completion status. Partial results
are retained after errors and can be exported for diagnosis.

Existing run folders are never overwritten. Rerun section 7 for a fresh
experiment. Fewer prompts reduce total runtime, not single-generation memory.
''')

cell("code", r'''
def run_baseline(output_dir, limit):
    ensure_runner_fp32()
    current = {"run_id": RUN_ID, "seed": SEED, "prompts": PROMPTS,
               "checkpoint": str(CHECKPOINT), "llm_dtype": LLM_DTYPE,
               "vq_dtype": "float32", "do_sample": False}
    if current != RUN_CONFIG or PROMPT_FILE.read_text(encoding="utf-8") != "\n".join(PROMPTS) + "\n":
        raise RuntimeError("Settings changed. Rerun section 7 to create a fresh experiment.")
    if output_dir.exists():
        raise FileExistsError(f"Results exist at {output_dir}. Rerun section 7 for a fresh experiment.")
    command = [
        str(PYTHON), "-u", str(RUNNER),
        "--llm-backbone", str(REPO / "models/gemma-2-2b-it"),
        "--llm-ckpt", str(CHECKPOINT),
        "--prompt-file", str(PROMPT_FILE),
        "--output-dir", str(output_dir),
        "--llm-dtype", LLM_DTYPE,
        "--limit", str(limit), "--seed", str(SEED), "--render",
    ]
    log_path = EXPERIMENT_DIR / f"{output_dir.name}.log"
    status_path = EXPERIMENT_DIR / f"{output_dir.name}_status.json"
    status = {"run": output_dir.name, "expected_prompts": limit, "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(), "command": command}
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    try:
        with log_path.open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, env=RUN_ENV, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    print(line, end="")
                    log.write(line)
                    log.flush()
                returncode = process.wait()
        status["returncode"] = returncode
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        status["status"] = "complete"
    except BaseException as exc:
        status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if output_dir.is_dir():
            shutil.copy2(PROMPT_FILE, output_dir / "prompts.txt")
            shutil.copy2(EXPERIMENT_DIR / "experiment.json", output_dir / "experiment.json")

run_baseline(SMOKE_DIR, 1)
''')

cell("markdown", r'''
## 9. Inspect the smoke test

Confirm both animations are viewable and depict a human motion. Read the
trace to see whether assessment/refinement occurred and whether the response
looks complete. A single motion block means no observable refinement.
The model's assessment is generated text, not an independent quality score.
''')

cell("code", r'''
from IPython.display import Image, Markdown, display

first_prompt = SMOKE_DIR / "000"
trace = json.loads((first_prompt / "trace.json").read_text())
print("Prompt:", trace["prompt"])
print("Motion blocks:", trace["num_motion_blocks"])
print("Generation seconds:", round(trace["generation_seconds"], 1))
for label in ("first", "final"):
    gif = first_prompt / f"{label}.gif"
    if not gif.is_file():
        raise FileNotFoundError(f"Incomplete smoke run: {gif}")
    display(Markdown(f"**{label.title()} motion**"))
    display(Image(filename=str(gif)))
print(trace["generated_text"])
''')

cell("markdown", r'''
## 10. Run all prompts after inspection

Set `RUN_BATCH = True` after inspecting the smoke test. The batch processes
all entries in `PROMPTS` sequentially on GPU 0 with greedy decoding and saves
to `batch/`. The smoke prompt is generated again as part of this full batch;
the smoke and batch are not independent experimental samples.
''')

cell("code", r'''
RUN_BATCH = False
if RUN_BATCH:
    run_baseline(BATCH_DIR, len(PROMPTS))
else:
    print("Batch skipped. Inspect the smoke test, set RUN_BATCH = True, and rerun this cell.")
''')

cell("markdown", r'''
## 11. Review and export this experiment

The next cell creates a browsable `index.html` with links to every prompt's
first/final GIFs and trace, plus separate observation CSVs for smoke and batch.
Incomplete or unstarted samples are listed explicitly. Existing assessment
columns are preserved when this cell is rerun. Review action, body side,
direction, event order, and physical defects. First/final motions are from
one model response, not independent models; this is a qualitative pilot.

Run the final export cell even after a failed generation to save partial
results and logs. It creates a uniquely named ZIP containing only this
experiment, with relative links, metadata, prompts, arrays, GIFs, checksums,
and transfer instructions. No models, environments, or credentials are added.
After editing observation CSVs, rerun both cells to refresh the export.
Download the ZIP using the displayed link and retain notebook outputs with
Save Version. Extract it on your PC under `outputs/irg-baseline-results/`
and open its `index.html`. For the physical pilot, explicitly pass the
extracted `batch/` directory to `pilot prepare --baseline`; the new names
are not recognised by the old five-prompt automatic discovery.
''')

cell("code", r'''
import csv
from html import escape
from IPython.display import FileLink, display

cards = []
for run_dir, expected in ((SMOKE_DIR, 1), (BATCH_DIR, len(RUN_CONFIG["prompts"]))):
    status_file = EXPERIMENT_DIR / f"{run_dir.name}_status.json"
    if not run_dir.exists() and not status_file.exists():
        continue
    run_dir.mkdir(exist_ok=True)
    observations = run_dir / "observations.csv"
    previous = {}
    if observations.exists():
        with observations.open(newline="", encoding="utf-8") as handle:
            previous = {row["sample"]: row for row in csv.DictReader(handle)}
    rows = []
    for index, prompt in enumerate(RUN_CONFIG["prompts"][:expected]):
        sample = f"{index:03d}"
        sample_dir = run_dir / sample
        trace_file = sample_dir / "trace.json"
        item = json.loads(trace_file.read_text(encoding="utf-8")) if trace_file.exists() else {}
        required = ["trace.json", "first.gif", "final.gif", "first_263.npy", "final_263.npy", "first_xyz.npy", "final_xyz.npy"]
        complete = all((sample_dir / name).is_file() for name in required)
        row = {"sample": sample, "prompt": prompt, "motion_blocks": item.get("num_motion_blocks", ""),
               "outputs_complete": complete, "generation_seconds": item.get("generation_seconds", ""),
               "first_matches": "", "final_matches": "", "better_same_worse": "", "notes": ""}
        for field in ("first_matches", "final_matches", "better_same_worse", "notes"):
            row[field] = previous.get(sample, {}).get(field, "")
        rows.append(row)
        links = []
        for name in required:
            if (sample_dir / name).is_file():
                relative = f"{run_dir.name}/{sample}/{name}"
                links.append(f'<a href="{relative}">{name}</a>')
        animations = []
        for label in ("first", "final"):
            if (sample_dir / f"{label}.gif").is_file():
                animations.append(f'<figure><figcaption>{label.title()}</figcaption><img loading="lazy" src="{run_dir.name}/{sample}/{label}.gif" alt="{label} motion"></figure>')
        state = "Complete" if complete else "Incomplete / not generated"
        cards.append(f'<article><h2>{run_dir.name} / {sample}: {escape(prompt)}</h2><p>{state}</p><div class="motions">{"".join(animations)}</div><p>{" | ".join(links)}</p></article>')
    with observations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(run_dir.name, f"{sum(row['outputs_complete'] for row in rows)}/{expected} complete", observations)
if not cards:
    raise RuntimeError("Run the smoke test first, or use the export cell to save setup metadata.")
page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>IRG baseline results</title><style>body{font-family:system-ui;margin:2rem;max-width:1100px}article{border-top:1px solid #bbb;padding:1rem 0}.motions{display:flex;flex-wrap:wrap}figure{margin:0 1rem 1rem 0}img{width:360px;max-width:100%}a{overflow-wrap:anywhere}</style><h1>' + escape(RUN_ID) + '</h1><p>First and final motions within each model response. Open observations.csv to record your assessment.</p>' + "".join(cards) + '</html>'
(EXPERIMENT_DIR / "index.html").write_text(page, encoding="utf-8")
display(FileLink(str(EXPERIMENT_DIR / "index.html")))
''')

cell("code", r'''
import hashlib
from zipfile import ZipFile
from IPython.display import FileLink, display

transfer_notes = f"""IRG baseline experiment: {RUN_ID}

Extract this ZIP into outputs/irg-baseline-results/ in your local repository.
Open {RUN_ID}/index.html in a browser for first/final animations and file links
(if present). Each run's observations.csv is editable; traces and motion arrays
are kept beside each prompt. Failed runs may contain partial results only.

experiment.json records the configured prompts, seed and checkpoint.
smoke_status.json / batch_status.json record execution status and commands.
smoke.log / batch.log record model loading, warnings and generation output.
run.json inside each generated run contains actual model/runtime metadata.
asset_revisions.json records downloaded Hugging Face model revisions if available.
SHA256SUMS.json lists file hashes relative to this experiment directory.
Absolute model paths are provenance; model weights are not included.

To import the full batch into a NEW physical-pilot work directory, run from
the repository root after extraction (replace NEW-WORK with a new folder):
python -m pilot --work outputs/NEW-WORK prepare --baseline outputs/irg-baseline-results/{RUN_ID}/batch
Review imported motions before running physical correction comparisons.
"""
(EXPERIMENT_DIR / "TRANSFER_README.txt").write_text(transfer_notes, encoding="utf-8")
checksums = {}
for file in sorted(EXPERIMENT_DIR.rglob("*")):
    if file.is_file() and file.name != "SHA256SUMS.json":
        checksums[file.relative_to(EXPERIMENT_DIR).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
(EXPERIMENT_DIR / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
export_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
archive_base = Path("/kaggle/working") / f"{RUN_ID}-export-{export_id}"
archive = shutil.make_archive(str(archive_base), "zip", root_dir=EXPERIMENT_DIR.parent, base_dir=EXPERIMENT_DIR.name)
with ZipFile(archive) as bundle:
    if bundle.testzip() is not None:
        raise RuntimeError("ZIP integrity check failed.")
print("Experiment:", EXPERIMENT_DIR)
print("Archive:", archive)
print("Size MiB:", round(Path(archive).stat().st_size / 2**20, 2))
display(FileLink(archive))
''')

cell("markdown", r'''
## Validation and troubleshooting

Notebook JSON, every Python cell, and embedded subprocess scripts were
statically checked locally. The runner's CLI and precision selection were
checked locally without downloading models. Full Kaggle inference, memory
fit, FP16 numerical behavior, and GIF rendering require the smoke test above.

- **No repository found:** attach your complete updated source, or set `REPO_URL`.
- **Gemma access error:** enable Internet, grant the notebook `HF_TOKEN`, and
  accept access using the matching Hugging Face account.
- **No `--llm-dtype` flag:** your attached/cloned source predates this notebook update.
- **Out of memory:** stop other GPU work or use a larger-memory GPU. This runner
  already uses one prompt at a time and does not pool both T4 memories.
- **Missing checkpoint parameters or malformed motion:** inspect the trace/error;
  verify the backbone/checkpoint selection before interpreting any result.
- **Float/Double error in joint recovery:** this notebook automatically fixes
  recognised older scratch runners in section 2 and before each inference run.
  Upload this notebook version and rerun section 2, then section 7 for a fresh
  experiment and section 8. Existing model downloads can be reused.
- **Interrupted run:** existing outputs are kept. Rerun section 7 to create a new experiment
  for a retry; a trace alone does not indicate a completed result.
''')

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
root = Path(__file__).resolve().parents[1]
destination = root / "notebooks/baseline.ipynb"
destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8", newline="\n")
loaded = json.loads(destination.read_text(encoding="utf-8"))
embedded = {"download_script", "stats_check", "gpu_check"}
for item in loaded["cells"]:
    if item["cell_type"] == "code":
        tree = ast.parse("".join(item["source"]))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in embedded for t in node.targets):
                ast.parse(ast.literal_eval(node.value))
assert len({item["id"] for item in loaded["cells"]}) == len(loaded["cells"])
assert all(not c.get("outputs") and c.get("execution_count") is None for c in loaded["cells"] if c["cell_type"] == "code")
print(f"Created and statically validated {destination}: {len(cells)} cells")
