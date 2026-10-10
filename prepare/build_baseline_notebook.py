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


cell("markdown", '''
# IRG-MotionLLM: first Kaggle baseline

Run five text prompts through a pretrained **Stage-2** model and compare its
first and final generated motions. Start with **one prompt** to verify model
loading, FP16 generation, motion decoding, and GIF rendering.

Run this notebook on **Kaggle**, from top to bottom, one cell at a time.
No HumanML3D reference motions or training annotations are needed for this pilot.
The motion decoder and pretrained normalization statistics are required.

Before running:

1. Import this `.ipynb` into a Kaggle notebook and select **GPU T4 ×2** and **Internet on**.
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

cell("markdown", '''
## 1. Check the Kaggle runtime

Expect T4 GPUs in the output and CUDA available. This checks the notebook's
preinstalled software; inference will use its own Python 3.11 environment.
''')
cell("code", '''
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

cell("markdown", '''
## 2. Load your repository source

Prefer attaching your updated source folder as a private Kaggle Dataset. The
cell will locate it under `/kaggle/input` and copy it to writable scratch space.
If there are multiple copies, set `SOURCE_REPO` to the correct folder.

Alternatively, set `REPO_URL` to your public repo's HTTPS clone URL. Commit and
push these notebook/precision changes first. Keep private Git credentials out
of URLs. To refresh an existing scratch copy, choose a new `REPO` directory.
''')
cell("code", '''
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
                        ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__"))
RUNNER = REPO / "SFT/HumanML3D/run_prompt_baseline.py"
for path in (RUNNER, REPO / "requirements-baseline.txt", REPO / "prepare/baseline_prompts.txt"):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Upload the complete updated source.")
if "--llm-dtype" not in RUNNER.read_text():
    raise RuntimeError("This source copy is outdated; upload the version with FP16 support.")
print("Repository:", REPO)
print("Results:", RESULTS_ROOT)
''')

cell("markdown", '''
## 3. Install an isolated inference environment

This can take several minutes. `uv` provides Python 3.11 and a separate
environment; the pinned NumPy version does not support Python 3.12+.
The notebook kernel's PyTorch/NumPy packages are left in place.
Inference is launched as a subprocess using the new environment.
The pinned CUDA 12.1 PyTorch wheel follows the
[official version instructions](https://pytorch.org/get-started/previous-versions/#v251).
''')
cell("code", '''
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

cell("markdown", '''
## 4. Enable Hugging Face access

Create and enable `HF_TOKEN` in Kaggle Secrets. Accept Gemma access with the
same account. The token is passed through the subprocess environment and is
not printed or saved to the notebook. Downloads will fail if Internet is off
or Gemma access has not been granted.
''')
cell("code", '''
from kaggle_secrets import UserSecretsClient
RUN_ENV["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
if not RUN_ENV["HF_TOKEN"]:
    raise RuntimeError("Enable a Kaggle Secret named HF_TOKEN.")
print("Hugging Face secret loaded.")
''')

cell("markdown", '''
## 5. Download Gemma and Stage-2

These are large downloads. Only Stage-2 is downloaded from the IRG repository.
The cell records model repository revisions alongside the results. Downloads
resume within the same session. Hugging Face's
[download documentation](https://huggingface.co/docs/huggingface_hub/guides/download)
explains the file filtering and local folders.
''')
cell("code", '''
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

cell("markdown", '''
## 6. Download the motion decoder/statistics and create asset links

The existing helpers download the Motion-Agent archive and evaluator archive.
Only the VQ decoder and pretrained statistics are loaded for this experiment.
The extractor helper also downloads KIT files; no KIT evaluation is run.
Scripts are normalized to LF in the scratch copy so Windows line endings do
not break Bash. Existing assets and links are preserved.
''')
cell("code", '''
VQ = REPO / "models/motion-agent/ckpt/vqvae.pth"
META = REPO / "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
jobs = [
    (not VQ.is_file(), "prepare/download_base_model.sh"),
    (not all((META / f).is_file() for f in ("mean.npy", "std.npy")), "prepare/download_extractor.sh"),
]
for needed, relative_script in jobs:
    if needed:
        script = REPO / relative_script
        script.write_text(script.read_text(), newline="\\n")
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

cell("markdown", '''
## 7. Select the checkpoint and prompts

If multiple `.pth` files are found, set `CHECKPOINT_CHOICE` to the full path
you intend to evaluate. You can edit the five prompts here before starting.
Keep the same prompts/settings when comparing checkpoints.
''')
cell("code", '''
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
    "A person walks forward, turns left, and stops.",
    "A person raises their right arm and waves.",
    "A person squats down and then stands up.",
    "A person takes two steps backwards.",
    "A person jumps once and lands on both feet.",
]
PROMPT_FILE = RESULTS_ROOT / "prompts.txt"
PROMPT_FILE.write_text("\\n".join(PROMPTS) + "\\n", encoding="utf-8")
SEED = 42
print("Selected:", CHECKPOINT)
print(PROMPT_FILE.read_text())
''')

cell("markdown", '''
## 8. Run ONE prompt first

Use `--llm-dtype float16` for the T4. The decoder remains float32, including
after loading the checkpoint. The runner saves `run.json`, per-prompt
`trace.json`, first/final joint arrays, and GIFs.

Output folders must be new: choose another folder name for a rerun instead
of overwriting results. If loading fails or generation runs out of memory,
stop here and inspect the traceback. Running fewer prompts does not reduce
the memory required for a single generation.
''')
cell("code", '''
SMOKE_DIR = RESULTS_ROOT / "smoke-stage2-seed42"

def run_baseline(output_dir, limit):
    if output_dir.exists():
        raise FileExistsError(f"Results already exist at {output_dir}; choose a fresh output folder.")
    subprocess.run([
        str(PYTHON), str(RUNNER),
        "--llm-backbone", str(REPO / "models/gemma-2-2b-it"),
        "--llm-ckpt", str(CHECKPOINT),
        "--prompt-file", str(PROMPT_FILE),
        "--output-dir", str(output_dir),
        "--llm-dtype", "float16",
        "--limit", str(limit), "--seed", str(SEED), "--render",
    ], env=RUN_ENV, check=True)

run_baseline(SMOKE_DIR, 1)
''')

cell("markdown", '''
## 9. Inspect the smoke test

Confirm both animations are viewable and depict a human motion. Read the
trace to see whether assessment/refinement occurred and whether the response
looks complete. A single motion block means no observable refinement.
The model's assessment is generated text, not an independent quality score.
''')
cell("code", '''
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

cell("markdown", '''
## 10. Run the five-prompt pilot after inspection

Set `RUN_FIVE_PROMPTS = True` only after the smoke test succeeds. This flag
prevents an initial Run All from launching the larger pilot before inspection.
The runner processes prompts sequentially on GPU 0 with greedy decoding.
''')
cell("code", '''
RUN_FIVE_PROMPTS = False
FIVE_DIR = RESULTS_ROOT / "five-prompts-stage2-seed42"
if RUN_FIVE_PROMPTS:
    run_baseline(FIVE_DIR, len(PROMPTS))
else:
    print("Five-prompt run skipped. Inspect the smoke test, set the flag to True, and rerun this cell.")
''')

cell("markdown", '''
## 11. Record observations and save results

Record whether the first/final motion matches the requested action, body
side, direction, and event order. Note foot sliding or unstable poses.
This compares motions within one model response, not independent direct and
refinement models. Five examples are a qualitative pilot, not benchmark metrics.

The following cell writes a CSV template; fill its assessment columns after
viewing the GIFs. Existing observations are preserved. Rerun the archive cell
after updating the CSV. Use Save Version to retain notebook outputs, and
download the ZIP from Kaggle's output files. Only results go into the ZIP;
model weights and the environment stay in session scratch space.
''')
cell("code", '''
import csv
evaluation_dir = FIVE_DIR if FIVE_DIR.exists() else SMOKE_DIR
rows = []
for file in sorted(evaluation_dir.glob("*/trace.json")):
    item = json.loads(file.read_text())
    complete = all((file.parent / f"{label}.gif").is_file() for label in ("first", "final"))
    rows.append({"sample": file.parent.name, "prompt": item["prompt"],
                 "motion_blocks": item["num_motion_blocks"], "outputs_complete": complete,
                 "generation_seconds": round(item["generation_seconds"], 2),
                 "first_matches": "", "final_matches": "", "better_same_worse": "", "notes": ""})
if not rows:
    raise RuntimeError("No generated traces found. Complete the smoke test first.")
observations = evaluation_dir / "observations.csv"
if not observations.exists():
    with observations.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
for row in rows:
    print(row["sample"], "complete:", row["outputs_complete"], row["prompt"])
print("Fill observations in:", observations)
''')
cell("code", '''
archive = shutil.make_archive("/kaggle/working/irg-baseline-results", "zip", RESULTS_ROOT)
print("Download this file from Kaggle output files:", archive)
''')

cell("markdown", '''
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
- **Interrupted run:** existing outputs are kept. Select a new output directory
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
