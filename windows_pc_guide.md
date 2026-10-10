# First IRG-MotionLLM baseline

For Kaggle T4 GPUs, use [notebooks/baseline.ipynb](notebooks/baseline.ipynb).
It includes an isolated Python 3.11 environment and an FP16 inference path via
`--llm-dtype float16`; the VQ decoder remains FP32. The Linux workstation
instructions below use the original BF16 configuration. Kaggle inference and
FP16 numerical behavior still require the notebook's one-prompt smoke test.

## What is present in this checkout

Checked on 10 October 2026 (Australia/Sydney):

- Repository: `C:\Users\User\OneDrive\Desktop\Honours\honours-motion-experiments`.
- SFT inference and training code, RL code, download helpers, and demo annotations exist.
- No `.venv`, root `models/`, `checkpoints/`, `glove/`, or processed HumanML3D data exist here.
- Windows reports an NVIDIA GeForce RTX 2070 SUPER, 8 GB VRAM.
- Python 3.11.5 exists at `C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe`.
  The Windows launcher does not discover it in this session, and `python` resolves first to a WindowsApps alias.
  The required ML packages are not installed in that Python 3.11 environment.
- WSL is available as a command, but has no installed Linux distribution.
- `LOCAL_SETUP.md` describes an earlier macOS environment. Its claims about installed packages/downloads do not describe this checkout.

The existing `test_unified.py` is a benchmark evaluator. It hardcodes the
author's HumanML3D path, drops incomplete batches, and calculates metrics that
cannot finish on five examples. Its motion-saving code is commented out.
It also uses NCCL. Use the new prompt runner for the initial experiment.

## 1. Define the experiment

Run five prompts through a pretrained Stage-2 model, one prompt at a time,
with greedy decoding and seed 42. Compare the first generated motion with the
last generated motion from each interleaved generation/assessment/refinement
response. Save both motions and the complete text trace.

This tests inference and the effect of refinement within the same model.
It is a qualitative pilot, not a reproduction of the paper's benchmark or
a comparison against an independently trained direct-generation model.
The model may produce only one motion block; report that case rather than
assuming refinement happened or improved the result.

No SFT/RL training, annotation generation, or reference motion dataset is
required for this experiment. The VQ decoder and pretrained normalization
statistics are still required to turn motion tokens into joint positions.

## 2. Choose the execution machine

Use a Linux machine with an NVIDIA GPU that supports native bfloat16.
A 24 GB GPU is a practical starting target; this is a planning recommendation,
not a measured minimum. Allow about 32 GB system RAM and 40 GB free disk for
dependencies, download archives, model weights, caches, and outputs.
The upstream engine retains generation scores, so its memory use is larger
than the model weights alone.

Your RTX 2070 SUPER is a Turing GPU and does not have native bfloat16 support.
The upstream model explicitly requests bfloat16. Running on your card would
require a separate compatibility and memory optimization task; installing
WSL alone does not resolve those issues. For the first baseline, use an
available university Linux GPU workstation/server or another suitable GPU
machine. No paid provider is required by these instructions.

Copy this entire repository, including the new files, to that machine.
Recreate the environment there. In the commands below, `~/honours-motion-experiments`
is the destination; adjust it to your actual destination.
Every remaining shell command is for **Linux Bash**, not Windows PowerShell.

## 3. Create a small inference environment

Use Python 3.11, supplied by your GPU machine's Python installation or Conda:

```bash
cd ~/honours-motion-experiments
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-baseline.txt
python -m pip check
```

The CUDA 12.1 wheel is one of the official installation choices for the
repo's pinned PyTorch version. The machine's NVIDIA driver must support it.
See the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/#v251).
Use a GPU supported by this PyTorch build; a much newer GPU may require a
newer CUDA/PyTorch build and separate compatibility verification.

`requirements-baseline.txt` contains the dependencies used by the prompt
runner. The full `requirements.txt` also includes training, DeepSpeed, and
evaluation packages; these are unnecessary for this pilot. FlashAttention
is not enabled. The new dependency subset has not been installed or tested
in this Windows checkout.

Check the GPU from inside the environment:

```bash
nvidia-smi
python -c 'import torch; print("torch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name()); print("native bf16:", torch.cuda.is_bf16_supported(including_emulation=False))'
```

Continue when CUDA and native bf16 both report `True`.

## 4. Download Gemma, the VQ decoder, and Stage-2 weights

First request/accept access to
[google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)
in your browser. Use the instruction-tuned `-it` model expected by the code.
Authenticate at the terminal prompt; keep your token out of scripts.

Run from the repository root:

```bash
export HF_HOME="$PWD/.cache/huggingface"
hf auth login
hf download google/gemma-2-2b-it --local-dir models/gemma-2-2b-it
bash prepare/download_base_model.sh
bash prepare/download_extractor.sh
hf download Lymann/IRG-MotionLLM-HumanML3D --include 'Stage-2/*' --local-dir models/IRG-MotionLLM-HumanML3D
```

The base archive supplies `vqvae.pth`. The extractor archive supplies the
normalization statistics the model hardcodes. Its evaluation networks are
downloaded too, but the prompt runner does not load them. The existing helper
downloads both HumanML3D and KIT extractors; only HumanML3D is used here.
The Stage-2 download is filtered so you do not also download Stage-1 and
Stage-3. The checkpoint source is the repo's
[published model repository](https://huggingface.co/Lymann/IRG-MotionLLM-HumanML3D).

Create just the two asset links required by the prompt runner:

```bash
ln -s ../../models/motion-agent/ckpt SFT/HumanML3D/ckpt
ln -s ../../checkpoints SFT/HumanML3D/checkpoints
```

Run those link commands once on the fresh copy. If a link already exists,
inspect it with `ls -l SFT/HumanML3D/ckpt SFT/HumanML3D/checkpoints` instead
of replacing it. The existing `prepare/link_assets.py` also requires GloVe;
the two links above avoid that extra download for prompt-only inference.

Check the actual files:

```bash
ls -l SFT/HumanML3D/ckpt/vqvae.pth
ls -l SFT/HumanML3D/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy
ls -l SFT/HumanML3D/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy
find models/IRG-MotionLLM-HumanML3D/Stage-2 -type f -name '*.pth'
```

Set the variable below to the **actual Stage-2 checkpoint file printed by
`find`**, not to the Stage-2 directory. A filename is deliberately not assumed:

```bash
export IRG_STAGE2_CKPT='/absolute/path/to/the/actual/Stage-2/checkpoint.pth'
test -f "$IRG_STAGE2_CKPT"
```

Check the statistics without recomputing them:

```bash
python -c 'import numpy as np; from pathlib import Path; p=Path("checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"); m=np.load(p/"mean.npy"); s=np.load(p/"std.npy"); print(m.shape, s.shape); assert m.shape == s.shape == (263,); assert np.isfinite(m).all() and np.isfinite(s).all() and (s > 0).all()'
```

The statistics must match the pretrained model. Re-estimating them on five
clips changes the motion representation and invalidates the comparison.

## 5. Run a single prompt first

The supplied `prepare/baseline_prompts.txt` has five simple motion prompts.
Check imports and argument parsing, then generate only its first prompt:

```bash
python SFT/HumanML3D/run_prompt_baseline.py --help
python SFT/HumanML3D/run_prompt_baseline.py \
  --llm-backbone "$PWD/models/gemma-2-2b-it" \
  --llm-ckpt "$IRG_STAGE2_CKPT" \
  --prompt-file prepare/baseline_prompts.txt \
  --output-dir outputs/smoke-stage2-seed42 \
  --limit 1 --seed 42 --render
```

The runner uses the upstream `MotionLLM`, `TextProcessor`, and
`multi_round_inference_engine`, with task `unified_mogen_cot_v3`.
It follows the evaluator's merge-before-load behavior for full checkpoints.
Checkpoint loading happens while the LLM is on CPU to reduce temporary GPU
memory pressure. It stops if checkpoint parameters are missing or the output
does not contain a complete motion block. It does not substitute a dummy motion.

`--help` tests argument parsing only. A completed command with finite joint
arrays and an inspectable GIF is the first actual inference check.

## 6. Run all five prompts

After the one-prompt run works:

```bash
python SFT/HumanML3D/run_prompt_baseline.py \
  --llm-backbone "$PWD/models/gemma-2-2b-it" \
  --llm-ckpt "$IRG_STAGE2_CKPT" \
  --prompt-file prepare/baseline_prompts.txt \
  --output-dir outputs/baseline-stage2-seed42 \
  --limit 5 --seed 42 --render
```

Use a new output directory for each run. Existing output directories are
rejected to preserve previous results. All CLI paths are resolved from the
directory where you run the command. Internal upstream asset paths are
resolved from `SFT/HumanML3D` by the runner.

For each prompt, expect:

```text
outputs/baseline-stage2-seed42/
  run.json
  000/
    trace.json
    first_263.npy
    first_xyz.npy
    first.gif
    final_263.npy
    final_xyz.npy
    final.gif
  001/
  ...
```

`run.json` records prompts, checkpoint path, seed, GPU, and selected library
versions. `trace.json` records generated text, motion token sequences, motion
block count, and generation time (excluding GIF rendering). The `*_263.npy`
arrays are denormalized HumanML3D features, shaped `(frames, 263)`. The
`*_xyz.npy` arrays are recovered positions shaped `(frames, 22, 3)`.
GIFs use the HumanML3D rate of 20 fps. Intermediate tokens for all motion
blocks are retained even though only first/final motions are rendered.

The upstream v3 engine generates an interleaved response in one LLM call;
it is not an external Python assessment loop. Do not assume that
`--max-refine-round` controls this task: the ordinary v3 branch does not
use that argument. The model determines how many motion blocks it emits
within the upstream generation length limit. Inspect the end of each trace
for signs of truncation or an unfinished assessment.

## 7. Record the baseline result

View each first/final GIF pair and read the corresponding trace. Record:

| Prompt | Motion blocks | First motion matches? | Final motion matches? | Better/same/worse | Visible issue |
| --- | --- | --- | --- | --- | --- |
| Walk and turn left | | | | | |
| Wave right arm | | | | | |
| Squat and stand | | | | | |
| Two backward steps | | | | | |
| Jump and land | | | | | |

Check action, body side, direction, event order, foot sliding, and obvious
instability. Treat the model's own assessment as generated text, not ground
truth. Count a single-block response as having no observable refinement.
Do not infer a general improvement from five examples. Seed 42 and greedy
decoding establish the initial configuration; exact reproducibility across
different hardware/software is not guaranteed.

Keep outputs, the prompt file, and your observations together. A later
comparison can use a different checkpoint with the same prompts/settings.

## 8. Optional: use five real HumanML3D examples

If you already have processed HumanML3D, select five valid IDs from its
official `test.txt`. Copy their matching `new_joint_vecs/<id>.npy` and
`texts/<id>.txt` into `data/HumanML3D-mini/` and write those IDs to its
`test.txt`. Choose `(T, 263)` motion arrays with 40--196 frames. Keep original
caption/POS/time-tag lines, and retain the pretrained normalization statistics.
The mini folder then contains `test.txt`, `new_joint_vecs/`, and `texts/`.

To run their captions through the prompt runner, create a text file from one
full-motion caption per ID (the final two fields are both zero):

```bash
python - <<'PY'
from pathlib import Path
root = Path("data/HumanML3D-mini")
prompts = []
for name in (root / "test.txt").read_text().splitlines():
    if not name.strip():
        continue
    for line in (root / "texts" / (name.strip() + ".txt")).read_text().splitlines():
        fields = line.split("#")
        if len(fields) == 4 and float(fields[2]) == float(fields[3]) == 0:
            prompts.append(fields[0])
            break
    else:
        raise ValueError(f"No full-motion caption for {name}; choose a different ID")
Path("prepare/humanml_mini_prompts.txt").write_text("\n".join(prompts) + "\n")
PY
```

Pass `--prompt-file prepare/humanml_mini_prompts.txt` and a fresh output
directory to the same runner. It reads only captions; reference motion
arrays can be inspected separately. Ground-truth motions do not condition
this text-to-motion generation. This route avoids changes to the benchmark
loader and avoids its small-dataset metric failures.

If you do not have processed motions, use the
[official HumanML3D preparation instructions](https://github.com/EricGuo5513/HumanML3D#how-to-obtain-the-data).
The authors provide reconstruction code because they cannot directly
redistribute AMASS-derived motion data. Their sequence is
`raw_pose_processing.ipynb`, `motion_representation.ipynb`, then
`cal_mean_variance.ipynb`, using the required source data and body models.
That is a separate data preparation task; custom prompts let you begin the
baseline without waiting for it.

## If a step fails

- **Gemma access denied:** accept access to the exact `gemma-2-2b-it` repo and
  authenticate with the same Hugging Face account.
- **Missing VQ/statistics:** check extraction results and the two symlink targets.
- **Missing checkpoint parameters:** confirm you selected a full Stage-2/3
  checkpoint from the published repo. Do not silently run partially loaded weights.
- **CUDA out of memory:** the runner already uses one prompt at a time;
  stop other GPU jobs or use more GPU memory. Reducing the number of prompts
  does not reduce the memory needed for one generation.
- **No native bf16:** use a suitable GPU. The local RTX 2070 SUPER is outside
  this runner's supported configuration.
- **No complete motion tags:** inspect the saved trace. Check checkpoint and
  backbone compatibility before treating any returned tokens as motion.

The runner and guide were added in this Windows checkout. Python 3.11 syntax
compilation and the runner's `--help` command passed. The inference dependency
set, model weights, checkpoint loading, and GIF rendering could not be tested
here because the required ML packages and assets are absent. Treat the
one-prompt smoke run on a suitable GPU as the required first inference
validation before interpreting results.
