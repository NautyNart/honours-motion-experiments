# Local preparation on macOS

The Python 3.11 environment is installed in `.venv`. Activate it from this
repository's root with `source .venv/bin/activate`.

All packages in `requirements.txt` are installed. `pip check` passes, and
`train_unified_sft.py --help` and `test_unified.py --help` load successfully
from `SFT/HumanML3D`. This verifies imports and argument parsing, not model
inference or training. The supplied launchers require CUDA/NCCL, so this Mac
is being used to prepare files for a Linux NVIDIA GPU machine.

## Public downloads

GloVe and both evaluation extractor archives are downloaded and extracted.
GloVe lookup and the 263-dimensional HumanML3D normalization statistics were
checked successfully. A background job is downloading and extracting the
remaining model checkpoints. Check the live state from the repository root:

```bash
cat models/setup-download.status
tail -f models/setup-download.log
```

`COMPLETE` means the public downloads and links finished. `FAILED` means the
log contains an error. If the job has stopped, resume with:

```bash
bash prepare/download_public_assets.sh > models/setup-download.log 2>&1
```

Do not launch a second copy while the existing job is running. Its PID is
recorded in `models/setup-download.pid`; `ps -p "$(cat models/setup-download.pid)"`
checks whether it is still running. Keep the Mac awake and connected while
downloading.

Assets are stored in:

- `glove/`: evaluation word embeddings.
- `checkpoints/`: HumanML3D and KIT evaluation extractors.
- `models/motion-agent/`: original base checkpoint and VQ tokenizer.
- `models/IRG-MotionLLM-HumanML3D/Stage-1/`, `Stage-2/`, `Stage-3/`: public
  IRG-MotionLLM checkpoints (about 17 GB total).

The job links `glove`, `checkpoints`, and `ckpt` into the SFT and RL working
directories without overwriting existing paths. Keep these relative links
when copying the repository to the GPU machine. Recreate `.venv` there; a
macOS virtual environment cannot be reused on Linux.

## Remaining prerequisites

1. **Gemma access.** No Hugging Face login was found, and the backbone
   download returned `GatedRepoError`. Request/accept access on
   https://huggingface.co/google/gemma-2-2b-it, then run locally:

   ```bash
   source .venv/bin/activate
   export HF_HOME="$PWD/.cache/huggingface"
   hf auth login
   hf download google/gemma-2-2b-it --local-dir models/gemma-2-2b-it
   ```

   Enter the token at the terminal prompt. The model code expects the
   instruction-tuned `gemma-2-2b-it`, despite the upstream README linking
   to the base `gemma-2-2b` in one place.

2. **HumanML3D data.** Obtain/process the motion dataset using the official
   instructions at https://github.com/EricGuo5513/HumanML3D. The authors
   cannot redistribute the AMASS-derived motions directly. The SFT evaluator
   expects `SFT/HumanML3D/dataset/HumanML3D/` with `new_joint_vecs/`, `texts/`,
   split files, and normalization statistics. Full SFT/RL annotations must
   also be prepared; this repository only includes demonstration annotations.

3. **GPU execution configuration.** Set real dataset, backbone, checkpoint,
   and distributed-launch paths on the Linux GPU machine before launching.
   FlashAttention is required if using `--w-flash-attention`; it has not
   been installed on this Mac. The original Motion-Agent `t2m` LoRA must
   be converted to the `shared` adapter name before using it for IRG training,
   as described in the upstream README. Downloaded weights are preserved.

## Upstream gaps observed

- `prepare/download_base_model.sh` was a TODO and the README's
  `download_motionllm_ckpt.sh` was missing. Both now use the official
  Motion-Agent checkpoint source.
- `SFT/KIT-ML` and `RL/KIT-ML` are absent. The advertised KIT-ML Hugging Face
  repository currently contains only a README, with no weights.
- `SFT/HumanML3D/test_unified_m2t.py` is absent despite its README command.
- The stage-3 trainer imports `models.tmr_eval_wrapper` and
  `models.motionpatch_eval_wrapper`, which are absent from this checkout.
  Downloading dependencies alone cannot supply those project modules.

These gaps mean the repository is not yet verified as runnable end to end.
