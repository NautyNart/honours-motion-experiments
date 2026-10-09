"""Run the existing IRG inference engine on a small prompt list, without metrics.

Uses the same model defaults and loading order as test_unified.py. Paths supplied
on the CLI are resolved from the caller's directory; internal asset paths are
resolved from SFT/HumanML3D. See ../../BASELINE_GUIDE.md.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time


def render_gif(path, joints, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from utils.paramUtil import t2m_kinematic_chain

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    # HumanML3D uses Y for height. Display X/Z on the floor and Y vertically.
    lo, hi = joints.min(axis=(0, 1)), joints.max(axis=(0, 1))
    centre = (lo + hi) / 2
    radius = max(float((hi - lo).max()) / 2, 1.0)
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[2] - radius, centre[2] + radius)
    ax.set_zlim(centre[1] - radius, centre[1] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, wrap=True)
    lines = [ax.plot([], [], [], linewidth=2)[0] for _ in t2m_kinematic_chain]

    def update(frame):
        for line, chain in zip(lines, t2m_kinematic_chain):
            xyz = joints[frame, chain]
            line.set_data(xyz[:, 0], xyz[:, 2])
            line.set_3d_properties(xyz[:, 1])
        return lines

    FuncAnimation(fig, update, frames=len(joints), interval=50).save(
        path, writer=PillowWriter(fps=20)
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-backbone", required=True)
    parser.add_argument("--llm-ckpt", required=True, help="A full Stage-2/3 .pth checkpoint")
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true", help="Save first/final motion GIFs")
    cli = parser.parse_args()
    if cli.limit < 1:
        parser.error("--limit must be positive")
    prompt_file = cli.prompt_file.resolve()
    output = cli.output_dir.resolve()
    checkpoint = Path(cli.llm_ckpt).resolve()
    backbone = str(Path(cli.llm_backbone).resolve()) if Path(cli.llm_backbone).exists() else cli.llm_backbone
    prompts = [p.strip() for p in prompt_file.read_text(encoding="utf-8-sig").splitlines() if p.strip()][:cli.limit]
    if not prompts:
        parser.error("The prompt file is empty")
    if not checkpoint.is_file():
        parser.error(f"Checkpoint file does not exist: {checkpoint}")
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(Path(__file__).resolve().parent)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import numpy as np
    import torch
    from options.option_llm import get_args_parser
    from models.mllm_single_lora import MotionLLM
    from models.multi_round_inference import multi_round_inference_engine
    from dataset.data_processor import TextProcessor
    from utils.motion_process import recover_from_ric

    # Reuse all upstream architecture defaults instead of maintaining a copy.
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        args = get_args_parser()
    finally:
        sys.argv = original_argv
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this baseline. See BASELINE_GUIDE.md.")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("The upstream model uses bfloat16; use a GPU with native bfloat16 support.")
    for asset in (
        "ckpt/vqvae.pth",
        "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy",
        "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy",
    ):
        if not Path(asset).is_file():
            raise FileNotFoundError(f"Missing {Path(asset).resolve()}. See BASELINE_GUIDE.md.")
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)
    args.device = "cuda:0"
    args.llm_backbone = backbone
    args.eval_task = "unified_mogen_cot_v3"
    args.generation_mode = "think_w_analysis_multi_round_gen"
    args.prompt_w_response = True
    model = MotionLLM(args)
    model.need_normalize = False
    model.training_task = args.eval_task
    model.with_cot_forward = True
    model.generation_mode = args.generation_mode
    model.prompt_w_response = True
    # Public full checkpoints use the merged model layout, as in test_unified.py.
    # Load while the LLM is on CPU to avoid another full checkpoint in GPU RAM.
    model.activate_full_model()
    _, loaded, missing = model.load_model(str(checkpoint), verbose=True)
    if missing:
        raise RuntimeError(f"Checkpoint left {len(missing)} parameters unloaded: {sorted(missing)[:8]}")
    model = model.to(args.device).eval()
    processor = TextProcessor(args)
    manifest = {
        "checkpoint": str(checkpoint), "backbone": backbone, "seed": cli.seed,
        "task": args.eval_task, "do_sample": False, "prompts": prompts,
        "torch": torch.__version__, "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(), "loaded_parameters": len(loaded),
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with torch.inference_mode():
        for idx, prompt in enumerate(prompts):
            folder = output / f"{idx:03d}"
            folder.mkdir()
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = multi_round_inference_engine(
                model, processor, task_type=args.eval_task,
                input_info={"texts": [{"<Goal_Caption>": prompt}]},
                device=args.device, do_sample_=False, return_inter_motions=True,
            )[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            stages = result.get("think_motion_tokens", [])
            report = {
                "prompt": prompt, "input_text": result["input_text"],
                "generated_text": result["best_beam_text"], "generation_seconds": elapsed,
                "num_motion_blocks": result.get("num_gens"),
                "intermediate_tokens": [t.detach().cpu().tolist() for t in stages],
                "final_tokens": result["motion_tokens"].detach().cpu().tolist(),
            }
            (folder / "trace.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            # The upstream parser can fabricate tokens when motion tags are absent.
            # Reject that fallback so a malformed response is not a successful run.
            text = result["best_beam_text"]
            if "<Motion>" not in text or "</Motion>" not in text:
                raise RuntimeError(f"No complete motion block for prompt {idx}; inspect {folder / 'trace.json'}")
            selected = {"final": result["motion_tokens"]}
            if stages:
                selected["first"] = stages[0]
            for label, tokens in selected.items():
                if tokens.numel() == 0 or not ((tokens >= 0) & (tokens < args.nb_code)).all():
                    raise RuntimeError(f"Invalid {label} motion for prompt {idx}; inspect trace.json")
                normalized = model.net.forward_decoder(tokens.long())
                features = model.motion_denorm(normalized.float()).cpu()
                joints = recover_from_ric(features, 22).numpy()[0]
                if not np.isfinite(joints).all():
                    raise RuntimeError(f"Non-finite joint positions for prompt {idx}")
                np.save(folder / f"{label}_263.npy", features.numpy()[0])
                np.save(folder / f"{label}_xyz.npy", joints)
                if cli.render:
                    render_gif(folder / f"{label}.gif", joints, prompt)
            print(f"[{idx + 1}/{len(prompts)}] {prompt} ({elapsed:.1f}s)", flush=True)
            del result, stages, selected
    print(f"Saved baseline to {output}")


if __name__ == "__main__":
    main()
