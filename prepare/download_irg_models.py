"""Download the public HumanML3D checkpoints; safely resume partial downloads."""
from pathlib import Path

from huggingface_hub import snapshot_download


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    snapshot_download(
        "Lymann/IRG-MotionLLM-HumanML3D",
        local_dir=root / "models" / "IRG-MotionLLM-HumanML3D",
        ignore_patterns=[".DS_Store", ".gitattributes"],
        max_workers=1,
    )
