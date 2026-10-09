"""Link downloaded assets into the working directories used by SFT and RL."""
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sources = {
        "glove": root / "glove",
        "checkpoints": root / "checkpoints",
        "ckpt": root / "models/motion-agent/ckpt",
    }
    for folder in ("SFT/HumanML3D", "RL/HumanML3D/mogen_r1/src"):
        for name, target in sources.items():
            if not target.is_dir():
                raise FileNotFoundError(f"Missing asset directory: {target}")
            link = root / folder / name
            if link.exists() or link.is_symlink():
                if link.resolve() != target.resolve():
                    raise FileExistsError(f"Preserving existing path: {link}")
                continue
            link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
            print(f"Linked {link.relative_to(root)}")


if __name__ == "__main__":
    main()
