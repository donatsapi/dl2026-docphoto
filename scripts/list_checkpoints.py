"""List the checkpoints in models/: model type, input size, epoch, augmentation, validation RMSE.

    python scripts/list_checkpoints.py
"""
from __future__ import annotations

import glob

import torch


def main() -> None:
    for path in sorted(glob.glob("models/*.pt")):
        c = torch.load(path, map_location="cpu", weights_only=False)
        cfg = c["config"]
        train = cfg.get("train", {})
        rmse = c.get("val_rmse")
        print(f"{path:40s} | {cfg['model']['type']:11s} | px {cfg['data']['pixel_size']:4d} | "
              f"epoch {c.get('epoch')!s:4s} | augment {train.get('augment', '-')!s:5s} | "
              f"val RMSE {'-' if rmse is None else f'{rmse:.4f}'}")


if __name__ == "__main__":
    main()
