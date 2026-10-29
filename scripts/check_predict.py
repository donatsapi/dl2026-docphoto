"""Check that src/predict.py gives the numbers the training loop gives.

    python scripts/check_predict.py --config configs/lab3_mlp_es.yaml --checkpoint models/lab3_mlp_es.pt

Takes twenty training photographs, predicts each one twice: once the way the
training loop does (`train.predict` on rows of the pixel table) and once the
way a submission is made (`predict.predict_one` on the original JPEG). The two
must agree to float precision; if they do not, prediction and training see
different inputs and the leaderboard will say so before you notice.

It also writes data/work/check_sample.csv, a sample submission for the same
twenty photographs, so the whole command line can be tried end to end:

    python -m src.predict --config ... --checkpoint ... \\
        --test-dir data/raw/train --sample data/work/check_sample.csv --out data/work/check_submission.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src import predict as submit
from src import train as trainer
from src.data.dataset import WORK, load_table, pixel_table
from src.models.mlp import MLP


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tolerance", type=float, default=1e-5)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))

    df = load_table()
    rows = np.sort(np.random.default_rng(0).choice(len(df), a.n, replace=False))
    sub = df.iloc[rows]

    # The training loop's way: the pixel table, the stored standardiser, train.predict.
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    model = MLP(in_dim=ckpt["in_dim"], hidden=list(ckpt["config"]["model"]["hidden"]),
                dropout=ckpt["config"]["model"]["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    X = pixel_table(df, cfg["data"]["pixel_size"])[rows]
    X = torch.as_tensor((X - ckpt["scaler_mean"]) / ckpt["scaler_scale"], dtype=torch.float32)
    in_loop = trainer.predict(model, X)

    # The submission's way: one original JPEG at a time.
    run = submit.load_model(Path(a.checkpoint), cfg)
    one_by_one = np.array([submit.predict_one(run, Path(p), cfg) for p in sub["path"]])

    out = pd.DataFrame({"image_id": sub["image_id"].to_numpy(), "usability": sub["usability"].to_numpy(),
                        "train.predict": in_loop, "predict_one": one_by_one,
                        "abs diff": np.abs(in_loop - one_by_one)})
    print(out.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    worst = float(out["abs diff"].max())
    print(f"\nlargest difference {worst:.2e}  (tolerance {a.tolerance:.0e})")

    WORK.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": sub["image_id"], "usability": 0.5}).to_csv(WORK / "check_sample.csv", index=False)
    print(f"sample submission for these {a.n} photographs: {WORK / 'check_sample.csv'}")
    if worst > a.tolerance:
        raise SystemExit("MISMATCH: predict.py does not reproduce the training loop")
    print("OK: predict.py reproduces the training loop")


if __name__ == "__main__":
    main()
