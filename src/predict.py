"""Turn a trained checkpoint into a Kaggle submission file.

    python -m src.predict --config configs/cnn.yaml \
                          --checkpoint models/cnn_best.pt \
                          --out submissions/cnn_best.csv

The parts that define the *submission contract* are written out, because
getting them wrong wastes a submission. The model parts handle two kinds of
checkpoint:

* lab 4 on (CNN): the model is rebuilt with `build_model` from the config,
  input shape and loss stored in the checkpoint, and a photograph is
  prepared by `inputs.preprocess_one`, the same steps as the training table;
* lab 3 (MLP): rebuilt from its config, input size and standardiser.

Dry run on the validation unit, before the competition opens:

    python -m src.predict --config configs/lab4_cnn_768.yaml \
                          --checkpoint models/<run>.pt --dry-run \
                          --out data/work/dry_run.csv

predicts every validation photograph of the config's split from the
originals, one at a time, and prints the RMSE on the scoreable rows: it must
match the validation RMSE the training run logged for this checkpoint.

THE CONTRACT
------------
Output a CSV with exactly two columns and one row per test image:

    image_id,usability
    ph_00659482,0.8213
    ph_0112e28f,0.1044

* `image_id` is the test image filename **without its extension**, exactly as it
  appears in the competition's `sample_submission.csv`. Do not invent your own
  ids, do not reorder rows, and do not add an index column.
* `usability` is a float in [0, 1]. Clip it. A prediction of 1.3 is not a bolder
  guess, it is an error the metric will charge you for.
* Every id in `sample_submission.csv` must appear exactly once. A missing row is
  a rejected submission, not a zero.

THE RULE THAT SHAPES THIS FILE
------------------------------
Your prediction for an image must depend on **that image and your trained
weights only**. Not on other test images, not on any network call, not on an
OCR engine.

That is why the loop below runs one image at a time and why nothing is shared
between iterations. Batching for speed is fine, since a batch of independent
images still gives each its own answer. What is not fine is anything that pools
information across the test set. See section 2 of the project description.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src import config, losses
from src.data import inputs
from src.data.dataset import load_table, pixels_of_image
from src.data.splits import build_splits
from src.evaluate import rmse
from src.models.build import build_model
from src.models.mlp import MLP


def _check_same(trained: dict, cfg: dict, checkpoint: Path) -> None:
    """The config on the command line must describe the network in the checkpoint."""
    for section in ("model", "data"):
        for field, value in trained[section].items():
            if field in cfg[section] and cfg[section][field] != value:
                raise SystemExit(f"{checkpoint} was trained with {section}.{field}={value!r}, "
                                 f"the config says {cfg[section][field]!r}")


def load_model(checkpoint: Path, cfg: dict):
    """Rebuild the architecture from the config and load the saved weights.

    Put the model in eval mode and disable gradients. Return something callable
    that maps one preprocessed image to one float. The callable carries the
    checkpoint as `.checkpoint`, because preprocessing needs its standardiser.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)   # your own file
    _check_same(ckpt["config"], cfg, checkpoint)

    if "loss_state" in ckpt:                                   # lab 4 on: a CNN through build_model
        loss = losses.from_state(ckpt["loss_state"])
        model = build_model(ckpt["config"], tuple(ckpt["in_shape"]), loss)
        model.load_state_dict(ckpt["state_dict"])
        model.eval().requires_grad_(False)

        def run(x: torch.Tensor) -> float:
            with torch.no_grad():
                return float(loss.predict(model(x)).reshape(-1)[0])

    else:                                                      # lab 3: the MLP
        trained = ckpt["config"]
        model = MLP(in_dim=ckpt["in_dim"], hidden=list(trained["model"]["hidden"]),
                    dropout=trained["model"]["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval().requires_grad_(False)

        def run(x: torch.Tensor) -> float:
            with torch.no_grad():
                return float(model(x)[0])

    run.checkpoint = ckpt
    return run


def preprocess(image_path: Path, cfg: dict, checkpoint: dict):
    """Read one image and turn it into model input.

    This must match training exactly. A resize or a normalisation that differs
    between training and prediction is the single most common reason a good
    validation score turns into a bad leaderboard score. Both branches go
    through the bytes the 768 px cache would hold (`inputs.as_cached`).
    """
    if "loss_state" in checkpoint:
        return inputs.preprocess_one(image_path, cfg, checkpoint)
    x = pixels_of_image(inputs.as_cached(image_path), cfg["data"]["pixel_size"])
    x = (x - checkpoint["scaler_mean"]) / checkpoint["scaler_scale"]
    return torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)


def predict_one(model, image_path: Path, cfg: dict) -> float:
    """One image in, one number out."""
    return min(max(model(preprocess(image_path, cfg, model.checkpoint)), 0.0), 1.0)


def dry_run(model, cfg: dict) -> pd.DataFrame:
    """Predict the validation unit of the config's split from the originals; print the RMSE."""
    df = load_table()
    sp = cfg["split"]
    _, val_idx = build_splits(df, sp["group_column"], sp["val_fraction"], sp["seed"])
    val = df.iloc[val_idx]
    pred = np.array([predict_one(model, Path(p), cfg) for p in val["path"]])
    ok = (val["eval_ok"] == 1).to_numpy()
    score = rmse(val[config.TARGET].to_numpy()[ok], pred[ok])
    print(f"dry run: {len(val)} validation photographs, strict units "
          f"{sorted(int(u) for u in val['group_strict'].unique())}, "
          f"RMSE {score:.4f} on {ok.sum()} scoreable rows")
    logged = model.checkpoint.get("val_rmse")
    if logged is not None:
        print(f"the training run logged {logged:.4f} for this checkpoint (epoch {model.checkpoint.get('epoch')}): "
              f"difference {abs(score - logged):.1e}, {'OK' if abs(score - logged) < 1e-3 else 'MISMATCH'}")
    return pd.DataFrame({"image_id": val["image_id"].to_numpy(), "usability": pred})


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a Kaggle submission file")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="the weights that will produce this submission")
    ap.add_argument("--test-dir", default="data/raw/test",
                    help="directory of test images")
    ap.add_argument("--sample", default="data/raw/sample_submission.csv",
                    help="the competition's sample submission; defines ids and row order")
    ap.add_argument("--out", required=True, help="where to write the submission CSV")
    ap.add_argument("--dry-run", action="store_true",
                    help="predict the validation unit from data/raw/train instead of the test set")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)

    model = load_model(Path(a.checkpoint), cfg)

    if a.dry_run:
        out = dry_run(model, cfg)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
        print(f"wrote {len(out)} rows to {a.out} (validation, not a submission)")
        return

    # The sample submission is the source of truth for which ids exist and in
    # what order. Build the output from it rather than from a directory listing,
    # and the two can never disagree.
    sample = pd.read_csv(a.sample)
    if list(sample.columns) != ["image_id", "usability"]:
        raise SystemExit(f"unexpected sample columns: {list(sample.columns)}")

    test_dir = Path(a.test_dir)

    predictions = []
    for image_id in sample["image_id"]:
        image_path = test_dir / f"{image_id}.jpg"
        if not image_path.exists():
            raise SystemExit(f"missing test image: {image_path}")
        predictions.append(predict_one(model, image_path, cfg))

    out = pd.DataFrame({"image_id": sample["image_id"], "usability": predictions})
    out["usability"] = out["usability"].clip(0.0, 1.0)

    # Checks worth running before every upload. Each of these has cost somebody
    # a submission slot at some point.
    assert len(out) == len(sample), "row count does not match the sample submission"
    assert out["image_id"].is_unique, "duplicate image_id"
    assert out["usability"].notna().all(), "NaN in predictions"
    assert out["usability"].between(0, 1).all(), "prediction outside [0, 1]"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} rows to {a.out}")
    print(out["usability"].describe())
    print("\nRemember: the checkpoint that produced this file is part of your "
          "submission. Keep it, and record which run it came from.")


if __name__ == "__main__":
    main()
