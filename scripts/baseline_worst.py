"""The baseline's worst validation photographs, drawn at full resolution.

    python scripts/baseline_worst.py            # 12 worst, into data/work/baseline_worst/

Fits the seven-feature baseline on the training units, predicts the scoreable
validation rows, and for the k largest errors writes the 2048 px original with
the 4 x 4 grid of the tile features drawn on it. Each tile carries its
Laplacian variance (L) and dark fraction (D) as the features see them at
768 px; the tile `tile_lapvar_min` picked is outlined in red. Open them at
100 % zoom: the question for each one is what is wrong with the page that
none of the seven numbers in `worst.csv` moved for.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import yaml

from src import config, evaluate
from src.data.dataset import WORK, cache_dir, load_table
from src.data.splits import build_splits
from src.features import handcrafted
from src.models import baseline


def tile_values(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = handcrafted.GRID
    lap = np.array([cv2.Laplacian(t, cv2.CV_64F).var() for t in handcrafted._tiles(gray)]).reshape(g, g)
    dark = np.array([(t < 64).mean() for t in handcrafted._tiles(gray)]).reshape(g, g)
    return lap, dark


def draw(image_id: str, path: str, title: str) -> np.ndarray:
    full = cv2.imread(path)
    lap, dark = tile_values(cv2.imread(str(cache_dir(768) / f"{image_id}.jpg"), cv2.IMREAD_GRAYSCALE))
    inner = lap[1:-1, 1:-1]
    pick = tuple(np.argwhere(inner == inner.min())[0] + 1)
    h, w = full.shape[:2]
    g = handcrafted.GRID
    for i in range(g):
        for j in range(g):
            y0, y1, x0, x1 = i * h // g, (i + 1) * h // g, j * w // g, (j + 1) * w // g
            red = (i, j) == pick
            cv2.rectangle(full, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255) if red else (0, 200, 255), 6 if red else 2)
            cv2.putText(full, f"L {lap[i, j]:.0f}  D {dark[i, j]:.2f}", (x0 + 12, y0 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(full, title, (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 255), 4, cv2.LINE_AA)
    return full


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw the baseline's worst validation photographs")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("-k", type=int, default=12)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))

    df = load_table()
    train_idx, val_idx = build_splits(df, cfg["split"]["group_column"], cfg["split"]["val_fraction"],
                                      cfg["split"]["seed"])
    F = handcrafted.feature_table(df)
    y = df[config.TARGET].to_numpy()
    model = baseline.fit_baseline(F.iloc[train_idx], y[train_idx], seed=cfg["seed"])

    val = df.iloc[val_idx]
    ok = (val["eval_ok"] == 1).to_numpy()
    pred = baseline.predict_baseline(model, F.iloc[val_idx])
    print(f"baseline val RMSE {evaluate.rmse(y[val_idx][ok], pred[ok]):.4f}")

    worst = evaluate.worst(val[ok], pred[ok], k=a.k)
    worst = worst.join(F, how="left")
    out = WORK / "baseline_worst"
    out.mkdir(parents=True, exist_ok=True)
    worst.to_csv(out / "worst.csv", index=False)
    print(worst.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    paths = df.set_index("image_id")["path"]
    for rank, r in enumerate(worst.itertuples(), 1):
        title = f"#{rank} {r.image_id}  usability {getattr(r, config.TARGET):.3f}  pred {r.pred:.3f}"
        cv2.imwrite(str(out / f"{rank:02d}_{r.image_id}.jpg"), draw(r.image_id, paths[r.image_id], title),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\n{len(worst)} annotated originals in {out}")


if __name__ == "__main__":
    main()
