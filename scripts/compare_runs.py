"""Compare runs on the same validation photographs, with a paired bootstrap over pages.

    python scripts/compare_runs.py data/work/dry_e8.csv data/work/dry_e12.csv data/work/dry_e16.csv

Each CSV is the output of `python -m src.predict ... --dry-run` for one
checkpoint: the validation photographs and their predictions. The first file
is the reference; every other one is compared with it.

Two runs that differ by a few thousandths may not differ at all. Photographs
of the same page are not independent, so the bootstrap resamples whole
pages, recomputes RMSE(run) - RMSE(reference) on each resample, and reports
the 95 % interval and how often the run is better. An interval that contains
0 is a tie.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data.dataset import load_table
from src.evaluate import rmse


def page_bootstrap(y: np.ndarray, pages: np.ndarray, pa: np.ndarray, pb: np.ndarray,
                   n: int = 2000, seed: int = 0) -> np.ndarray:
    """RMSE(b) - RMSE(a) on n resamples of whole pages."""
    ids = np.unique(pages)
    rows_of = {p: np.flatnonzero(pages == p) for p in ids}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for k in range(n):
        idx = np.concatenate([rows_of[p] for p in rng.choice(ids, len(ids))])
        diffs[k] = rmse(y[idx], pb[idx]) - rmse(y[idx], pa[idx])
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare dry-run predictions with a page bootstrap")
    ap.add_argument("csv", nargs="+", help="dry-run outputs; the first one is the reference")
    ap.add_argument("--n", type=int, default=2000, help="bootstrap resamples")
    a = ap.parse_args()
    if len(a.csv) < 2:
        raise SystemExit("give at least two dry-run files")

    df = load_table().set_index("image_id")
    runs = {Path(p).stem: pd.read_csv(p).set_index("image_id")["usability"] for p in a.csv}
    ids = next(iter(runs.values())).index
    for name, r in runs.items():
        if set(r.index) != set(ids):
            raise SystemExit(f"{name} has other photographs than {Path(a.csv[0]).stem}: not the same validation set")

    rows = df.loc[ids]
    ok = (rows["eval_ok"] == 1).to_numpy()
    y = rows[config.TARGET].to_numpy()[ok]
    pages = rows["page_id"].to_numpy()[ok]
    preds = {name: r.loc[ids].to_numpy()[ok] for name, r in runs.items()}
    print(f"{ok.sum()} scoreable validation photographs on {len(np.unique(pages))} pages\n")

    ref = next(iter(preds))
    print("| run | val RMSE | vs " + ref + " | 95 % interval (page bootstrap) | better in |")
    print("|---|---|---|---|---|")
    print(f"| {ref} | {rmse(y, preds[ref]):.4f} | reference | | |")
    for name, p in list(preds.items())[1:]:
        diffs = page_bootstrap(y, pages, preds[ref], p, a.n)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        d = rmse(y, p) - rmse(y, preds[ref])
        verdict = "tie" if lo <= 0 <= hi else ("better" if hi < 0 else "worse")
        print(f"| {name} | {rmse(y, p):.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] {verdict} | "
              f"{np.mean(diffs < 0):.0%} of resamples |")


if __name__ == "__main__":
    main()
