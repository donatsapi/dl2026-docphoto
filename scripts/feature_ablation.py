"""Does my own feature help the baseline? Seven features against seven plus mine.

    python scripts/feature_ablation.py

Three numbers, because one split cannot resolve a difference under about 0.01:

1. the lab's split (strict unit 4 held out), the number comparable with the board;
2. grouped k-fold over the six strict units: every photograph predicted by the
   model that never saw its unit, RMSE over all scoreable rows;
3. a paired bootstrap over pages on those out-of-fold predictions: how often
   the sign of the difference survives resampling.

Prints a table ready to paste into the README, whichever way it comes out.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from src import config, evaluate
from src.data.dataset import WORK, load_table
from src.data.splits import build_splits
from src.features import handcrafted
from src.models import baseline


def cached_features(df: pd.DataFrame, name: str, features: dict) -> pd.DataFrame:
    store = WORK / f"features_{name}.csv"
    if store.exists():
        F = pd.read_csv(store, index_col=0)
        if list(F.columns) == list(features) and len(F) == len(df) and (F.index == df.index).all():
            return F
    F = handcrafted.feature_table(df, features=features)
    F.to_csv(store)
    return F


def out_of_fold(F: pd.DataFrame, df: pd.DataFrame, seed: int) -> np.ndarray:
    y = df[config.TARGET].to_numpy()
    units = df["group_strict"].to_numpy()
    pred = np.empty(len(df))
    for u in np.unique(units):
        test = units == u
        pred[test] = baseline.predict_baseline(baseline.fit_baseline(F[~test], y[~test], seed=seed), F[test])
    return pred


def page_bootstrap(df: pd.DataFrame, pa: np.ndarray, pb: np.ndarray, n: int = 2000, seed: int = 0) -> np.ndarray:
    """RMSE(b) - RMSE(a) on n resamples of whole pages."""
    y = df[config.TARGET].to_numpy()
    pages = df["page_id"].to_numpy()
    ids = np.unique(pages)
    rows_of = {p: np.flatnonzero(pages == p) for p in ids}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for k in range(n):
        idx = np.concatenate([rows_of[p] for p in rng.choice(ids, len(ids))])
        diffs[k] = evaluate.rmse(y[idx], pb[idx]) - evaluate.rmse(y[idx], pa[idx])
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser(description="Seven features against seven plus my own")
    ap.add_argument("--config", default="configs/baseline.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    seed = cfg["seed"]

    df = load_table()
    y = df[config.TARGET].to_numpy()
    ok = (df["eval_ok"] == 1).to_numpy()
    F7 = cached_features(df, "seven", handcrafted.FEATURES)
    Fo = cached_features(df, "own", handcrafted.OWN_FEATURES)
    F8 = pd.concat([F7, Fo], axis=1)
    own = list(handcrafted.OWN_FEATURES)
    print(f"own feature(s) {own}: {int(Fo.isna().sum().sum())} NaN of {len(Fo)} rows")
    print("Spearman correlation of own feature(s) with the target and with the seven:")
    print(F8.assign(usability=y)[ok].corr(method="spearman")[own].round(3).to_string())

    # 1. the lab's split
    train_idx, val_idx = build_splits(df, cfg["split"]["group_column"], cfg["split"]["val_fraction"],
                                      cfg["split"]["seed"])
    vok = ok[val_idx]
    split = {}
    for name, F in (("7 features", F7), ("7 + own", F8)):
        m = baseline.fit_baseline(F.iloc[train_idx], y[train_idx], seed=seed)
        p = baseline.predict_baseline(m, F.iloc[val_idx])
        split[name] = evaluate.rmse(y[val_idx][vok], p[vok])
        if name == "7 + own":
            imp = baseline.importance(m, F.iloc[val_idx][vok], y[val_idx][vok], seed=seed)

    # 2. grouped k-fold over the strict units
    oof7, oof8 = out_of_fold(F7, df, seed), out_of_fold(F8, df, seed)
    units = df["group_strict"].to_numpy()
    per_unit = pd.DataFrame([
        {"unit": u, "n": int((ok & (units == u)).sum()),
         "7 features": evaluate.rmse(y[ok & (units == u)], oof7[ok & (units == u)]),
         "7 + own": evaluate.rmse(y[ok & (units == u)], oof8[ok & (units == u)])}
        for u in np.unique(units)
    ]).set_index("unit")

    # 3. paired bootstrap over pages
    diffs = page_bootstrap(df[ok], oof7[ok], oof8[ok])
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    print("\nper held-out strict unit (out-of-fold RMSE):")
    print(per_unit.round(4).to_string())
    print("\npermutation importance on the lab's validation split, 7 + own:")
    print(imp.round(4).to_string())

    o7, o8 = evaluate.rmse(y[ok], oof7[ok]), evaluate.rmse(y[ok], oof8[ok])
    print("\n| baseline | val RMSE, strict unit 4 | grouped 6-fold RMSE |")
    print("|---|---|---|")
    print(f"| gradient boosting, 7 features | {split['7 features']:.4f} | {o7:.4f} |")
    print(f"| gradient boosting, 7 + `{'`, `'.join(own)}` | {split['7 + own']:.4f} | {o8:.4f} |")
    print(f"\npaired page bootstrap on the 6-fold predictions: RMSE(7+own) - RMSE(7) = {o8 - o7:+.4f}, "
          f"95 % interval [{lo:+.4f}, {hi:+.4f}], better in {np.mean(diffs < 0):.0%} of resamples")


if __name__ == "__main__":
    main()
