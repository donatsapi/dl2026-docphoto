"""Train and validation splits.

The same page appears in dozens of photographs, every layout exists in a
Hungarian and an English version typeset identically, and each photographer's
photos share a phone and a room. `train.csv` has two grouping columns:

* `group`: no page and no photographer appears in two groups;
* `group_strict`: coarser, and in addition both language versions of a
  layout are in the same unit. The competition's test set was cut along it.

Split by `group_strict`. Measured on this data with a ResNet-18 at 768 px,
RMSE by validation scheme against the held-out test:

    random 5-fold   0.113
    by group        0.136
    by group_strict 0.143
    held-out test   0.148

The random split is a fifth too good. `group` is still 8 % too good, because
the two language versions of a layout sit in two neighbouring groups and the
network recognises the layout. `group_strict` lands on the test.

TASK 1 is `build_splits`. The test in `tests/test_splits.py` is red until it
is written.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def build_splits(df: pd.DataFrame, group_column: str = "group_strict", val_fraction: float = 0.2,
                 seed: int = config.SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    units = np.sort(df[group_column].unique())

    np.random.default_rng(seed).shuffle(units)

    groups = df[group_column].to_numpy()
    target = val_fraction * len(df)
    val_units = []
    val_count = 0
    for unit in units:
        if val_count >= target:
            break
        val_units.append(unit)
        val_count += int((groups == unit).sum())

    is_val = np.isin(groups, val_units)
    val_idx = np.flatnonzero(is_val)
    train_idx = np.flatnonzero(~is_val)
    return train_idx, val_idx
