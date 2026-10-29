"""Train the MLP on raw pixels and log the run to Weights & Biases.

    python -m src.train --config configs/lab3_mlp.yaml

Given: pixel standardisation, the epoch loop with its logging, and the
checkpoint. Yours, each a few lines, each with its contract in the docstring:

    make_optimizer   TASK 3
    train_epoch      TASK 4   the minibatch loop
    predict          TASK 5   model outputs as numpy
    validate         TASK 6   RMSE on the scoreable validation rows

Read the given parts once. There is no framework underneath: this file is the
whole of supervised training.

Early stopping (homework 1). With `train.early_stopping` in the config, a
part of the *training* groups is set aside as a stopping set: the model never
fits it, the best epoch is chosen on it, and those weights are restored and
checkpointed. The validation rows play no part in the choice, so the
validation score of the restored model is an honest one. Picking the epoch by
the validation curve, as the lab's "best epoch" column did, is not.
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import yaml

from src import config, tracking
from src.data.dataset import load_table, pixel_table
from src.data.splits import build_splits
from src.evaluate import rmse  # noqa: F401  (TASK 6 needs it)
from src.models.mlp import MLP
from src.seed import set_seed


class Standardiser:
    """Zero mean and unit variance over all pixels, fitted on the training rows only.

    One mean and one scale for the whole image, not one per pixel: with a
    thousand training rows a per-pixel estimate is mostly noise. Fitting on
    all rows would leak the validation distribution into training; a small
    leak, but the habit matters. The two numbers go into the checkpoint so
    that prediction uses exactly the same transform.
    """

    def fit(self, X: np.ndarray) -> Standardiser:
        self.mean = float(X.mean())
        self.scale = float(X.std()) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


def to_tensor(a, device: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)


# --------------------------------------------------------------- your part

def make_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """TASK 3. The optimiser, using cfg['train']['lr'] and cfg['train']['weight_decay']."""
    tr = cfg["train"]
    return torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])


def train_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
                X: torch.Tensor, y: torch.Tensor, batch_size: int, generator: torch.Generator) -> float:
    """TASK 4. One pass over the training rows in shuffled minibatches.

    Draw the order with `torch.randperm(len(X), generator=generator)` so the
    run is reproducible, step the optimiser once per minibatch, and return the
    mean training loss over the epoch, weighted by minibatch size.
    """
    model.train()                                           # dropout bekapcsolva
    perm = torch.randperm(len(X), generator=generator)      # véletlen, de reprodukálható sorrend
    total = 0.0
    for start in range(0, len(X), batch_size):
        idx = perm[start:start + batch_size].to(X.device)
        xb, yb = X[idx], y[idx]
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(idx)                     # súlyozás a minibatch méretével
    return total / len(X)


def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    """TASK 5. The model's outputs for every row of X, as a numpy array of shape (len(X),).

    Evaluation mode, no gradients, and it must give the same answer for a row
    whether that row is predicted alone or inside a batch.
    """
    model.eval()                                            # dropout kikapcsolva
    if len(X) == 0:
        return np.empty(0, dtype=np.float32)
    outs = []
    with torch.no_grad():                                   # nincs gradiens-számolás
        for start in range(0, len(X), batch_size):
            outs.append(model(X[start:start + batch_size]).cpu())
    return torch.cat(outs).numpy()


def validate(model: torch.nn.Module, X_val: torch.Tensor, y_val: np.ndarray, ok: np.ndarray) -> float:
    """TASK 6. RMSE on the validation rows where `ok` is True (the scoreable ones)."""
    pred = predict(model, X_val)
    ok = np.asarray(ok, dtype=bool)
    return float(rmse(np.asarray(y_val)[ok], pred[ok]))

# --------------------------------------------------------------- early stopping

def early_stopping_split(df: pd.DataFrame, train_idx: np.ndarray, es: dict) -> tuple[np.ndarray, np.ndarray]:
    """Split the training rows into the rows the model fits and the rows that choose its epoch.

    Whole groups of `es['group_column']` go to the stopping side, drawn with the
    split procedure and seed, so that no page and no photographer sits on both
    sides. `group` rather than `group_strict` by default: training has only
    five strict units, and setting one aside would take a whole document family
    out of fitting. The price is that the other language version of a stopping
    layout may be in the fitting rows, which makes the stopping score a little
    optimistic; it only ranks epochs, so that matters far less than it would
    for a reported number.
    """
    sub = df.iloc[train_idx]
    fit_pos, stop_pos = build_splits(sub, es.get("group_column", "group"), es.get("fraction", 0.15),
                                     es.get("seed", config.SPLIT_SEED))
    return train_idx[fit_pos], train_idx[stop_pos]


# --------------------------------------------------------------- given

def train_mlp(cfg: dict, X: np.ndarray, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
              device: str | None = None, tags: list[str] | None = None, log=print) -> dict:
    """Train on the pixel rows train_idx, validate on val_idx after every epoch, log everything.

    `X` is the pixel table, one row per row of `df`. Returns the history, the
    validation predictions after the first and the last epoch, the trained
    model, the checkpoint path and the W&B run page. With early stopping
    configured, the model, the checkpoint, `val_pred` and `val_rmse` are those
    of the best epoch on the stopping set; otherwise of the last epoch.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    tr = cfg["train"]
    es = tr.get("early_stopping")

    fit_idx, stop_idx = early_stopping_split(df, train_idx, es) if es else (train_idx, train_idx[:0])
    y_all, ok_all = df[config.TARGET].to_numpy(), df["eval_ok"].to_numpy() == 1

    scaler = Standardiser().fit(X[fit_idx])
    X_tr = to_tensor(scaler.transform(X[fit_idx]), device)
    X_st = to_tensor(scaler.transform(X[stop_idx]), device)
    X_va = to_tensor(scaler.transform(X[val_idx]), device)
    y_tr = to_tensor(y_all[fit_idx], device)
    y_st, ok_st = y_all[stop_idx], ok_all[stop_idx]
    y_va, ok = y_all[val_idx], ok_all[val_idx]

    model = MLP(in_dim=X.shape[1], hidden=list(cfg["model"]["hidden"]), dropout=cfg["model"]["dropout"],
                prior=float(y_tr.mean())).to(device)
    optimizer = make_optimizer(model, cfg)
    loss_fn = torch.nn.MSELoss()           # the metric is RMSE; same minimiser
    generator = torch.Generator().manual_seed(cfg["seed"])

    tracking.init_run(cfg, tags=tags)
    history, snapshots = [], {}
    best_stop, best_epoch, best_state = float("inf"), None, None
    stop_note = f" ({len(fit_idx)} fit / {len(stop_idx)} stopping)" if es else ""
    log(f"{cfg['name']}: {len(train_idx)} train{stop_note} / {len(val_idx)} val rows, {X.shape[1]} inputs, "
        f"{sum(p.numel() for p in model.parameters()):,} parameters, {device}")
    for epoch in range(1, tr["epochs"] + 1):
        t0 = time.time()
        loss = train_epoch(model, optimizer, loss_fn, X_tr, y_tr, tr["batch_size"], generator)
        val = validate(model, X_va, y_va, ok)
        train_eval = rmse(y_tr.cpu().numpy(), predict(model, X_tr))   # dropout off, as in validation
        row = {"epoch": epoch, "train_loss": loss, "train_rmse": train_eval, "val_rmse": val,
               "seconds": time.time() - t0}
        if es:
            # The epoch is chosen here, on training groups the model does not fit. Never on `val`.
            row["stop_rmse"] = validate(model, X_st, y_st, ok_st)
            if row["stop_rmse"] < best_stop:
                best_stop, best_epoch = row["stop_rmse"], epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(row)
        tracking.log_metrics({k: v for k, v in row.items() if k not in ("epoch", "seconds")}, step=epoch)
        if epoch in (1, tr["epochs"]) or epoch % tr.get("log_every", 10) == 0:
            stop_txt = f"  stop RMSE {row['stop_rmse']:.4f}" if es else ""
            log(f"epoch {epoch:3d}  train loss {loss:.4f}  train RMSE {train_eval:.4f}{stop_txt}  val RMSE {val:.4f}")
        if epoch == 1:
            snapshots[epoch] = predict(model, X_va)
        if es and es.get("patience") and epoch - best_epoch >= es["patience"]:
            log(f"no improvement on the stopping set for {es['patience']} epochs, stopping at epoch {epoch}")
            break
    snapshots[epoch] = predict(model, X_va)

    if es:
        model.load_state_dict(best_state)
        log(f"restored epoch {best_epoch} (stopping RMSE {best_stop:.4f})")
    val_pred = predict(model, X_va)
    final_val = float(rmse(y_va[ok], val_pred[ok]))
    summary = {"final_val_rmse": final_val, "last_epoch_val_rmse": history[-1]["val_rmse"]}
    if es:
        summary |= {"best_epoch": best_epoch, "best_stop_rmse": best_stop}

    ckpt = config.ROOT / "models" / f"{cfg['name']}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "in_dim": X.shape[1],
                "scaler_mean": scaler.mean, "scaler_scale": scaler.scale, "epoch": best_epoch or epoch}, ckpt)
    tracking.log_checkpoint(ckpt, cfg["name"], summary)
    tracking.log_metrics(summary)
    url = tracking.run_url()
    tracking.finish()
    return {"history": pd.DataFrame(history), "val_pred": val_pred, "val_rmse": final_val,
            "best_epoch": best_epoch, "snapshots": snapshots, "model": model, "checkpoint": ckpt,
            "url": url, "val_index": val_idx, "ok": ok}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the MLP from a config")
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    df = load_table()
    X = pixel_table(df, cfg["data"]["pixel_size"])
    train_idx, val_idx = build_splits(df, cfg["split"]["group_column"], cfg["split"]["val_fraction"], cfg["split"]["seed"])
    result = train_mlp(cfg, X, df, train_idx, val_idx)
    chosen = f"best epoch {result['best_epoch']} on the stopping set" if result["best_epoch"] else "last epoch"
    print(f"val RMSE {result['val_rmse']:.4f} ({chosen})  checkpoint {result['checkpoint']}  "
          f"run {result['url']}")


if __name__ == "__main__":
    main()
