"""E3 force regressor training loop."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset_e3 import E3ForceWindows, E3SplitMeta, make_loaders, N_LSTM_FEATURES
from .model_e3 import E3ForceLSTM


@dataclass
class E3TrainConfig:
    max_epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    min_batch_size: int = 32
    device: str = "cuda:0"
    seed: int = 0


@dataclass
class E3EpochStats:
    epoch: int
    train_loss: float
    test_loss: float
    test_r2: float
    test_pearson: float
    lr: float
    seconds: float


@dataclass
class E3History:
    epochs: list[E3EpochStats] = field(default_factory=list)


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = y_true.astype(np.float64) - y_true.mean()
    b = y_pred.astype(np.float64) - y_pred.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + 1e-12
    return float((a * b).sum() / denom)


def _run_epoch(model: E3ForceLSTM, loader: DataLoader, loss_fn: nn.Module,
               opt: torch.optim.Optimizer | None, device: torch.device,
               use_features: bool
               ) -> tuple[float, np.ndarray, np.ndarray]:
    train = opt is not None
    model.train(train)
    total = 0.0
    n = 0
    ys, ps = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, feats, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            f = feats.to(device, non_blocking=True) if use_features else None
            out = model(x, f)
            loss = loss_fn(out.force, y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            bs = x.shape[0]
            n += bs
            total += float(loss.detach()) * bs
            if not train:
                ys.append(y.detach().cpu().numpy())
                ps.append(out.force.detach().cpu().numpy())
    avg = total / max(n, 1)
    return avg, (np.concatenate(ys) if ys else np.empty(0)), (np.concatenate(ps) if ps else np.empty(0))


def train(train_ds: E3ForceWindows, test_ds: E3ForceWindows, meta: E3SplitMeta,
          out_dir: str | Path, cfg: E3TrainConfig = E3TrainConfig(),
          variant_name: str = "rich"
          ) -> tuple[E3ForceLSTM, E3History, dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[e3-train:{variant_name}] device = {device}, "
          f"features_dim={meta.feature_dim}")

    bs = cfg.batch_size
    while True:
        try:
            model = E3ForceLSTM(
                n_lstm_features=N_LSTM_FEATURES,
                n_rich_features=meta.feature_dim,
            ).to(device)
            train_loader, test_loader = make_loaders(train_ds, test_ds, batch_size=bs)
            xb, fb, yb = next(iter(train_loader))
            xb, yb = xb.to(device), yb.to(device)
            fb_dev = fb.to(device) if meta.feature_dim > 0 else None
            out = model(xb, fb_dev)
            out.force.sum().backward()
            del out, xb, yb, fb
            torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= cfg.min_batch_size:
                raise
            new_bs = max(cfg.min_batch_size, bs // 2)
            print(f"[e3-train] OOM at bs={bs}, retry {new_bs}")
            bs = new_bs

    cfg.batch_size = bs
    # fresh model
    model = E3ForceLSTM(
        n_lstm_features=N_LSTM_FEATURES,
        n_rich_features=meta.feature_dim,
    ).to(device)
    train_loader, test_loader = make_loaders(train_ds, test_ds, batch_size=bs)

    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)

    hist = E3History()
    best_loss = float("inf")
    best_r2 = -float("inf")
    best_state = None
    best_ys = best_ps = None
    no_improve = 0

    print(f"[e3-train:{variant_name}] starting; max_epochs={cfg.max_epochs} "
          f"patience={cfg.patience} bs={bs}")
    for ep in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        try:
            tr_loss, _, _ = _run_epoch(model, train_loader, loss_fn, opt,
                                       device, meta.feature_dim > 0)
            te_loss, ys, ps = _run_epoch(model, test_loader, loss_fn, None,
                                         device, meta.feature_dim > 0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= cfg.min_batch_size:
                raise
            new_bs = max(cfg.min_batch_size, bs // 2)
            print(f"[e3-train] mid-OOM, reducing bs {bs} -> {new_bs}")
            bs = new_bs; cfg.batch_size = bs
            train_loader, test_loader = make_loaders(train_ds, test_ds, batch_size=bs)
            continue
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        r2 = _r2(ys, ps); rho = _pearson(ys, ps)
        es = E3EpochStats(epoch=ep, train_loss=tr_loss, test_loss=te_loss,
                          test_r2=r2, test_pearson=rho,
                          lr=cur_lr, seconds=time.time() - t0)
        hist.epochs.append(es)
        print(f"  ep {ep:>2d}  lr={cur_lr:.2e}  "
              f"train_mse={tr_loss:.5f}  test_mse={te_loss:.5f}  "
              f"R^2={r2:+.4f}  r={rho:+.4f}  ({es.seconds:.1f}s)")

        if te_loss < best_loss:
            best_loss = te_loss
            best_r2 = r2
            no_improve = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_ys, best_ps = ys, ps
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"[e3-train] early stop after {cfg.patience} no-improve epochs")
                break

    if best_state is None:
        raise RuntimeError("no epoch completed")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    ck = out_dir / f"force_model_best_{variant_name}.pt"
    torch.save({
        "model_state": best_state,
        "best_test_mse": float(best_loss),
        "best_test_r2": float(best_r2),
        "feature_dim": int(meta.feature_dim),
        "feat_mean": meta.feat_mean,
        "feat_std": meta.feat_std,
        "force_lo": float(meta.force_lo),
        "force_hi": float(meta.force_hi),
        "n_lstm_features": int(N_LSTM_FEATURES),
        "hparams": asdict(cfg),
        "variant": variant_name,
    }, ck)
    print(f"[e3-train:{variant_name}] saved checkpoint -> {ck}")

    info = {
        "best_test_mse": float(best_loss),
        "best_test_r2": float(best_r2),
        "best_test_pearson": float(_pearson(best_ys, best_ps)),
        "best_ys": best_ys,
        "best_ps": best_ps,
        "epochs_run": len(hist.epochs),
        "history": [asdict(e) for e in hist.epochs],
    }
    return model, hist, info
