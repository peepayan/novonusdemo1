"""Joint training loop for the intent + force LSTM.

Loss:    L = CE(class) + lambda_force * MSE(force)
         CE is class-weighted by inverse train frequency.
Opt:     Adam, lr 1e-3, cosine annealing across max_epochs.
Stop:    early stop after `patience` epochs without val-accuracy improvement.
OOM:     if a CUDA OOM is hit, halve batch size and retry (down to a floor).
Output:  loss_curve.png + best checkpoint with weights/class mapping/hparams.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .class_mapping import INTENT_NAMES, N_INTENT_CLASSES, mapping_json_dict
from .dataset import (
    SplitStats, WindowedNinaproDataset, make_loaders,
)
from .model import IntentForceLSTM


@dataclass
class TrainConfig:
    max_epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    lambda_force: float = 1.0
    patience: int = 10
    min_batch_size: int = 32
    device: str = "cuda:0"
    seed: int = 0


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_ce: float
    train_mse: float
    val_loss: float
    val_ce: float
    val_mse: float
    val_acc: float
    lr: float
    seconds: float


@dataclass
class TrainHistory:
    epochs: list[EpochStats] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.epochs)

    def best_acc(self) -> float:
        return max((e.val_acc for e in self.epochs), default=0.0)


# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(model: IntentForceLSTM,
               loader: DataLoader,
               ce_loss: nn.Module,
               mse_loss: nn.Module,
               optimizer: torch.optim.Optimizer | None,
               device: torch.device,
               lambda_force: float
               ) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    train = optimizer is not None
    model.train(train)

    total_loss = total_ce = total_mse = 0.0
    n = 0
    correct = 0
    K = N_INTENT_CLASSES
    confusion = np.zeros((K, K), dtype=np.int64)
    # for later force-MSE per intent diagnostic
    force_pred_all: list[np.ndarray] = []
    force_true_all: list[np.ndarray] = []

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            f = f.to(device, non_blocking=True)

            out = model(x)
            ce = ce_loss(out.logits, y)
            mse = mse_loss(out.force, f)
            loss = ce + lambda_force * mse

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = x.shape[0]
            n += bs
            total_loss += float(loss.detach()) * bs
            total_ce += float(ce.detach()) * bs
            total_mse += float(mse.detach()) * bs

            preds = out.logits.argmax(dim=-1)
            correct += int((preds == y).sum().item())
            if not train:
                y_np = y.detach().cpu().numpy()
                p_np = preds.detach().cpu().numpy()
                np.add.at(confusion, (y_np, p_np), 1)
                force_pred_all.append(out.force.detach().cpu().numpy())
                force_true_all.append(f.detach().cpu().numpy())

    if n == 0:
        raise RuntimeError("empty loader")
    avg_loss = total_loss / n
    avg_ce = total_ce / n
    avg_mse = total_mse / n
    acc = correct / n
    fp = np.concatenate(force_pred_all) if force_pred_all else np.empty(0)
    ft = np.concatenate(force_true_all) if force_true_all else np.empty(0)
    return avg_loss, avg_ce, avg_mse, acc, confusion, (fp, ft)


def _try_train_loaders(train_ds: WindowedNinaproDataset,
                       val_ds: WindowedNinaproDataset,
                       batch_size: int):
    return make_loaders(train_ds, val_ds, batch_size=batch_size)


def train(train_ds: WindowedNinaproDataset,
          val_ds: WindowedNinaproDataset,
          stats: SplitStats,
          out_dir: str | Path,
          cfg: TrainConfig = TrainConfig()
          ) -> tuple[IntentForceLSTM, TrainHistory, dict]:
    """Train the model. Returns (best model loaded, history, info dict)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    bs = cfg.batch_size
    while True:
        try:
            train_loader, val_loader = _try_train_loaders(train_ds, val_ds, bs)
            # try a single dummy forward+backward to actually catch OOM here
            model = IntentForceLSTM().to(device)
            xb, yb, fb = next(iter(train_loader))
            xb, yb, fb = xb.to(device), yb.to(device), fb.to(device)
            out = model(xb)
            (out.logits.sum() + out.force.sum()).backward()
            del out, xb, yb, fb
            torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= cfg.min_batch_size:
                raise
            new_bs = max(cfg.min_batch_size, bs // 2)
            print(f"[train] CUDA OOM at batch_size={bs}, retrying at {new_bs}")
            bs = new_bs

    cfg.batch_size = bs
    print(f"[train] using batch_size={bs}")

    # fresh model + optimizer (the trial run above was scratch)
    model = IntentForceLSTM().to(device)
    train_loader, val_loader = _try_train_loaders(train_ds, val_ds, bs)

    weights = torch.from_numpy(stats.train_class_weights).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=weights)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs,
    )

    history = TrainHistory()
    best_acc = -1.0
    best_state: dict | None = None
    epochs_no_improve = 0
    best_confusion: np.ndarray | None = None
    best_force = (np.empty(0), np.empty(0))

    print(f"[train] starting; max_epochs={cfg.max_epochs}, patience={cfg.patience}, "
          f"lambda_force={cfg.lambda_force}")
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        try:
            tr_loss, tr_ce, tr_mse, tr_acc, _, _ = _run_epoch(
                model, train_loader, ce_loss, mse_loss, optimizer,
                device, cfg.lambda_force,
            )
            val_loss, val_ce, val_mse, val_acc, val_cm, val_force = _run_epoch(
                model, val_loader, ce_loss, mse_loss, None,
                device, cfg.lambda_force,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= cfg.min_batch_size:
                raise
            new_bs = max(cfg.min_batch_size, bs // 2)
            print(f"[train] CUDA OOM mid-epoch; reducing batch_size {bs} -> {new_bs}")
            bs = new_bs
            cfg.batch_size = bs
            train_loader, val_loader = _try_train_loaders(train_ds, val_ds, bs)
            continue

        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        es = EpochStats(
            epoch=epoch,
            train_loss=tr_loss, train_ce=tr_ce, train_mse=tr_mse,
            val_loss=val_loss, val_ce=val_ce, val_mse=val_mse,
            val_acc=val_acc, lr=cur_lr, seconds=time.time() - t0,
        )
        history.epochs.append(es)
        print(
            f"  ep {epoch:>2d}  lr={cur_lr:.2e}  "
            f"train: loss={tr_loss:.4f} ce={tr_ce:.4f} mse={tr_mse:.4f}  "
            f"val: loss={val_loss:.4f} ce={val_ce:.4f} mse={val_mse:.4f} acc={val_acc*100:.2f}%  "
            f"({es.seconds:.1f}s)"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            best_state = {
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "val_acc": val_acc,
                "val_mse": val_mse,
            }
            best_confusion = val_cm
            best_force = val_force
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"[train] early stop: no val-acc improvement for "
                      f"{cfg.patience} epochs")
                break

        # Diagnostic checkpoint: at epoch ~20, warn if accuracy is still poor.
        if epoch == 20 and best_acc < 0.85:
            print(f"[train] WARNING: best val_acc only {best_acc*100:.2f}% after 20 epochs;"
                  " consider tuning lambda_force / class weights / model size.")

    # ---------- Save outputs ----------
    if best_state is None:
        raise RuntimeError("no successful epoch completed")

    # rebuild model from best state for return
    model.load_state_dict({k: v.to(device) for k, v in best_state["model_state"].items()})
    model.eval()

    ckpt_path = out_dir / "lstm_best.pt"
    torch.save({
        "model_state": best_state["model_state"],
        "best_epoch": best_state["epoch"],
        "best_val_acc": best_state["val_acc"],
        "best_val_mse": best_state["val_mse"],
        "intent_class_mapping": mapping_json_dict(),
        "hparams": asdict(cfg),
        "n_features": 70,
        "hidden_size": 128,
        "n_layers": 2,
    }, ckpt_path)
    print(f"[train] saved best checkpoint -> {ckpt_path}")

    # Loss/acc curves
    _save_loss_curve(history, out_dir / "loss_curve.png")
    print(f"[train] saved loss curve -> {out_dir / 'loss_curve.png'}")

    info = {
        "best_epoch": best_state["epoch"],
        "best_val_acc": float(best_state["val_acc"]),
        "best_val_mse": float(best_state["val_mse"]),
        "final_batch_size": cfg.batch_size,
        "confusion_matrix": best_confusion.tolist() if best_confusion is not None else None,
        "epochs_run": len(history.epochs),
        "history": [asdict(e) for e in history.epochs],
        "val_force_pred": best_force[0],
        "val_force_true": best_force[1],
    }
    return model, history, info


# ---------------------------------------------------------------------------

def _save_loss_curve(history: TrainHistory, path: Path) -> None:
    ep = [e.epoch for e in history.epochs]
    tr_loss = [e.train_loss for e in history.epochs]
    va_loss = [e.val_loss for e in history.epochs]
    va_acc = [e.val_acc * 100.0 for e in history.epochs]
    va_mse = [e.val_mse for e in history.epochs]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax_loss, ax_acc, ax_mse = axes
    ax_loss.plot(ep, tr_loss, color="#00e5ff", label="train")
    ax_loss.plot(ep, va_loss, color="#ff5b7c", label="val")
    ax_loss.set_title("Total loss (CE + lambda*MSE)")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss")
    ax_loss.grid(alpha=0.15); ax_loss.legend(frameon=False)

    ax_acc.plot(ep, va_acc, color="#a3e635", lw=2)
    ax_acc.set_title("Val intent-classification accuracy")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy (%)")
    ax_acc.set_ylim(0, 100); ax_acc.grid(alpha=0.15)

    ax_mse.plot(ep, va_mse, color="#facc15", lw=2)
    ax_mse.set_title("Val force-MSE")
    ax_mse.set_xlabel("epoch"); ax_mse.set_ylabel("MSE")
    ax_mse.grid(alpha=0.15)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, path: Path,
                          class_names: tuple[str, ...] = INTENT_NAMES) -> None:
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.where(row_sums > 0, cm / np.maximum(row_sums, 1.0), 0.0)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Validation confusion matrix (row-normalized)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{norm[i,j]*100:.0f}%\n{int(cm[i,j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if norm[i,j] < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_training_summary(path: Path, cfg: TrainConfig, stats: SplitStats,
                           info: dict) -> None:
    lines = []
    lines.append("Novonus Stage 2 — training summary")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"best epoch:           {info['best_epoch']}")
    lines.append(f"best val intent-acc:  {info['best_val_acc']*100:.2f}%")
    lines.append(f"best val force-MSE:   {info['best_val_mse']:.6f}")
    lines.append(f"epochs run:           {info['epochs_run']}")
    lines.append(f"final batch size:     {info['final_batch_size']}")
    lines.append("")
    lines.append("hyperparameters:")
    for k, v in asdict(cfg).items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(stats.report())
    lines.append("")
    lines.append("NOTE: the force-intensity target is an EMG-amplitude proxy,")
    lines.append("computed as the smoothed mean of the 12-channel MVC-normalized")
    lines.append("EMG envelope, NOT a calibrated/measured force. The trained")
    lines.append("force head is therefore a learned readout of muscle effort,")
    lines.append("not a measurement. See force_validation_summary.txt for")
    lines.append("independent E3 correlation analysis vs measured force.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
