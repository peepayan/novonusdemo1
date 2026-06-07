"""Train the EMG-conditioned diffusion policy on the cached features.

Standard diffusion training objective: add noise to (normalized) action
sequences, predict and remove that noise conditioned on the
multimodal observation. Adam, lr 1e-4, cosine annealing, early stopping
when val loss plateaus.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..stage3_sim import physics_constants as pc
from .dataset import (
    ActionNormalizer, DemoActionDataset, load_cached_samples,
    split_train_val,
)
from .policy import DiffusionPolicy, PolicyConfig


@dataclass
class TrainCfg:
    lr: float = 1e-4
    weight_decay: float = 1e-6
    batch_size: int = 32
    max_epochs: int = 200
    early_stop_patience: int = 20
    val_size: int = 20
    seed: int = 0
    device: str = "cuda:0"
    num_workers: int = 0     # Windows
    log_every: int = 1


def train(cfg: TrainCfg, *, include_emg: bool, out_dir: Path,
          tag: str) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"=== Stage 7 — loading cached samples ===", flush=True)
    samples = load_cached_samples(
        pc.PROJECT_ROOT / "outputs" / "stage5" / "dataset_index.json",
        pc.PROJECT_ROOT / "outputs" / "stage7" / "cached_features",
    )
    if not samples:
        print("[error] no cached samples found", file=sys.stderr)
        return {}
    print(f"  loaded {len(samples)} samples (stage4="
          f"{sum(1 for s in samples if s.stage=='stage4')}, "
          f"stage5={sum(1 for s in samples if s.stage=='stage5')})",
          flush=True)

    train_samples, val_samples = split_train_val(
        samples, seed=cfg.seed, val_size=cfg.val_size)
    print(f"  train={len(train_samples)}  val={len(val_samples)}",
          flush=True)

    normalizer = ActionNormalizer.fit(train_samples)
    print(f"  action mean={normalizer.mean}  std={normalizer.std}",
          flush=True)

    train_ds = DemoActionDataset(train_samples, include_emg=include_emg)
    val_ds = DemoActionDataset(val_samples, include_emg=include_emg)
    print(f"  train windows={len(train_ds)}  val windows={len(val_ds)}",
          flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)

    print(f"\n=== Stage 7 — building policy (include_emg={include_emg}) ===",
          flush=True)
    pcfg = PolicyConfig(include_emg=include_emg)
    policy = DiffusionPolicy(pcfg).to(device)
    n_params = sum(p.numel() for p in policy.parameters()
                   if p.requires_grad)
    print(f"  trainable params: {n_params/1e6:.2f} M", flush=True)

    optimizer = torch.optim.Adam(
        policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs)

    norm_t = normalizer.to(device)

    def _to_device(batch: dict) -> dict:
        out = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        # normalize action target
        out["action_norm"] = (
            (out["action"] - norm_t["mean"]) / norm_t["std"])
        return out

    def _epoch(loader: DataLoader, training: bool) -> float:
        if training:
            policy.train()
        else:
            policy.eval()
        ctx = torch.enable_grad() if training else torch.no_grad()
        total = 0.0
        n = 0
        with ctx:
            for batch in loader:
                b = _to_device(batch)
                loss = policy.compute_loss(
                    dino_feat=b["dino"], state=b["state"],
                    action_norm=b["action_norm"],
                    lstm_feat=b.get("lstm"),
                )
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), 1.0)
                    optimizer.step()
                total += float(loss.item()) * b["dino"].shape[0]
                n += b["dino"].shape[0]
        return total / max(n, 1)

    history = {"epoch": [], "train": [], "val": []}
    best_val = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    ckpt_path = out_dir / f"policy_{tag}_best.pt"
    t0 = time.time()
    print(f"\n=== Stage 7 — training ({tag}) ===", flush=True)
    for ep in range(1, cfg.max_epochs + 1):
        tl = _epoch(train_loader, training=True)
        vl = _epoch(val_loader, training=False)
        scheduler.step()
        history["epoch"].append(ep)
        history["train"].append(tl)
        history["val"].append(vl)
        if vl < best_val - 1e-6:
            best_val = vl
            best_epoch = ep
            epochs_since_improve = 0
            torch.save({
                "model_state": policy.state_dict(),
                "policy_cfg": asdict(pcfg),
                "train_cfg": asdict(cfg),
                "normalizer": {"mean": normalizer.mean.tolist(),
                               "std": normalizer.std.tolist()},
                "best_val": float(best_val),
                "best_epoch": int(best_epoch),
            }, ckpt_path)
        else:
            epochs_since_improve += 1
        if ep % cfg.log_every == 0:
            print(f"  epoch {ep:3d}  train={tl:.4f}  val={vl:.4f}  "
                  f"best={best_val:.4f}@{best_epoch}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if epochs_since_improve >= cfg.early_stop_patience:
            print(f"[stop] no improvement for "
                  f"{cfg.early_stop_patience} epochs", flush=True)
            break

    # loss curve
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0d111a")
    ax.plot(history["epoch"], history["train"], label="train",
            color="#60a5fa", lw=1.8)
    ax.plot(history["epoch"], history["val"], label="val",
            color="#f97316", lw=1.8)
    ax.axvline(best_epoch, color="#34d399", ls="--", alpha=0.6,
               label=f"best @ epoch {best_epoch}")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE noise loss")
    ax.set_title(f"Stage 7 diffusion policy ({tag})  "
                 f"best val = {best_val:.4f}", color="#cccccc")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_facecolor("#0d111a"); ax.tick_params(colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_color("#444")
    curve_path = out_dir / f"loss_curve_{tag}.png"
    fig.savefig(curve_path, dpi=130, facecolor="#0d111a")
    plt.close(fig)

    # save history JSON for downstream charts
    (out_dir / f"loss_history_{tag}.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")

    print(f"\n[stage7] train done   best val={best_val:.4f}@{best_epoch}  "
          f"wall={time.time()-t0:.0f}s  -> {ckpt_path}", flush=True)
    return {
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "wall_s": float(time.time() - t0),
        "ckpt": str(ckpt_path),
        "curve_png": str(curve_path),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-emg", action="store_true",
                    help="train the vision-only baseline instead")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--out-dir", default=str(
        pc.PROJECT_ROOT / "outputs" / "stage7"))
    args = ap.parse_args(argv)

    cfg = TrainCfg(
        lr=args.lr, batch_size=args.batch_size,
        max_epochs=args.epochs, early_stop_patience=args.patience,
    )
    tag = "baseline" if args.no_emg else "emg"
    res = train(cfg, include_emg=(not args.no_emg),
                out_dir=Path(args.out_dir), tag=tag)
    return 0 if res else 1


if __name__ == "__main__":
    sys.exit(main())
