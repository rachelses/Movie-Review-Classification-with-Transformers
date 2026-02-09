# src/train.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt

from src.data.tokenizer import WordTokenizer
from src.data.imdb_dataset import IMDBDataset
from src.data.collate import IMDBCollator, CollateConfig

from src.models.rnn_classifier import RNNClassifier
from src.models.transformer_classifier import VanillaTransformerClassifier
from src.models.transformer_tweaked import TweakedTransformerClassifierPreNorm


# -------------------------
# utils
# -------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_csv(log_rows: list[dict], path: Path) -> None:
    import csv

    if not log_rows:
        return
    fieldnames = list(log_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(log_rows)


def plot_curves(log_rows: list[dict], out_dir: Path) -> None:
    epochs = [r["epoch"] for r in log_rows]

    # Loss curve
    plt.figure()
    plt.plot(epochs, [r["train_loss"] for r in log_rows], label="train")
    plt.plot(epochs, [r["val_loss"] for r in log_rows], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "curves_loss.png")
    plt.close()

    # Acc curve
    plt.figure()
    plt.plot(epochs, [r["train_acc"] for r in log_rows], label="train")
    plt.plot(epochs, [r["val_acc"] for r in log_rows], label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "curves_acc.png")
    plt.close()

    # LR curve (optional, if present)
    if "lr" in log_rows[0]:
        plt.figure()
        plt.plot(epochs, [r["lr"] for r in log_rows], label="lr")
        plt.xlabel("epoch")
        plt.ylabel("learning rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "curves_lr.png")
        plt.close()


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
):
    """
    Warmup + Cosine decay scheduler (LambdaLR)
    - step 0..warmup_steps: lr increases linearly from 0 -> base_lr
    - after warmup: cosine decay from base_lr -> 0
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")
    warmup_steps = max(0, int(warmup_steps))
    warmup_steps = min(warmup_steps, total_steps)

    def lr_lambda(step: int):
        # step is 0-based
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        # cosine decay
        if total_steps == warmup_steps:
            return 1.0  # edge: no decay part
        progress = float(step - warmup_steps + 1) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# -------------------------
# evaluation
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device, is_rnn: bool) -> dict:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if is_rnn:
            lengths = batch["lengths"].to(device)
            logits = model(input_ids, lengths)
        else:
            pad_mask = batch["src_key_padding_mask"].to(device)
            logits = model(input_ids, pad_mask)

        loss = loss_fn(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).to(labels.dtype)
        correct = (preds == labels).sum().item()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += correct
        total_count += bs

    return {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
    }


# -------------------------
# main
# -------------------------
def main():
    p = argparse.ArgumentParser()

    # data paths
    p.add_argument("--processed_dir", type=str, default="data/processed")

    # model selection
    p.add_argument(
        "--model",
        type=str,
        default="rnn",
        choices=["rnn", "vanilla", "tweaked_prenorm", "tweaked_warmup"],
    )
    p.add_argument("--run_name", type=str, default="run")

    # training
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)

    # rnn hparams
    p.add_argument("--rnn_type", type=str, default="lstm", choices=["lstm", "gru"])
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--bidirectional", action="store_true", default=True)
    p.add_argument("--no_bidirectional", action="store_false", dest="bidirectional")

    # transformer hparams (vanilla / tweaked 공통)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    # warmup (tweaked_warmup에서 사용)
    p.add_argument("--warmup_ratio", type=float, default=0.1)   # 10% of total steps
    p.add_argument("--warmup_steps", type=int, default=-1)      # if >=0, overrides warmup_ratio

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_root", type=str, default="outputs/runs")

    args = p.parse_args()
    set_seed(args.seed)

    device = get_device()
    print("device:", device)

    processed_dir = Path(args.processed_dir)
    vocab_path = processed_dir / "vocab.json"
    splits_path = processed_dir / "splits.json"

    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing {vocab_path}. Run scripts/build_vocab.py first.")
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing {splits_path}. Run scripts/build_vocab.py first.")

    # load tokenizer + splits
    tok = WordTokenizer.load(vocab_path)
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    tr_idx = splits["train_indices"]
    va_idx = splits["val_indices"]

    # load imdb
    ds = load_dataset("imdb")
    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts = ds["test"]["text"]
    test_labels = ds["test"]["label"]

    tr_texts = [train_texts[i] for i in tr_idx]
    tr_labels = [train_labels[i] for i in tr_idx]
    va_texts = [train_texts[i] for i in va_idx]
    va_labels = [train_labels[i] for i in va_idx]

    # dataset / loader
    train_ds = IMDBDataset(tr_texts, tr_labels, tok)
    val_ds = IMDBDataset(va_texts, va_labels, tok)
    test_ds = IMDBDataset(test_texts, test_labels, tok)

    collate = IMDBCollator(CollateConfig(pad_idx=tok.cfg.pad_idx))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    vocab_size = len(tok.itos)

    # -------------------------
    # build model + optimizer + (optional) scheduler
    # -------------------------
    is_rnn = (args.model == "rnn")
    scheduler = None

    if args.model == "rnn":
        model = RNNClassifier(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            rnn_type=args.rnn_type,
            bidirectional=args.bidirectional,
            dropout=args.dropout,
            pad_idx=tok.cfg.pad_idx,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    elif args.model == "vanilla":
        model = VanillaTransformerClassifier(
            vocab_size=vocab_size,
            pad_idx=tok.cfg.pad_idx,
            max_len=tok.cfg.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
        ).to(device)

        wd = args.weight_decay if args.weight_decay > 0 else 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd)

    elif args.model == "tweaked_prenorm":
        model = TweakedTransformerClassifierPreNorm(
            vocab_size=vocab_size,
            pad_idx=tok.cfg.pad_idx,
            max_len=tok.cfg.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
        ).to(device)

        wd = args.weight_decay if args.weight_decay > 0 else 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd)

    else:  # tweaked_warmup  (Pre-Norm + Warmup+Cosine)
        model = TweakedTransformerClassifierPreNorm(
            vocab_size=vocab_size,
            pad_idx=tok.cfg.pad_idx,
            max_len=tok.cfg.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
        ).to(device)

        wd = args.weight_decay if args.weight_decay > 0 else 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd)

        total_steps = args.epochs * len(train_loader)
        if args.warmup_steps >= 0:
            warmup_steps = args.warmup_steps
        else:
            warmup_steps = int(args.warmup_ratio * total_steps)

        scheduler = make_warmup_cosine_scheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

    loss_fn = nn.BCEWithLogitsLoss()

    # -------------------------
    # outputs
    # -------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / f"{args.run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "tokenizer_config.json").write_text(
        json.dumps(asdict(tok.cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # -------------------------
    # train
    # -------------------------
    best_val_acc = -1.0
    best_state_path = run_dir / "best_model.pt"
    log_rows: list[dict] = []

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            # forward 분기 (핵심)
            if is_rnn:
                lengths = batch["lengths"].to(device)
                logits = model(input_ids, lengths)
            else:
                pad_mask = batch["src_key_padding_mask"].to(device)
                logits = model(input_ids, pad_mask)

            loss = loss_fn(logits, labels)
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(labels.dtype)
            correct = (preds == labels).sum().item()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_correct += correct
            total_count += bs

        train_loss = total_loss / max(total_count, 1)
        train_acc = total_correct / max(total_count, 1)

        val_metrics = evaluate(model, val_loader, device, is_rnn=is_rnn)
        val_loss, val_acc = val_metrics["loss"], val_metrics["acc"]

        # epoch 단위 lr 기록(대표값)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        }
        log_rows.append(row)

        print(
            f"[epoch {epoch:02d}] "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_state_path)

    # save logs + curves
    save_csv(log_rows, run_dir / "train_log.csv")
    plot_curves(log_rows, run_dir)

    # -------------------------
    # test (best checkpoint)
    # -------------------------
    model.load_state_dict(torch.load(best_state_path, map_location="cpu"))
    model.to(device)
    test_metrics = evaluate(model, test_loader, device, is_rnn=is_rnn)

    best_payload = {
        "best_val_acc": best_val_acc,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "best_state_path": str(best_state_path),
    }
    (run_dir / "best_metrics.json").write_text(
        json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== BEST ===")
    print(json.dumps(best_payload, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":
    main()
