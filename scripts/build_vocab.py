from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.data.tokenizer import WordTokenizer, VocabConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top_n", type=int, default=20000)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="data/processed")
    # keep tokenizer settings consistent with your tokenizer.py defaults
    p.add_argument("--lower", action="store_true", default=True)
    p.add_argument("--no_lower", action="store_false", dest="lower")
    p.add_argument("--keep_punct", action="store_true", default=True)
    p.add_argument("--no_keep_punct", action="store_false", dest="keep_punct")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load IMDB (cached by HF)
    ds = load_dataset("imdb")
    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]

    # 2) Make deterministic split indices (recommended to save)
    all_idx = list(range(len(train_texts)))
    tr_idx, va_idx = train_test_split(
        all_idx,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=train_labels,
    )

    # 3) Build vocab from TRAIN SPLIT ONLY
    cfg = VocabConfig(
        top_n=args.top_n,
        seq_len=args.seq_len,
        lower=args.lower,
        keep_punct=args.keep_punct,
    )
    tok = WordTokenizer(cfg)

    tr_texts = [train_texts[i] for i in tr_idx]
    tok.build_vocab(tr_texts)

    vocab_path = out_dir / "vocab.json"
    tok.save(vocab_path)

    # 4) Save split metadata (so every run uses the same train/val)
    splits_path = out_dir / "splits.json"
    payload = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train_size": len(tr_idx),
        "val_size": len(va_idx),
        "train_indices": tr_idx,
        "val_indices": va_idx,
        "source": "imdb_hf_datasets",
    }
    splits_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] saved vocab:  {vocab_path}")
    print(f"[OK] saved splits: {splits_path}")
    print(f"     train/val sizes: {len(tr_idx)}/{len(va_idx)}")


if __name__ == "__main__":
    main()
