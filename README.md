# Movie-Review-Classification-with-Transformers
Movie Review Sentiment Classification with RNN and Custom Transformers

This project performs binary sentiment classification (positive/negative) on the **IMDB** movie review dataset.  
We train and compare multiple architectures under the same preprocessing pipeline (word-level tokenization, fixed length 256).

Models compared:
- **RNN Baseline** (LSTM/GRU)
- **Vanilla Transformer** (Post-Norm, scratch multi-head self-attention)
- **Custom/Tweaked Transformer**
  - **Pre-Norm** (LayerNorm before sublayers)
  - **Pre-Norm + Warmup (+ Cosine decay)**

---

## Environment

- Python 3.10+ recommended
- macOS (MPS) or CPU

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data Preparation
Build the word-level vocabulary and create a train/val split:
```bash
python scripts/build_vocab.py
```

---

## Training
All runs save artifacts to: `outputs/runs/<run_name>_<timestamp>/`
#### 1. RNN (LSTM baseline)
```bash
python -m src.train --model rnn --run_name rnn_lstm
```

#### 2. Vanilla Transformer (Post-Norm)
```bash
python -m src.train --model vanilla --run_name vanilla_tf --lr 3e-4 --dropout 0.1
```

#### 3. Tweaked Transformer (Pre-Norm)
```bash
python -m src.train --model tweaked_prenorm --run_name tweaked_prenorm --lr 3e-4 --dropout 0.1
```

#### 4. Tweaked Transformer (Pre-Norm + Warmup + Cosine)
```bash
python -m src.train --model tweaked_warmup --run_name tweaked_warmup --lr 3e-4 --warmup_steps 200 --dropout 0.1
```
also can specify warmup as a ratio :
```bash
python -m src.train --model tweaked_warmup --run_name tweaked_warmup --lr 3e-4 --warmup_ratio 0.1 --dropout 0.1
```

---

## Outputs / Logging
Each run directory contains:
- `best_model.pt` — checkpoint selected by best validation metric
- `best_metrics.json` — summary (e.g., `best_val_acc`, `test_acc`, `test_loss`)
- `train_log.csv` — epoch-wise train/val metrics
- `curves_loss.png`, `curves_acc.png` — training curves
- `curves_lr.png` — learning rate curve (useful for warmup runs)

---

## Notes
- For fair comparison, all models share the same preprocessing: word-level tokenizer, top-N vocab, and fixed sequence length = 256.
Warmup behavior can be sensitive to `warmup_steps` / `warmup_ratio`, total steps, and learning rate—tune these if needed.
