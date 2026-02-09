from datasets import load_dataset
from sklearn.model_selection import train_test_split

SEED = 42
VAL_RATIO = 0.1

ds = load_dataset("imdb")

train_texts = ds["train"]["text"]
train_labels = ds["train"]["label"]
test_texts  = ds["test"]["text"]
test_labels = ds["test"]["label"]

tr_texts, va_texts, tr_labels, va_labels = train_test_split(
    train_texts, train_labels,
    test_size=VAL_RATIO,
    random_state=SEED,
    stratify=train_labels,
)

print("train:", len(tr_texts), "val:", len(va_texts), "test:", len(test_texts))
print("label ratio (train/val/test):",
      sum(tr_labels)/len(tr_labels),
      sum(va_labels)/len(va_labels),
      sum(test_labels)/len(test_labels))
print("sample:", tr_labels[0], tr_texts[0][:200])
