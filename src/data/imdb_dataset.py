from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Optional

from torch.utils.data import Dataset

from src.data.tokenizer import WordTokenizer


@dataclass
class IMDBDatasetConfig:
    include_length: bool = False  # if True, __getitem__ returns "length" too


class IMDBDataset(Dataset):
    """
    Minimal dataset for IMDB sentiment classification.
    Expects:
      - texts: list[str]
      - labels: list[int] (0/1)
      - tokenizer: WordTokenizer (vocab already built or loaded)

    Returns each item as:
      {"input_ids": List[int], "label": int}  (+ optional "length")
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: WordTokenizer,
        cfg: Optional[IMDBDatasetConfig] = None,
    ):
        if len(texts) != len(labels):
            raise ValueError(f"texts/labels length mismatch: {len(texts)} vs {len(labels)}")

        self.texts = list(texts)
        self.labels = list(labels)
        self.tok = tokenizer
        self.cfg = cfg or IMDBDatasetConfig()

        if not self.tok.is_built():
            raise RuntimeError("Tokenizer vocab not built/loaded. Build vocab or load vocab.json first.")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        label = int(self.labels[idx])

        input_ids, length = self.tok.encode(text)

        item: Dict[str, Any] = {
            "input_ids": input_ids,
            "label": label,
        }
        if self.cfg.include_length:
            item["length"] = length

        return item
