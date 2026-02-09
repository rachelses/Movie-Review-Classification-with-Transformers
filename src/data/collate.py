# src/data/collate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch


BatchItem = Union[
    Tuple[List[int], int],              # (input_ids, label)
    Dict[str, Any],                     # {"input_ids": [...], "label": ...}
]


@dataclass
class CollateConfig:
    pad_idx: int = 0
    # labels for BCEWithLogitsLoss are typically float (0.0/1.0)
    label_dtype: torch.dtype = torch.float32


class IMDBCollator:
    """
    Collator that:
    - stacks fixed-length input_ids (B, L)
    - builds src_key_padding_mask: True where PAD (B, L)
    - computes lengths (#non-pad tokens) (B,)
    """

    def __init__(self, cfg: CollateConfig | None = None):
        self.cfg = cfg or CollateConfig()

    def __call__(self, batch: Sequence[BatchItem]) -> Dict[str, torch.Tensor]:
        input_ids_list: List[List[int]] = []
        labels_list: List[float] = []

        for item in batch:
            if isinstance(item, dict):
                ids = item["input_ids"]
                y = item["label"]
            else:
                ids, y = item

            input_ids_list.append(ids)
            labels_list.append(float(y))

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)  # (B, L)
        labels = torch.tensor(labels_list, dtype=self.cfg.label_dtype)  # (B,)

        # True where PAD (so attention should ignore those positions)
        src_key_padding_mask = (input_ids == self.cfg.pad_idx)  # (B, L) bool

        # lengths = number of non-pad tokens
        lengths = (~src_key_padding_mask).sum(dim=1).to(torch.long)  # (B,)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "src_key_padding_mask": src_key_padding_mask,
            "lengths": lengths,
        }
