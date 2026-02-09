# src/models/transformer_tweaked.py
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Scratch multi-head self-attention.
    - mask: src_key_padding_mask shape (B, L), True where PAD (ignore)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model({d_model}) must be divisible by n_heads({n_heads}).")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=True)
        self.wk = nn.Linear(d_model, d_model, bias=True)
        self.wv = nn.Linear(d_model, d_model, bias=True)
        self.wo = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, L, D)
        src_key_padding_mask: (B, L) bool, True where PAD
        return: (B, L, D)
        """
        B, L, D = x.shape

        q = self.wq(x)  # (B, L, D)
        k = self.wk(x)
        v = self.wv(x)

        # split heads: (B, heads, L, d_head)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # scores: (B, heads, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # apply padding mask on keys (last dimension)
        if src_key_padding_mask is not None:
            # mask: (B, 1, 1, L) broadcast to scores
            mask = src_key_padding_mask[:, None, None, :]  # True = PAD
            scores = scores.masked_fill(mask, -1e9)

        attn = F.softmax(scores, dim=-1)  # (B, heads, L, L)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, heads, L, d_head)
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)
        out = self.wo(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Vanilla FFN: Linear -> ReLU -> Linear"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderLayerPreNorm(nn.Module):
    """
    ✅ Pre-Norm Encoder Layer:
      x = x + Attn(LN(x))
      x = x + FFN(LN(x))
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        a = self.attn(self.ln1(x), src_key_padding_mask)
        x = x + a

        f = self.ffn(self.ln2(x))
        x = x + f

        return x


class TransformerEncoderPreNorm(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayerPreNorm(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        # Often used in Pre-Norm stacks for extra stability
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        x = self.final_ln(x)
        return x


class TweakedTransformerClassifierPreNorm(nn.Module):
    """
    Pre-Norm Transformer classifier.
    NOTE: Warmup is handled in training loop (optimizer/scheduler), not here.
    """
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int = 0,
        max_len: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder = TransformerEncoderPreNorm(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        input_ids: (B, L)
        src_key_padding_mask: (B, L) bool True=PAD
        return logits: (B,)
        """
        B, L = input_ids.shape
        if L != self.max_len:
            raise ValueError(f"Expected seq_len={self.max_len}, got L={L}")

        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        x = self.encoder(x, src_key_padding_mask)  # (B, L, D)

        # masked mean pooling (exclude PAD)
        if src_key_padding_mask is None:
            pooled = x.mean(dim=1)
        else:
            non_pad = (~src_key_padding_mask).unsqueeze(-1)  # (B, L, 1)
            x_masked = x * non_pad
            denom = non_pad.sum(dim=1).clamp(min=1)  # (B, 1)
            pooled = x_masked.sum(dim=1) / denom  # (B, D)

        logits = self.head(pooled).squeeze(1)
        return logits
