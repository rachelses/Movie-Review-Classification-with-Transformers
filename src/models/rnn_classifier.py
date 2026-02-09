from __future__ import annotations

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    """
    Baseline sentiment classifier using (Bi)LSTM or (Bi)GRU.

    Input:
      input_ids: (B, L) LongTensor
      lengths:   (B,) LongTensor  (# of non-pad tokens, <= L)

    Output:
      logits: (B,) FloatTensor  (use BCEWithLogitsLoss)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 1,
        rnn_type: str = "lstm",            # "lstm" or "gru"
        bidirectional: bool = True,
        dropout: float = 0.2,              # applied between RNN layers if num_layers>1
        pad_idx: int = 0,
    ):
        super().__init__()

        rnn_type = rnn_type.lower()
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")

        self.pad_idx = pad_idx
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=(dropout if num_layers > 1 else 0.0),
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        lengths must be on CPU for pack_padded_sequence in many setups.
        We'll safely move it to CPU.
        """
        x = self.embedding(input_ids)  # (B, L, E)

        # pack padded to ignore PAD in RNN computation
        lengths_cpu = lengths.detach().to("cpu")
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        packed_out, h = self.rnn(packed)

        if self.rnn_type == "lstm":
            h_n, c_n = h  # h_n: (num_layers*num_dirs, B, H)
        else:
            h_n = h       # (num_layers*num_dirs, B, H)

        # take last layer's hidden states
        if self.bidirectional:
            # last layer forward is -2, backward is -1
            h_f = h_n[-2]  # (B, H)
            h_b = h_n[-1]  # (B, H)
            feat = torch.cat([h_f, h_b], dim=1)  # (B, 2H)
        else:
            feat = h_n[-1]  # (B, H)

        feat = self.dropout(feat)
        logits = self.fc(feat).squeeze(1)  # (B,)
        return logits
