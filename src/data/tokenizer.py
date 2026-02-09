# src/data/tokenizer.py
from __future__ import annotations

import json
import re
import html
from dataclasses import dataclass, asdict
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional


@dataclass
class VocabConfig:
    top_n: int = 20000 # 단어장에 등록할 최대 단어 수
    seq_len: int = 256 # 모델에 입력될 시퀀스의 고정 길이 (길면 자르고, 짧으면 패딩)
    lower: bool = True # 대소문자 구분 없앨지 여부 (소문자화)
    # keep punctuation tokens like "!" "?" etc. (recommended for sentiment)
    keep_punct: bool = True # 문장부호 별도의 토큰으로 살려둘지 여부

    pad_token: str = "<PAD>" # 인덱스 0 : 길이 맞추기 위한 패딩 토큰
    unk_token: str = "<UNK>" # 인덱스 1 : OoV 처리하는 Unknown 토큰
    pad_idx: int = 0
    unk_idx: int = 1

    # For reproducibility / reporting
    tokenizer_name: str = "simple_word_level_v1"


class WordTokenizer:
    """
    Word-level tokenizer + Top-N vocabulary builder.
    - <PAD>=0, <UNK>=1 fixed
    - encode() returns fixed-length seq_len with truncate/pad
    """

    # Regex:
    # - words with optional internal apostrophe (don't, it's)
    # - numbers
    # - punctuation as separate tokens (if keep_punct)
    _WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
    _PUNCT_RE = re.compile(r"[!?.,;:()\[\]\"-]")

    # Basic HTML tag remover (handles <br />, <br>, etc.)
    _HTML_TAG_RE = re.compile(r"<[^>]+>")

    def __init__(self, config: Optional[VocabConfig] = None):
        self.cfg = config or VocabConfig()
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []

    # ---------- Tokenization ----------
    def tokenize(self, text: str) -> List[str]:
        """
        Deterministic tokenization:
        1) HTML unescape
        2) remove tags
        3) optional lowercasing
        4) extract word tokens (+ optional punctuation tokens)
        """
        if not isinstance(text, str):
            text = str(text)

        text = html.unescape(text)
        text = self._HTML_TAG_RE.sub(" ", text)

        if self.cfg.lower:
            text = text.lower()

        # Find words and (optionally) punctuation; keep order by scanning
        tokens: List[str] = []
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]

            # Skip whitespace
            if ch.isspace():
                i += 1
                continue

            # Word match at position i
            m = self._WORD_RE.match(text, i)
            if m:
                tokens.append(m.group(0))
                i = m.end()
                continue

            # Punctuation token
            if self.cfg.keep_punct:
                m2 = self._PUNCT_RE.match(text, i)
                if m2:
                    tokens.append(m2.group(0))
                    i = m2.end()
                    continue

            # Otherwise skip unknown char
            i += 1

        return tokens

    # ---------- Vocab ----------
    def build_vocab(self, texts: Iterable[str]) -> None:
        """
        Build top-N vocab from *train texts only*.
        Deterministic ordering: freq desc, token asc.
        """
        counter = Counter()
        for t in texts:
            counter.update(self.tokenize(t))

        # Sort deterministically: (-freq, token)
        items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        top_items = items[: self.cfg.top_n]
        vocab_tokens = [tok for tok, _ in top_items]

        # special tokens first, fixed indices
        self.itos = [self.cfg.pad_token, self.cfg.unk_token] + vocab_tokens
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

        # Enforce indices (safety)
        assert self.stoi[self.cfg.pad_token] == self.cfg.pad_idx
        assert self.stoi[self.cfg.unk_token] == self.cfg.unk_idx

    def is_built(self) -> bool:
        return len(self.stoi) > 0 and len(self.itos) > 0

    # ---------- Encode / Decode ----------
    def encode(self, text: str) -> Tuple[List[int], int]:
        """
        Returns (ids_fixed_len, length_before_pad).
        length_before_pad is clipped to seq_len (after truncation).
        """
        if not self.is_built():
            raise RuntimeError("Vocab not built. Call build_vocab(train_texts) first.")

        tokens = self.tokenize(text)
        ids = [self.stoi.get(tok, self.cfg.unk_idx) for tok in tokens]

        # Truncate
        if len(ids) > self.cfg.seq_len:
            ids = ids[: self.cfg.seq_len]

        length = len(ids)

        # Pad to fixed seq_len
        if length < self.cfg.seq_len:
            ids = ids + [self.cfg.pad_idx] * (self.cfg.seq_len - length)

        return ids, length

    def decode(self, ids: List[int]) -> str:
        if not self.is_built():
            raise RuntimeError("Vocab not built/loaded.")
        toks = []
        for i in ids:
            if 0 <= i < len(self.itos):
                toks.append(self.itos[i])
            else:
                toks.append(self.cfg.unk_token)
        return " ".join(toks)

    # ---------- Save / Load ----------
    def save(self, path: str | Path) -> None:
        """
        Save vocab + config to json.
        """
        if not self.is_built():
            raise RuntimeError("Nothing to save. Build vocab first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": asdict(self.cfg),
            "itos": self.itos,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        cfg = VocabConfig(**payload["config"])
        tok = cls(cfg)
        tok.itos = payload["itos"]
        tok.stoi = {t: i for i, t in enumerate(tok.itos)}

        # Enforce fixed indices (safety)
        assert tok.stoi[cfg.pad_token] == cfg.pad_idx
        assert tok.stoi[cfg.unk_token] == cfg.unk_idx
        return tok
