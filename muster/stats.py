"""The two statistics that decide a store's representation, computed from its text with no model.

serial fraction  the share of non-empty lines (first 20,000) that open with "<integer>. <body>"
record share     the token-weighted share of blank-line records of at most the reference chunk

`store_stats` is scripts/derive.py:69-79 (one store, CRLF normalized, special-token strings read as text).
`collection_stats` is harness/derive_config.py:43-64 (the first rows of a benchmark cell, pooled), which the paper's
per-cell decisions used.
"""
import re
from dataclasses import dataclass

from . import bpe
from .records import normalize_newlines

NUM = re.compile(r"^\s*\d+\.\s+\S")
MAX_LINES = 20000


@dataclass(frozen=True)
class Stats:
    serial: float
    record_share: float
    tokens: int            # store: tokens in blank-line records; collection: mean whole-row tokens
    lines_used: int
    rows: int = 1
    scope: str = "store"

    def to_dict(self):
        return dict(serial=self.serial, record_share=self.record_share, tokens=self.tokens,
                    lines_used=self.lines_used, rows=self.rows, scope=self.scope)


def _serial(text):
    lines = [ln for ln in text.split("\n") if ln.strip()][:MAX_LINES]
    return (sum(1 for ln in lines if NUM.match(ln)) / float(len(lines)) if lines else 0.0), len(lines)


def store_stats(text: str, reference_chunk: int = 4096) -> Stats:
    enc = bpe.cl100k()
    text = normalize_newlines(text)
    total = fit = 0
    for rec in text.split("\n\n"):
        if rec.strip():
            t = bpe.count(enc, rec)
            total += t
            fit += t if t <= reference_chunk else 0
    serial, n = _serial(text)
    return Stats(serial, (fit / float(total) if total else 0.0), total, n)


def collection_stats(texts, reference_chunk: int, *, sample: int = 3, mode: str = "native") -> Stats:
    enc = bpe.cl100k()
    enc_count = (lambda s: len(enc.encode(s))) if mode == "exact" else (lambda s: bpe.count(enc, s))
    tot = fit = 0
    fracs, toks, used = [], [], 0
    rows = list(texts)[:sample]
    for txt in rows:
        txt = str(txt)
        toks.append(enc_count(txt))
        for p in txt.split("\n\n"):
            if not p.strip():
                continue
            t = enc_count(p)
            tot += t
            if t <= reference_chunk:
                fit += t
        lines = [ln for ln in txt.split("\n") if ln.strip()][:MAX_LINES]
        used += len(lines)
        if lines:
            fracs.append(sum(1 for ln in lines if NUM.match(ln)) / float(len(lines)))
    return Stats(sum(fracs) / len(fracs) if fracs else 0.0, fit / float(tot) if tot else 0.0,
                 (sum(toks) // len(toks)) if toks else 0, used, rows=len(rows), scope="collection")
