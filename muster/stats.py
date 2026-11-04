"""The two statistics that decide a store's representation, computed from its text with no model.

serial fraction  the share of non-empty lines (first 20,000) that open with "<integer>. <body>"
record share     the token-weighted share of blank-line records of at most the reference chunk

`store_stats` is scripts/derive.py:69-79 (one store, CRLF normalized, special-token strings read as text).
`collection_stats` is harness/derive_config.py:43-64 (the first rows of a benchmark cell, pooled), which the paper's
per-cell decisions used.

`lead_serial` is not in the paper: it finds a log that numbers its records in its own style ("Step 3: ..."),
and only `derive` reads it, after the serial fraction has failed.
"""
import re
import statistics
from dataclasses import dataclass

from . import bpe
from .records import normalize_newlines

NUM = re.compile(r"^\s*\d+\.\s+\S")
MAX_LINES = 20000
# a serial under the log's own spelling: an optional bracket and word, the integer, an optional closer, then the body
# ("Step 3: ", "#3 ", "(3) ", "[3] ", "3. "). Thresholds set after the paper (2026-11) on public benchmark logs.
LEAD = re.compile(r"^(\s*[\[(#]?\s*(?:[A-Za-z]+\s*)?)(\d{1,8})([\]).:]?\s+)(?=\S)")
LEAD_TAU, LEAD_LINE_TOKENS = 0.95, 20


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


def lead_match(s):
    """LEAD at the start of `s` when its integer ends within the first 32 characters, else None."""
    m = LEAD.match(s)
    return m if m and m.end(2) <= 32 else None


def lead_serial(text):
    """(share of lines after the first, among the first 20,000 non-empty lines, whose leading serial is the previous
    line's plus one, median cl100k tokens per line). The median is counted only when the share reaches LEAD_TAU, else
    it is None. A serial must end within a line's first 32 characters."""
    lines = [ln for ln in normalize_newlines(text).split("\n") if ln.strip()][:MAX_LINES]
    if not lines:
        return 0.0, None
    prev, step = None, 0
    for ln in lines:
        m = lead_match(ln)
        v = int(m.group(2)) if m else None
        if v is not None and prev is not None and v == prev + 1:
            step += 1
        prev = v
    share = step / float(max(len(lines) - 1, 1))
    if share < LEAD_TAU:
        return share, None
    enc = bpe.cl100k()
    return share, statistics.median([bpe.count(enc, ln) for ln in lines])


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
