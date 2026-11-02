#!/usr/bin/env python3
r"""LEDGER as a PARENT LAYER: what the FactConsolidation profile does by hand, for any store.

WHERE THE PARENT LAYER COMES FROM TODAY (lme_pf12.py:1157-1194) - three sources:

  PROFILES[ds]     hand-registered, and the registry has exactly ONE entry, factconsolidation.
                   It replaces unit, parent and order together: unit = one fact line, parent =
                   the semantic SLOT, order = the explicit serial.  That substitution is worth
                   5/100 -> 89/100 on that cell, and it is the single largest effect in the
                   whole system.
  detect_parents   text markers - CHAPTER, document blocks, headings.  On every ICL set and on
                   recsys it finds nothing usable and falls back to per_unit, so units == parents
                   == 1.0 and every slot-based view collapses to composite.  Measured today:
                   prune and composite feed BYTE-IDENTICAL context on 100% of questions on all
                   six of those cells, which is why their "tie" carries no information.
  legacy           per-unit or the eventqa chapter rule.

So the system's biggest lever is available on one dataset because someone wrote a profile for
it.  The LEDGER already computes the same three things for every store, at write time, and it
has been treated as a source of a STORE-LEVEL STATISTIC only:

    unit    one extracted assertion            (the ledger's "attribute :: value")
    parent  the slot                           (the attribute, normalised)
    order   the ingest index                   (free, and totally ordered by construction)

That is the profile interface exactly.  Nothing downstream of the loader is dataset-aware, so
substituting it needs no new read-time machinery at all.

THE CHEAPEST FALSIFIER, AND IT RUNS FIRST.  The reader can only answer from what is in its
window, so if extraction drops the answer strings the design is dead no matter how good the
slot layer is.  This module therefore reports, per store and BEFORE any paid run:

    gold_chunk    share of questions whose answer appears in the raw chunk corpus
    gold_ledger   share of questions whose answer survives into some extracted unit
    keep          gold_ledger / gold_chunk

A store where `keep` is low cannot be run this way, and saying so costs nothing.  $0: this
reads the cache LEDGER v2 already paid for and never calls the API.
"""
import collections
import hashlib
import json
import os
import re
import sys
import unicodedata

import pandas as pd

CHUNK = int(os.environ.get("CHUNK", "512"))
MODEL = os.environ.get("OAMODEL", "gpt-4o-mini-2024-07-18")
CACHE = os.environ.get("LEDGER_CACHE", "ledger_v2_cache")

PROMPT = (
    "From the passage below, extract EVERY factual assertion it states.\n"
    "Output one per line, exactly as:   attribute :: value :: kind :: time\n"
    "  attribute  the thing being described\n"
    "  value      what is said about it\n"
    "  kind       CORRECTION if the passage signals this replaces or fixes an earlier value\n"
    "             (updated to, corrected, actually, no longer, in fact, revised);\n"
    "             STATE if it describes how things were or became at some point\n"
    "             (then, later, after that, in 2019, by the end);\n"
    "             UNKNOWN if the passage gives no such signal.\n"
    "  time       the time the passage states for this assertion, verbatim; empty if none.\n"
    "Do not skip an assertion because it looks like it repeats or contradicts another.\n"
    "Max 40 lines. No other text.\n\nPassage:\n%s\n\nAssertions:")


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def slotkey(a):
    a = re.sub(r"[^a-z0-9 ]", " ", (a or "").lower())
    return " ".join(sorted(set(w for w in a.split() if len(w) > 2)))[:80]


def chunks(text, size):
    w = text.split()
    n = max(1, int(size * 0.75))
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)]


def cached(text):
    k = hashlib.sha1(("%s\x00%s\x00%s" % (MODEL, PROMPT, text)).encode()).hexdigest()
    fp = os.path.join(CACHE, k[:2], k + ".txt")
    return open(fp, encoding="utf-8").read() if os.path.exists(fp) else None


def build(text, chunk=None):
    """(units, parents, order, n_missing_chunks).  Never calls the API."""
    ch = chunks(text, chunk or CHUNK)
    units, par, order = [], [], []
    miss = 0
    for i, c in enumerate(ch):
        out = cached(c)
        if out is None:
            miss += 1
            continue
        for line in out.split("\n"):
            p = [z.strip() for z in line.split("::")]
            if len(p) >= 2 and p[0] and p[1]:
                k = slotkey(p[0])
                if not k:
                    continue
                # The unit the reader sees keeps the attribute, because a bare value is not
                # answerable: "Paris" alone does not say what it is the capital of.
                units.append("%s: %s" % (p[0], p[1]))
                par.append(k)
                order.append(i)
    return units, par, order, miss, len(ch)


def cover(units, answers):
    """Share of questions whose answer string appears in SOME unit."""
    body = [norm(u) for u in units]
    joined = "\n".join(body)
    n = 0
    for av in answers:
        vals = list(av) if not isinstance(av, str) else [av]
        if any(norm(a) and norm(a) in joined for a in vals):
            n += 1
    return n / max(len(answers), 1)


if __name__ == "__main__":
    AR = "data/Accurate_Retrieval-00000-of-00001.parquet"
    FCP = "data_all/data/Conflict_Resolution-00000-of-00001.parquet"
    LRU = "data_all/data/Long_Range_Understanding-00000-of-00001.parquet"
    TTL = "data_all/data/Test_Time_Learning-00000-of-00001.parquet"
    JOBS = json.loads(os.environ.get("JOBS", "null")) or [
        ["FC-MH", FCP, 3], ["ICL-b77", TTL, 1], ["ICL-trf", TTL, 5], ["recsys", TTL, 0],
        ["SH-QA", AR, 0], ["MH-QA", AR, 1], ["EventQA", AR, 2], ["LME", AR, 17],
        ["DetQA", LRU, 100]]
    print("LEDGER as a parent layer. $0 - cached extractions only.")
    print("A store can only be run this way if the answers SURVIVE extraction.\n")
    print("%-9s %7s %7s %6s | %9s %9s %6s | %s"
          % ("cell", "units", "slots", "u/par", "gold_chunk", "gold_ledg", "keep", "verdict"))
    for name, pq, row in JOBS:
        df = pd.read_parquet(pq)
        text = str(df["context"].iloc[int(row)])
        ans = list(df["answers"].iloc[int(row)])
        u, p, o, miss, nch = build(text)
        if miss:
            print("%-9s CACHE INCOMPLETE %d/%d chunks - NOT REPORTED" % (name, miss, nch))
            continue
        gc = cover(chunks(text, CHUNK), ans)
        gl = cover(u, ans)
        upar = len(u) / float(max(len(set(p)), 1))
        keep = gl / gc if gc else float("nan")
        print("%-9s %7d %7d %6.2f | %9.3f %9.3f %6.2f | %s"
              % (name, len(u), len(set(p)), upar, gc, gl, keep,
                 "RUNNABLE" if keep >= 0.80 else ("marginal" if keep >= 0.5 else "DEAD")))
    print("LEDGER_PROFILE_DONE", flush=True)


def build2(text, chunk=None):
    """(retrieval_key, reader_text, parents, order, n_missing, n_chunks).

    TWO TEXTS PER UNIT, and they are not the same thing.

      retrieval_key  the assertion, "attribute: value".  This is what gets embedded, what the
                     entity index is built over, and what a query is scored against - a precise,
                     single-claim string, which is the whole reason the ledger is worth having.
      reader_text    what is actually rendered into the reader's window.

    The harness's loader already returns those separately (rtext, rkey), so LPROFILE=2 can keep
    the slot layer WITHOUT paying for it in delivered evidence.  An assertion is roughly ten
    tokens; ten of them is a ~180-token window against the ~5,000 every other arm delivers, so
    LPROFILE=1 is a budget cut disguised as a representation change.  That is the right trade
    where the answer IS a single fact - which is what the hand-written FactConsolidation profile
    does, and it wins 5/100 -> 89/100 there - and the wrong trade wherever the reader needs bulk
    text, most obviously in-context learning, where accuracy tracks how many demonstrations are
    in the window.

      LPROFILE=1   reader sees the assertion    structure, ~180 tokens
      LPROFILE=2   reader sees the SOURCE CHUNK structure at the budget the protocol allows

    build() is kept beside this, unchanged and still returning five values, because lme_pf18 was
    generated against that signature and is running right now.  Changing a function's arity under
    a running job is how the two callers written today both died with
    "not enough values to unpack (expected 6, got 5)".
    """
    ch = chunks(text, chunk or CHUNK)
    key, rd, par, order = [], [], [], []
    miss = 0
    for i, c in enumerate(ch):
        out = cached(c)
        if out is None:
            miss += 1
            continue
        for line in out.split("\n"):
            p = [z.strip() for z in line.split("::")]
            if len(p) >= 2 and p[0] and p[1]:
                k = slotkey(p[0])
                if not k:
                    continue
                key.append("%s: %s" % (p[0], p[1]))
                rd.append(c)
                par.append(k)
                order.append(i)
    return key, rd, par, order, miss, len(ch)
