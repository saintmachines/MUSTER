#!/usr/bin/env python3
# metrics.py
# =============================================================================
# Exact-answer metrics for the mirror experiment. SubEM is the MemoryAgentBench
# Accurate-Retrieval metric (leg-2 headline); EM / token-F1 back leg-1 LongBench.
# Pure-Python, deterministic, zero dependencies. SQuAD-style normalization.
# =============================================================================
from __future__ import annotations
import argparse
import re
import string
from typing import List

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def subem(pred: str, golds: List[str]) -> float:
    """Sub-string Exact Match: 1.0 if any normalized gold is a substring of the
    normalized prediction (the MemoryAgentBench AR metric)."""
    p = normalize(pred)
    for g in golds:
        ng = normalize(g)
        if ng and ng in p:
            return 1.0
    return 0.0


def em(pred: str, golds: List[str]) -> float:
    p = normalize(pred)
    return 1.0 if any(normalize(g) == p for g in golds) else 0.0


def token_f1(pred: str, golds: List[str]) -> float:
    pt = normalize(pred).split()
    best = 0.0
    for g in golds:
        gt = normalize(g).split()
        if not pt or not gt:
            best = max(best, 1.0 if pt == gt else 0.0)
            continue
        common = 0
        gt_pool = list(gt)
        for tok in pt:
            if tok in gt_pool:
                gt_pool.remove(tok)
                common += 1
        if common == 0:
            continue
        prec = common / len(pt)
        rec = common / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def _selftest() -> int:
    checks = []
    def chk(name, cond):
        checks.append((name, bool(cond)))

    chk("subem_substring", subem("The capital is Paris, France.", ["paris"]) == 1.0)
    chk("subem_negative", subem("It is Lyon.", ["paris"]) == 0.0)
    chk("subem_articles", subem("the eiffel tower", ["Eiffel Tower"]) == 1.0)
    chk("subem_multi_gold", subem("answer: seine", ["rhine", "seine"]) == 1.0)
    chk("em_exact", em("Paris", ["paris"]) == 1.0)
    chk("em_not_substring", em("Paris, France", ["paris"]) == 0.0)
    chk("f1_partial", abs(token_f1("paris france", ["paris"]) - (2 * 0.5 * 1.0 / 1.5)) < 1e-9)
    chk("f1_zero", token_f1("lyon", ["paris"]) == 0.0)

    ok = all(v for _, v in checks)
    print("metrics SELFTEST")
    for name, v in checks:
        print(f"  [{'PASS' if v else 'FAIL'}] {name}")
    print("SELFTEST:", "ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        raise SystemExit(_selftest())
    ap.print_help()
