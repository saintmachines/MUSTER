#!/usr/bin/env python3
# stats.py
# =============================================================================
# Paired bootstrap confidence intervals for the pre-registered deltas
# (PREREG_E0.md §7). Deterministic (seeded), pure-Python, zero dependencies.
# Input = per-item metric differences between two arms (paired by query id).
# =============================================================================
from __future__ import annotations
import argparse
import random
from typing import List, Tuple


def paired_bootstrap(
    deltas: List[float], n_resamples: int = 10000, seed: int = 0, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Return (mean_delta, ci_lo, ci_hi) at the (1-alpha) level via a paired
    bootstrap over items. Resampling items (not arms) preserves the pairing."""
    m = len(deltas)
    if m == 0:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    boot = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(m):
            s += deltas[rng.randrange(m)]
        boot.append(s / m)
    boot.sort()
    lo = boot[int((alpha / 2.0) * n_resamples)]
    hi = boot[min(n_resamples - 1, int((1.0 - alpha / 2.0) * n_resamples))]
    return (sum(deltas) / m, lo, hi)


def significant(ci: Tuple[float, float, float]) -> bool:
    """A delta is 'clean' (outside the CI) iff the CI excludes 0 (PREREG_E0 kill cell)."""
    _, lo, hi = ci
    return lo > 0.0 or hi < 0.0


def _selftest() -> int:
    checks = []
    def chk(name, cond):
        checks.append((name, bool(cond)))

    pos = paired_bootstrap([0.4] * 50 + [0.6] * 50, seed=1)
    chk("all_positive_excludes_zero", pos[1] > 0.0 and significant(pos))
    zero = paired_bootstrap([0.0] * 100, seed=1)
    chk("all_zero_contains_zero", zero[1] == 0.0 and zero[2] == 0.0 and not significant(zero))
    mixed = paired_bootstrap([1.0, -1.0] * 50, seed=1)
    chk("mixed_contains_zero", mixed[1] < 0.0 < mixed[2] and not significant(mixed))
    a = paired_bootstrap([0.3, -0.1, 0.5, 0.2, -0.2, 0.4], seed=7)
    b = paired_bootstrap([0.3, -0.1, 0.5, 0.2, -0.2, 0.4], seed=7)
    chk("deterministic_same_seed", a == b)
    c = paired_bootstrap([0.3, -0.1, 0.5, 0.2, -0.2, 0.4], seed=8)
    chk("varies_by_seed", a[1:] != c[1:])

    ok = all(v for _, v in checks)
    print("stats SELFTEST")
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
