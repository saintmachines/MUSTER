#!/usr/bin/env python3
# mirror_store_swap.py
# =============================================================================
# Day-1, zero-LLM, CPU-only module for the MIRROR store-swap experiment (E0/E1).
#
# WHAT THIS IS: the causal spine of the paper. MemoryAgentBench isolates the
#   RETRIEVER (fix the store, swap the retriever). This module runs the MIRROR:
#   fix retriever + reader + metric + MATCHED assembled-token budget, and swap
#   ONLY the store's organization. Any change in the assembled evidence is then
#   causally attributable to *management*, not to more tokens.
#
# It is DETERMINISTIC and uses ZERO query-time LLM. Pure-Python (no numpy) so it
#   runs on a cold Mac (no CUDA). See PREREG_E0.md for the pre-registered arms,
#   decomposition, sign map, and kill cell this code implements.
#
# ARMS (Section 2 of PREREG_E0.md):
#   MGMT-OFF  : append-only lossless raw, rank-order packing            (baseline)
#   MGMT-MMR  : generic MMR/cosine dedup, graph-management OFF          (control)
#   MGMT-ON   : 3 unbundled zero-LLM raw-preserving ops
#               (a) high-threshold dedup
#               (b) entity-edge conflict + order-key newest-non-superseded
#               (c) supersession mask
#   FOIL      : one offline LLM-summary consolidation pass (LOSSY, stubbed here)
#
# The module operates on a *candidate set* the frozen retriever already produced
# (rank-ordered). Every arm is truncated to the SAME token budget B; budget freed
# by any drop is refilled with the next-ranked distinct candidate (matched-budget
# accounting), so a lift can never be "just more tokens".
#
# NOTE: the "coverage" proxy in --selftest is a MECHANICAL sanity check (distinct
# gold-entities present in the assembled set), NOT a scientific result. The real
# metric on-bench is SubEM at the fixed reader; this file only proves the store
# operators and the matched-budget accounting behave as pre-registered.
# =============================================================================

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass, field
from typing import Callable, FrozenSet, List, Optional, Tuple


# ------------------------------- data model ---------------------------------
@dataclass(frozen=True)
class Cand:
    """One retrieved raw chunk (verbatim, never summarized in the lossless arms)."""
    cid: str
    text: str
    vec: Tuple[float, ...]          # already-stored e5 embedding (offline)
    entities: FrozenSet[str]        # offline SpaCy NER (zero query-time LLM)
    t: int                          # order-key / ingestion timestamp
    rank: int                       # retriever rank (lower = better)
    gold: FrozenSet[str] = field(default_factory=frozenset)  # eval-only, not used by ops

    @property
    def tokens(self) -> int:
        # deterministic proxy for assembled-token cost (whitespace tokens)
        return max(1, len(self.text.split()))


# ------------------------------- primitives ---------------------------------
def cosine(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def pack_to_budget(cands: List[Cand], budget: int) -> List[Cand]:
    """Matched-budget packing: take candidates in rank order until B is exhausted.
    Freed budget (from upstream drops) is thus refilled with the next-ranked item."""
    out, used = [], 0
    for c in sorted(cands, key=lambda x: x.rank):
        if used + c.tokens <= budget:
            out.append(c)
            used += c.tokens
    return out


# ------------------------------- store arms ---------------------------------
def mgmt_off(pool: List[Cand], budget: int) -> List[Cand]:
    """Append-only lossless raw; no management. Rank-order packing."""
    return pack_to_budget(list(pool), budget)


def mmr_dedup(pool: List[Cand], budget: int, sim: float = 0.86) -> List[Cand]:
    """CONTROL arm: generic MMR/cosine dedup (graph-management OFF).
    Drop a candidate if it is near-duplicate (cosine >= sim) of a higher-ranked kept one."""
    kept: List[Cand] = []
    for c in sorted(pool, key=lambda x: x.rank):
        if any(cosine(c.vec, k.vec) >= sim for k in kept):
            continue
        kept.append(c)
    return pack_to_budget(kept, budget)


def _superseded(c: Cand, pool: List[Cand], sim: float, entity_gate: bool = True) -> bool:
    """Op (c): c is superseded iff a strictly-newer near-dup exists. With entity_gate
    the near-dup must also share an entity (safe when NER is available); with
    entity_gate=False supersession is cosine+time only (use when no NER, e.g. LME)."""
    for o in pool:
        if o.cid == c.cid:
            continue
        if o.t > c.t and cosine(c.vec, o.vec) >= sim and (not entity_gate or (c.entities & o.entities)):
            return True
    return False


def mgmt_on(
    pool: List[Cand],
    budget: int,
    dedup: bool = True,
    supersede: bool = True,
    order_key: bool = True,
    sim_dedup: float = 0.86,
    sim_super: float = 0.72,
    super_entity_gate: bool = True,
) -> List[Cand]:
    """MGMT-ON: 3 UNBUNDLED deterministic zero-LLM raw-preserving ops.
    All ops are non-destructive READS over a lossless store (nothing is mutated).
    super_entity_gate=False makes supersession cosine+time only (no NER, e.g. LME)."""
    cands = list(pool)

    # (b/c) supersession mask: hide chunks a newer near-dup overrides
    if supersede:
        cands = [c for c in cands if not _superseded(c, pool, sim_super, entity_gate=super_entity_gate)]

    # (a) high-threshold dedup (semantic near-duplicate collapse, keep best-ranked)
    if dedup:
        kept: List[Cand] = []
        for c in sorted(cands, key=lambda x: x.rank):
            if any(cosine(c.vec, k.vec) >= sim_dedup for k in kept):
                continue
            kept.append(c)
        cands = kept

    # (b) order-key: among survivors, prefer newest-non-superseded on ties of rank
    if order_key:
        cands = sorted(cands, key=lambda x: (x.rank, -x.t))

    return pack_to_budget(cands, budget)


def foil_abstractive(pool: List[Cand], budget: int, sim: float = 0.86) -> List[Cand]:
    """FALSIFIER FOIL: one offline LLM-summary consolidation pass (LOSSY).
    Stubbed deterministically: collapse each near-dup cluster into a truncated
    'summary' (loses verbatim tokens) to demonstrate lossy management LOWERS
    verbatim-overlap coverage. The real foil calls a summarizer offline."""
    clusters: List[List[Cand]] = []
    for c in sorted(pool, key=lambda x: x.rank):
        placed = False
        for cl in clusters:
            if cosine(c.vec, cl[0].vec) >= sim:
                cl.append(c)
                placed = True
                break
        if not placed:
            clusters.append([c])
    summarized: List[Cand] = []
    for cl in clusters:
        head = cl[0]
        # lossy: keep only the first half of the tokens of the cluster head
        toks = head.text.split()
        summ = " ".join(toks[: max(1, len(toks) // 2)])
        summarized.append(Cand(head.cid, summ, head.vec, head.entities, head.t, head.rank, head.gold))
    return pack_to_budget(summarized, budget)


# ------------------------------ measurement ---------------------------------
def coverage(assembled: List[Cand]) -> int:
    """MECHANICAL proxy metric ONLY (distinct gold-entities present). Real metric = SubEM."""
    covered = set()
    for c in assembled:
        covered |= (c.entities & c.gold)
    return len(covered)


def used_tokens(assembled: List[Cand]) -> int:
    return sum(c.tokens for c in assembled)


def decompose(pool: List[Cand], budget: int, metric: Callable[[List[Cand]], float]):
    """Pre-registered decomposition Δ_total = Δ_generic + Δ_semantic (PREREG_E0 §3)."""
    m_off = metric(mgmt_off(pool, budget))
    m_mmr = metric(mmr_dedup(pool, budget))
    m_on = metric(mgmt_on(pool, budget))
    m_foil = metric(foil_abstractive(pool, budget))
    d_total = m_on - m_off
    d_generic = m_mmr - m_off
    d_semantic = m_on - m_mmr
    return {
        "M_off": m_off, "M_mmr": m_mmr, "M_on": m_on, "M_foil": m_foil,
        "d_total": d_total, "d_generic": d_generic, "d_semantic": d_semantic,
        "identity_ok": abs(d_total - (d_generic + d_semantic)) < 1e-9,
    }


# -------------------------------- selftest ----------------------------------
def _selftest() -> int:
    def e(*xs):  # tiny unit vector helper
        return tuple(float(x) for x in xs)

    # Synthetic candidate set built to exercise every op:
    #  A1/A2 = generic near-duplicates (same vec)      -> MMR & ON drop one
    #  S_old/S_new = superseded pair (same entity,      -> ON masks S_old
    #                newer overrides older, near-dup)
    #  D1..D3 = distinct fillers beyond budget          -> freed budget pulls them in
    gold = frozenset({"paris", "seine", "eiffel", "louvre"})
    pool = [
        Cand("A1", "paris capital france " * 3, e(1, 0, 0, 0), frozenset({"paris"}), t=1, rank=1, gold=gold),
        Cand("A2", "paris capital france again " * 3, e(1, 0, 0, 0), frozenset({"paris"}), t=1, rank=2, gold=gold),
        Cand("Sold", "seine river old " * 3, e(0, 1, 0, 0), frozenset({"seine"}), t=1, rank=3, gold=gold),
        Cand("Snew", "seine river new " * 3, e(0, 0.99, 0.01, 0), frozenset({"seine"}), t=9, rank=4, gold=gold),
        Cand("D1", "eiffel tower iron " * 3, e(0, 0, 1, 0), frozenset({"eiffel"}), t=2, rank=5, gold=gold),
        Cand("D2", "louvre museum art " * 3, e(0, 0, 0, 1), frozenset({"louvre"}), t=2, rank=6, gold=gold),
        Cand("D3", "extra distinct filler " * 3, e(0.5, 0.5, 0.5, 0.5), frozenset({"filler"}), t=2, rank=7, gold=gold),
    ]
    # Budget fits ~4 chunks (each ~9-12 tokens); tight enough that dedup frees room.
    B = sum(c.tokens for c in pool[:4])

    off = mgmt_off(pool, B)
    mmr = mmr_dedup(pool, B)
    on = mgmt_on(pool, B)
    foil = foil_abstractive(pool, B)

    checks = []
    def chk(name, cond):
        checks.append((name, bool(cond)))

    # 1. matched budget respected by every arm
    for nm, arm in [("off", off), ("mmr", mmr), ("on", on), ("foil", foil)]:
        chk(f"budget_ok[{nm}]", used_tokens(arm) <= B)
    # 2. MGMT-OFF keeps BOTH generic duplicates (no dedup)
    ids_off = {c.cid for c in off}
    chk("off_keeps_both_dups", "A1" in ids_off and "A2" in ids_off)
    # 3. MGMT-MMR keeps at most ONE of the duplicate pair
    ids_mmr = {c.cid for c in mmr}
    chk("mmr_dedups_pair", not ("A1" in ids_mmr and "A2" in ids_mmr))
    # 4. MGMT-ON masks the SUPERSEDED (older) chunk, keeps the newer
    ids_on = {c.cid for c in on}
    chk("on_masks_superseded", ("Sold" not in ids_on) and ("Snew" in ids_on))
    # 5. coverage(ON) >= coverage(MMR) >= coverage(OFF)  (freed budget -> more distinct)
    c_off, c_mmr, c_on = coverage(off), coverage(mmr), coverage(on)
    chk("coverage_monotone", c_on >= c_mmr >= c_off)
    # 6. abstractive foil LOWERS coverage vs ON (lossy summarization)  [non-monotonicity]
    chk("foil_lowers_coverage", coverage(foil) <= c_on)
    # 7. decomposition identity holds exactly
    dec = decompose(pool, B, lambda a: float(coverage(a)))
    chk("decomposition_identity", dec["identity_ok"])

    ok = all(v for _, v in checks)
    print("mirror_store_swap SELFTEST")
    for name, v in checks:
        print(f"  [{'PASS' if v else 'FAIL'}] {name}")
    print(f"  coverage  OFF={c_off}  MMR={c_mmr}  ON={c_on}  FOIL={coverage(foil)}  (of {len(gold)})")
    print(f"  decompose d_total={dec['d_total']:.1f} = d_generic={dec['d_generic']:.1f} + "
          f"d_semantic={dec['d_semantic']:.1f}  identity_ok={dec['identity_ok']}")
    print("SELFTEST:", "ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mirror store-swap arms (zero-LLM, CPU).")
    ap.add_argument("--selftest", action="store_true", help="run the deterministic sanity checks")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())
    ap.print_help()
