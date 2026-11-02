#!/usr/bin/env python3
"""G1: BEAM-100K, U=20 and U=40, 800 questions. The library builds each conversation under preset beam.u20 (exact),
embeds on CPU, searches with the published reranker cache (read-only), and renders R4 from the selected ids.

Compared per question with (a) the harness's CPU replay of the same legs: the channel heads ce100, sparse30,
grank30, the pool, the fed ids and the reach counts; (b) the published R4 contexts the paper's reads consumed.

  EXPERIMENT_DIR=<tree> GATE_DIR=<gate dir with results/gate_u*_100k.jsonl, bundle_out/replay_r4_u*_100k.jsonl,
  ce_cache/> python tests/studio/g1_beam100k.py
"""
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT]
import pandas as pd

import muster
from muster.compat.ce_cache import LegacyCECache
from muster.models import Models

M, G = os.environ["EXPERIMENT_DIR"], os.environ["GATE_DIR"]
ROWS = [int(x) for x in os.environ.get("ROWS", ",".join(map(str, range(20)))).split(",")]
ARM = "flr_4_3_3"


def jl(p):
    return {(r["row"], r["qi"]): r for r in map(json.loads, open(p)) if True}


def main():
    t0 = time.time()
    df = pd.read_parquet(os.path.join(M, "beam/beam100k.parquet"))
    models = Models(device="cpu", local_files_only=True)
    cache = LegacyCECache(os.path.join(G, "ce_cache"))
    rep = {U: jl(os.path.join(G, "results", "gate_u%d_100k.jsonl" % U)) for U in (20, 40)}
    rr4 = {U: jl(os.path.join(G, "bundle_out", "replay_r4_u%d_100k.jsonl" % U)) for U in (20, 40)}
    pub = {U: jl(os.path.join(M, "bundle_out", "rules", "r4_u%d_100k.jsonl" % U)) for U in (20, 40)}
    tally = {U: dict(q=0, heads=0, pool=0, fed=0, reach=0, r4_replay=0, r4_published=0, misses=0) for U in (20, 40)}
    diffs = []
    for row in ROWS:
        mem = muster.build(str(df["context"].iloc[row]), "beam.u20", models=models)
        sc = cache.load(mem.units)
        qs = list(df["questions"].iloc[row])
        for U in (20, 40):
            for qi, q in enumerate(qs):
                k = (row, qi)
                sel = mem.search(str(q), U=U, cache=sc)
                ctx = mem.render(sel)
                h = rep[U][k]
                t = tally[U]
                t["q"] += 1
                t["misses"] += sel.ce_misses
                ok_h = (sel.heads["ce100"] == h["ce100"] and sel.heads["sparse30"] == h["sparse30"]
                        and sel.heads["grank30"] == h["grank30"])
                ok_p = list(sel.pool) == h["poolids|" + ARM]
                ok_f = sorted(sel.ids) == h["fedk|" + ARM]
                ok_r = sel.n_reach == h["n_reach"] and sel.n_reach_sess == h["n_reach_sess"]
                ok_4 = ctx.ctxhash == rr4[U][k]["ctxhash"]
                ok_o = ctx.ctxhash == pub[U][k]["ctxhash"]
                for name, ok in (("heads", ok_h), ("pool", ok_p), ("fed", ok_f), ("reach", ok_r),
                                 ("r4_replay", ok_4), ("r4_published", ok_o)):
                    t[name] += ok
                if not (ok_h and ok_p and ok_f and ok_r and ok_4 and ok_o):
                    diffs.append((U, row, qi, ok_h, ok_p, ok_f, ok_r, ok_4, ok_o))
        print("[%5.0fs] row %d done (%d units)" % (time.time() - t0, row, len(mem)), flush=True)
    for U in (20, 40):
        t = tally[U]
        print("U=%d: %d questions | heads %d | pool %d | fed ids %d | reach %d | R4 = harness replay %d | "
              "R4 = published %d | cache misses %d" % (U, t["q"], t["heads"], t["pool"], t["fed"], t["reach"],
                                                       t["r4_replay"], t["r4_published"], t["misses"]))
    for d in diffs[:20]:
        print("DIFF U=%d row %d qi %d heads=%s pool=%s fed=%s reach=%s r4_replay=%s r4_published=%s" % d)
    print("%.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
