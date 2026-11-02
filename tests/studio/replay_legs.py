#!/usr/bin/env python3
"""Replay published legs with the library (exact presets, CPU embeddings, the published reranker cache read-only) and
compare every question with (a) the published dump's channel heads, pool and fed ids, and (b) the published context
the paper's reads consumed (flat bundle, R4 bundle, or the LoCoMo R4 bundle).

Only questions whose reranker scores are all in the published cache are compared exactly; the others are counted
as "cold" and skipped, never scored live (CPU scores differ in the last bits from the MPS / CUDA ones).

  EXPERIMENT_DIR=<tree> CE_DIRS=<a:b> LEGS=lz_shqa,lcm_u10 python tests/studio/replay_legs.py
"""
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT]
import pandas as pd

import muster
from muster import presets
from muster.compat.ce_cache import LegacyCECache, qkey
from muster.models import Models

M = os.environ["EXPERIMENT_DIR"]
CE_DIRS = [d for d in os.environ.get("CE_DIRS", "").split(":") if d]
LEGS = [x for x in os.environ["LEGS"].split(",") if x]


def dumps_of(tag):
    fs = sorted(glob.glob(os.path.join(M, "len0928_in", tag + "_s*.jsonl")))
    if not fs:
        fs = sorted(glob.glob(os.path.join(M, "results", tag + "_s*.jsonl"))) or \
             sorted(glob.glob(os.path.join(M, "results", tag + ".jsonl")))
    recs = {}
    for f in fs:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                recs[(r["row"], r["qi"])] = r
    return recs


def published(tag, U):
    c = tag[3:] if tag.startswith("lz_") else tag
    for p in ("bundle_out/rules/r4_mab_%s.jsonl" % c, "bundle_out/len0928/full_%s.jsonl" % tag,
              "bundle_out/locomo/%s_r4.jsonl" % tag, "bundle_out/mab_glm/%s_notag.jsonl" % tag):
        if os.path.exists(os.path.join(M, p)):
            if "beam" in tag:
                break
            return p, {(r["row"], r["qi"]): r["ctxhash"] for r in map(json.loads, open(os.path.join(M, p)))}
    if c in ("b77", "clin", "nlu", "trcC", "trcF"):
        # the five in-context-learning cells were read from bundles packed under another label: compare with the
        # context hash the board's GPT-4o-mini read recorded for every question
        p = "results/nx_ra_%s.jsonl" % c
        return p, {(r["row"], r["qi"]): r["ctxhash|flrs_4_3_3|prune_annot"]
                   for r in map(json.loads, open(os.path.join(M, p))) if True}
    if "beam" in tag:
        size = tag.split("_beam")[1]
        p = "bundle_out/rules/r4_u%d_%s.jsonl" % (U, size)
        return p, {(r["row"], r["qi"]): r["ctxhash"] for r in map(json.loads, open(os.path.join(M, p)))}
    return None, {}


def cache_for(units, knobs):
    for d in CE_DIRS:
        c = LegacyCECache(d, knobs=knobs)
        if c.has(units):
            return c.load(units)
    return {}


COLD = os.environ.get("COLD", "") not in ("", "0")    # score missing pairs live on CPU and still compare
_MODELS = {}


def models_for(plan):
    k = (plan.encoder, plan.reranker)
    if k not in _MODELS:
        _MODELS[k] = Models(device="cpu", local_files_only=True, encoder_spec=plan.encoder, reranker_spec=plan.reranker)
    return _MODELS[k]


def main():
    t0 = time.time()
    allok = True
    for tag in LEGS:
        leg = presets.legs()[tag]
        plan = presets.get(tag)
        arm = leg["arm"].split("|")[0]
        df = pd.read_parquet(os.path.join(M, leg["parq"]))
        models = models_for(plan)
        recs = dumps_of(tag)
        pub_path, pub = published(tag, plan.U)
        t = dict(q=0, cold=0, heads=0, pool=0, fed=0, reach=0, context=0)
        diffs = []
        for row in leg["rows"]:
            keys = sorted(k for k in recs if k[0] == row)
            if not keys:
                continue
            mem = muster.build(str(df["context"].iloc[row]), plan, models=models, embed=False)
            sc = cache_for(list(mem.units), plan.compat["knobs"])
            qs = list(df["questions"].iloc[row])
            warm = keys if COLD else [k for k in keys if qkey(str(qs[k[1]])) in sc]
            t["cold"] += len(keys) - len(warm)
            if not warm:
                continue
            mem.embed(models)
            for k in warm:
                q = str(qs[k[1]])
                h = recs[k]
                sel = mem.search(q, cache=sc, models=models)
                if sel.ce_misses and not COLD:   # a question the cache only partly covers is cold, not compared
                    t["cold"] += 1
                    continue
                ctx = mem.render(sel)
                t["q"] += 1
                rec_heads = all(x in h for x in ("ce100", "sparse30", "grank30"))
                ok = dict(heads=((sel.heads["ce100"] == h["ce100"] and sel.heads["sparse30"] == h["sparse30"]
                                  and sel.heads["grank30"] == h["grank30"]) if rec_heads else None),
                          pool=(list(sel.pool) == h["poolids|" + arm]) if ("poolids|" + arm) in h else None,
                          # numbered logs: the dump's fed ids are the pool before the newest-per-slot walk
                          fed=((sorted(sel.pool) if mem.index.structural else sorted(sel.ids)) == h["fedk|" + arm])
                          if ("fedk|" + arm) in h else None,
                          reach=(sel.n_reach == h["n_reach"] and sel.n_reach_sess == h["n_reach_sess"])
                          if "n_reach" in h else None,
                          context=(pub.get(k) == ctx.ctxhash))
                for n, v in ok.items():
                    if v is None:
                        t["n/a " + n] = t.get("n/a " + n, 0) + 1
                    else:
                        t[n] += v
                if not all(v for v in ok.values() if v is not None):
                    why = {n: v for n, v in ok.items() if v is False}
                    for hn in ("ce100", "sparse30", "grank30") if rec_heads else ():
                        a, b = sel.heads[hn], h[hn]
                        if a != b:
                            j = next((x for x in range(min(len(a), len(b))) if a[x] != b[x]), min(len(a), len(b)))
                            why[hn] = "first difference at %d: ours %s, dump %s; same set: %s" % (
                                j, a[j:j + 3], b[j:j + 3], sorted(a) == sorted(b))
                    diffs.append((k, why))
        good = t["q"] and all(t[n] + t.get("n/a " + n, 0) == t["q"] for n in ("heads", "pool", "fed", "reach", "context"))
        allok &= bool(good)
        na = lambda n: ("n/a" if t.get("n/a " + n) == t["q"] else str(t[n]))
        print("%-16s %-9s warm %4d (cold %4d) | heads %4s | pool %4s | fed %4s | reach %4s | context = published %4d  [%s] %s"
              % (tag, plan.source[7:], t["q"], t["cold"], na("heads"), na("pool"), na("fed"), na("reach"), t["context"],
                 pub_path, "%.0fs" % (time.time() - t0)), flush=True)
        for d in diffs:
            print("    DIFF", d, flush=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
