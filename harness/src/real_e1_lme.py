#!/usr/bin/env python3
# real_e1_lme.py — REAL-DATA mirror on the TEMPORAL zone (LongMemEval-S, rows 17-21).
# This is the headline zone: dated chat sessions + knowledge-update / temporal-reasoning
# questions, where supersession + order-key (semantic management) can actually act.
# Metric = gold-span answer-recall (Track B, no LLM key). Chunk timestamp = session
# "Chat Time:". Supersession here is cosine+time only (no spaCy NER -> entity_gate=False).
from __future__ import annotations
import argparse, glob, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from mirror_store_swap import Cand, mgmt_off, mmr_dedup, mgmt_on, foil_abstractive
from metrics import subem
from stats import paired_bootstrap, significant

BGE = os.environ.get("MIRROR_BGE", "BAAI/bge-m3")
WORDS = 512
CHATTIME = re.compile(r"Chat Time:\s*(\d{4})/(\d{2})/(\d{2})\s*\([^)]*\)\s*(\d{2}):(\d{2})")


def parse_sessions(ctx):
    """Segment the LME context into (timestamp_int, text) by 'Chat Time:' markers."""
    ms = list(CHATTIME.finditer(ctx))
    out = []
    for i, m in enumerate(ms):
        ts = int("".join(m.group(k) for k in range(1, 6)))  # YYYYMMDDHHMM
        s, e = m.end(), (ms[i + 1].start() if i + 1 < len(ms) else len(ctx))
        out.append((ts, ctx[s:e]))
    return out


def chunks_with_ts(sessions, w=WORDS):
    chunks, ts = [], []
    for t, text in sessions:
        toks = text.split()
        for i in range(0, len(toks), w):
            chunks.append(" ".join(toks[i:i + w])); ts.append(t)
    return chunks, ts


def clean_q(q):
    q = str(q)
    return q.split("Now Answer the Question:")[-1].strip()


def golds_of(a):
    xs = list(a) if hasattr(a, "__len__") and not isinstance(a, str) else [a]
    return sorted({str(x) for x in xs})


def recall(assembled, golds):
    return subem(" ".join(c.text for c in assembled), golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, default=17)
    ap.add_argument("--nq", type=int, default=60)
    ap.add_argument("--topN", type=int, default=20)
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--sim_super", type=float, default=0.65)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    df = pd.read_parquet(glob.glob("data/**/Accurate_Retrieval*.parquet", recursive=True)[0])
    ctx = df["context"].iloc[a.row]
    qs = list(df["questions"].iloc[a.row])
    ans = list(df["answers"].iloc[a.row])
    _qt = df["metadata"].iloc[a.row].get("question_types")
    qtypes = list(_qt) if _qt is not None else ["?"] * len(qs)

    sess = parse_sessions(ctx)
    ch, cts = chunks_with_ts(sess)
    print(f"row {a.row}: {len(sess)} sessions -> {len(ch)} chunks | {len(qs)} questions", flush=True)
    print(f"question_types: {dict((t, qtypes.count(t)) for t in set(qtypes))}", flush=True)

    model = SentenceTransformer(BGE, device=__import__("os").environ.get("BGE_DEVICE", "mps"))
    print("embedding chunks…", flush=True)
    cvecs = np.asarray(model.encode(ch, batch_size=64, normalize_embeddings=True,
                                    show_progress_bar=False), dtype=np.float32)

    rng = random.Random(a.seed)
    qi_all = list(range(len(qs))); rng.shuffle(qi_all); qi_all = qi_all[:a.nq]
    qtext = [clean_q(qs[i]) for i in qi_all]
    qvecs = np.asarray(model.encode(qtext, normalize_embeddings=True), dtype=np.float32)

    B = a.budget * WORDS
    rows = []  # (qtype, off, mmr, on, foil)
    for k, qi in enumerate(qi_all):
        sims = cvecs @ qvecs[k]
        order = np.argsort(-sims)[:a.topN]
        pool = [Cand(f"c{ci}", ch[ci], tuple(float(x) for x in cvecs[ci]),
                     frozenset(), t=int(cts[ci]), rank=r) for r, ci in enumerate(order)]
        g = golds_of(ans[qi])
        off = recall(mgmt_off(pool, B), g)
        mmr = recall(mmr_dedup(pool, B), g)
        on = recall(mgmt_on(pool, B, sim_super=a.sim_super, super_entity_gate=False), g)
        foil = recall(foil_abstractive(pool, B), g)
        rows.append((qtypes[qi], off, mmr, on, foil))

    def report(label, rs):
        if not rs:
            print(f"  [{label}] n=0"); return
        off = [r[1] for r in rs]; mmr = [r[2] for r in rs]; on = [r[3] for r in rs]; foil = [r[4] for r in rs]
        dg = [mmr[i] - off[i] for i in range(len(rs))]
        ds = [on[i] - mmr[i] for i in range(len(rs))]
        dt = [on[i] - off[i] for i in range(len(rs))]
        cg, cs, ct = paired_bootstrap(dg, 5000, 1), paired_bootstrap(ds, 5000, 1), paired_bootstrap(dt, 5000, 1)
        M = lambda x: sum(x) / len(x)
        print(f"  [{label:18s}] n={len(rs):3d}  OFF={M(off):.3f} MMR={M(mmr):.3f} ON={M(on):.3f} FOIL={M(foil):.3f}")
        print(f"       Δ_generic ={cg[0]:+.3f}[{cg[1]:+.3f},{cg[2]:+.3f}]{' *' if significant(cg) else ''}"
              f"   Δ_semantic={cs[0]:+.3f}[{cs[1]:+.3f},{cs[2]:+.3f}]{' *' if significant(cs) else ''}"
              f"   Δ_total={ct[0]:+.3f}[{ct[1]:+.3f},{ct[2]:+.3f}]{' *' if significant(ct) else ''}")

    print("\n=== REAL-DATA LME temporal (Track B, no LLM) | sim_super=%.2f ===" % a.sim_super, flush=True)
    report("ALL", rows)
    for t in sorted(set(r[0] for r in rows)):
        report(t, [r for r in rows if r[0] == t])
    print("* = 95% bootstrap CI excludes 0. Δ_semantic on knowledge-update/temporal is the "
          "pre-registered headline; kill cell = Δ_semantic ≤ 0 beyond CI there.", flush=True)


if __name__ == "__main__":
    main()
