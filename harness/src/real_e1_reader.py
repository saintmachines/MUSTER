#!/usr/bin/env python3
# real_e1_reader.py — TRACK A: official SubEM with a FIXED Gemini 2.5 Flash reader.
# Tests the hypothesis answer-recall CANNOT: does semantic management (supersession)
# help the READER by removing stale distractors, even when raw recall is equal?
# Focus = knowledge-update questions (rows 17-21), arms OFF/MMR/ON/FOIL.
# Requires GEMINI_API_KEY in env.
from __future__ import annotations
import argparse, glob, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from google import genai
from mirror_store_swap import Cand, mgmt_off, mmr_dedup, mgmt_on, foil_abstractive
from real_e1_lme import parse_sessions, chunks_with_ts, clean_q, golds_of, WORDS, BGE
from metrics import subem
from stats import paired_bootstrap, significant

MODEL = "gemini-2.5-flash"
# Per-request timeout via the SDK (NOT a thread wrapper — the genai client is not thread-safe,
# a ThreadPoolExecutor wrapper hangs). 90s HTTP timeout so a stalled call errors + retries.
# Built on first use, not on import. Importing this module used to construct the client, and because
# real_e1_conflict_killer imports it for the entity helpers, merely importing the graph code demanded a
# paid API key - every zero-cost diagnostic needed one, and the retriever could not run inside a harness
# that has no Gemini credentials at all. Nothing about the request changes; only when the client is made.
_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from google.genai import types as _gt
            _client = genai.Client(http_options=_gt.HttpOptions(timeout=90_000))
        except Exception:
            _client = genai.Client()
    return _client


def ask(ctx, q):
    prompt = ("Answer the question using ONLY the context below. Reply with the shortest exact "
              "answer (a name, number, or phrase), nothing else.\n\nContext:\n" + ctx +
              "\n\nQuestion: " + q + "\nAnswer:")
    for att in range(6):
        try:
            r = _get_client().models.generate_content(model=MODEL, contents=prompt)
            return (r.text or "").strip()
        except Exception:
            if att == 5:
                return "__ERR__"
            time.sleep(2 * (att + 1))


def assemble(pool, B, arm):
    fn = {"OFF": mgmt_off, "MMR": mmr_dedup, "FOIL": foil_abstractive}.get(arm)
    if fn:
        got = fn(pool, B)
    else:  # ON
        got = mgmt_on(pool, B, sim_super=0.80, super_entity_gate=False)
    return " ".join(c.text for c in got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="17,18,19,20,21")
    ap.add_argument("--qtype", default="knowledge-update")
    ap.add_argument("--topN", type=int, default=20)
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--maxq", type=int, default=45)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rows = [int(x) for x in a.rows.split(",")]
    arms = ["OFF", "MMR", "ON", "FOIL"]

    df = pd.read_parquet(glob.glob("data/**/Accurate_Retrieval*.parquet", recursive=True)[0])
    model = SentenceTransformer(BGE, device=__import__("os").environ.get("BGE_DEVICE", "mps"))
    B = a.budget * WORDS
    rng = random.Random(a.seed)
    per = {arm: [] for arm in arms}
    n = 0
    print(f"reader={MODEL} | qtype={a.qtype} | rows={rows}", flush=True)
    for row in rows:
        ctx = df["context"].iloc[row]
        qs = list(df["questions"].iloc[row]); ans = list(df["answers"].iloc[row])
        _qt = df["metadata"].iloc[row].get("question_types")
        qtypes = list(_qt) if _qt is not None else ["?"] * len(qs)
        sess = parse_sessions(ctx); ch, cts = chunks_with_ts(sess)
        cvecs = np.asarray(model.encode(ch, batch_size=64, normalize_embeddings=True,
                                        show_progress_bar=False), dtype=np.float32)
        idx = [i for i in range(len(qs)) if qtypes[i] == a.qtype]
        for qi in idx:
            if n >= a.maxq:
                break
            qv = np.asarray(model.encode([clean_q(qs[qi])], normalize_embeddings=True), dtype=np.float32)[0]
            order = np.argsort(-(cvecs @ qv))[:a.topN]
            pool = [Cand(f"c{ci}", ch[ci], tuple(float(x) for x in cvecs[ci]),
                         frozenset(), t=int(cts[ci]), rank=r) for r, ci in enumerate(order)]
            g = golds_of(ans[qi])
            for arm in arms:
                per[arm].append(subem(ask(assemble(pool, B, arm), clean_q(qs[qi])), g))
            n += 1
            if n % 10 == 0:
                print(f"  {n} Q done", flush=True)
        if n >= a.maxq:
            break

    M = lambda x: sum(x) / len(x) if x else 0.0
    dg = [per["MMR"][i] - per["OFF"][i] for i in range(n)]
    ds = [per["ON"][i] - per["MMR"][i] for i in range(n)]
    dt = [per["ON"][i] - per["OFF"][i] for i in range(n)]
    cg, cs, ct = paired_bootstrap(dg, 5000, 1), paired_bootstrap(ds, 5000, 1), paired_bootstrap(dt, 5000, 1)
    print(f"\n=== TRACK A: Gemini SubEM on {a.qtype} (n={n}) ===", flush=True)
    print("  " + "  ".join(f"{arm}={M(per[arm]):.3f}" for arm in arms), flush=True)
    print(f"  Δ_generic ={cg[0]:+.3f}[{cg[1]:+.3f},{cg[2]:+.3f}]{' *' if significant(cg) else ''}", flush=True)
    print(f"  Δ_semantic={cs[0]:+.3f}[{cs[1]:+.3f},{cs[2]:+.3f}]{' *' if significant(cs) else ''}"
          "   <- reader-distractor test (the metric answer-recall could not see)", flush=True)
    print(f"  Δ_total   ={ct[0]:+.3f}[{ct[1]:+.3f},{ct[2]:+.3f}]{' *' if significant(ct) else ''}", flush=True)
    print(f"  FOIL(lossy) SubEM={M(per['FOIL']):.3f} vs ON={M(per['ON']):.3f} (non-monotonicity)", flush=True)


if __name__ == "__main__":
    main()
