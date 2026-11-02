#!/usr/bin/env python3
# real_e1_conflict_killer.py — the "killer table" arms on MemoryAgentBench factconsolidation.
# Adds the resolver-STRATEGY arms that stand in for the 2026 frontier (no dependency on their unrefereed
# code), so one table defends SCM / StateFuse / confound at our fixed reader:
#   * ours (prune-to-newest)  = graph_mh_super  (reader sees ONLY the newest per slot)
#   * rank_recency (SCM stand-in) = keep ALL collected facts but ORDER newest-first (recency ranking,
#     NO pruning) -> tests "prune > rank": does removing the stale version beat merely ranking it lower?
#   * preserve_all (StateFuse-conservative stand-in) = keep ALL, cosine order (== graph_mh) but we also
#     report it explicitly as the "preserve, do not collapse" operating point.
#   * keep_oldest / random_match = size-matched confound controls (M1).
# NOTE: StateFuse "Collapsed" at the READER is identical to ours (reader sees newest-only); the only
# difference is store-level destruction, which does not affect SubEM -> we state this, we do not add a
# redundant arm. Per-item dump for offline cluster bootstrap (reanalyze_bundle.py).
from __future__ import annotations
import argparse, glob, json, os, re, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from real_e1_reader import ask
from real_e1_conflict import parse_facts, golds_of
from real_e1_lme import BGE
from metrics import subem
from stats import paired_bootstrap, significant

_CAP = re.compile(r"\b([A-Z][\w.'-]+(?:\s+(?:of|the|and|de|van|von|di)?\s*[A-Z][\w.'-]+)*)\b")
def entities(text): return {m.strip() for m in _CAP.findall(text or "") if len(m.strip()) > 2}

def build_entity_index(fact_ents, n_facts, hub_frac=0.05):
    """Env HUB selects the hub rule: rel (ships) | abs | off.

    The relative rule is wrong in both directions, measured: at N=641 its cut of 32 postings removes
    load-bearing conversational entities and caps reach coverage at 0.513 where an absolute cut gives
    0.744; at N=18332 its cut of 916 postings removes 0.0% of the vocabulary - no proper noun appears
    in 917 fact sentences - so the device is silently off in the largest cells. It stays the default
    ONLY until the C3 measurement decides; the expansion-degree cap subsumes its intent (a hub is by
    definition low-IDF, so a top-k-by-IDF expansion never selects one) with strictly better failure
    semantics: this filter deletes an entity for every query at index time, the cap merely declines to
    expand through it while direct matches survive.
    """
    import os
    ent_facts = {}
    for i, es in enumerate(fact_ents):
        for e in es: ent_facts.setdefault(e, set()).add(i)
    mode = os.environ.get("HUB", "rel")
    if mode == "off":
        hubs = set()
    elif mode == "abs":
        cut = int(os.environ.get("HUB_ABS", "200"))
        hubs = {e for e, fs in ent_facts.items() if len(fs) > cut}
    else:
        hub_cut = max(3, int(hub_frac * n_facts))
        hubs = {e for e, fs in ent_facts.items() if len(fs) > hub_cut}
    return {e: fs for e, fs in ent_facts.items() if e not in hubs}, hubs

def multihop(qents, fact_ents, ent_facts, hubs, H, cap):
    collected, frontier = set(), set()
    for e in qents:
        if e in ent_facts: frontier |= ent_facts[e]
    collected |= frontier
    for _ in range(H):
        nxt = set()
        for fi in frontier:
            for e in (fact_ents[fi] - hubs): nxt |= ent_facts.get(e, set())
        nxt -= collected; collected |= nxt; frontier = nxt
        if len(collected) >= cap or not frontier: break
    return collected

def cell_of(df, row):
    try: return re.sub(r"_no\d+$", "", str(list(df["metadata"].iloc[row]["qa_pair_ids"])[0]))
    except Exception: return f"row{row}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq_per_row", type=int, default=100)
    ap.add_argument("--rows", default="all")
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--cap", type=int, default=30)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dump", default="results/killer_items.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.dump), exist_ok=True); dump = open(a.dump, "w")
    df = pd.read_parquet(glob.glob("data_all/**/Conflict_Resolution*.parquet", recursive=True)[0])
    rows = range(len(df)) if a.rows == "all" else [int(x) for x in a.rows.split(",")]
    rng = random.Random(a.seed); rrng = random.Random(a.seed + 999)
    model = SentenceTransformer(BGE, device=__import__("os").environ.get("BGE_DEVICE", "mps"))
    arms = ["vecRAG", "BM25", "graph_mh", "graph_mh_super", "rank_recency", "keep_oldest", "random_match"]
    per = {m: [] for m in arms}; nfed = {m: [] for m in arms}; cell_scores = {}; n = 0
    print(f"KILLER: prune-vs-rank + M1 controls | H={a.H} cap={a.cap} topk={a.topk} nq/row={a.nq_per_row}", flush=True)
    for row in rows:
        cell = cell_of(df, row); cell_scores.setdefault(cell, {m: [] for m in arms})
        facts = sorted(parse_facts(str(df["context"].iloc[row])))
        ftext = [t for _, t, _ in facts]; forder = [o for o, _, _ in facts]; fslot = [s for _, _, s in facts]
        fact_ents = [entities(t) for t in ftext]; ent_facts, hubs = build_entity_index(fact_ents, len(ftext))
        fvecs = np.asarray(model.encode(ftext, batch_size=128, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        bm25 = BM25Okapi([t.lower().split() for t in ftext])
        qs = list(df["questions"].iloc[row]); ans = list(df["answers"].iloc[row])
        idx = list(range(len(qs))); rng.shuffle(idx); idx = idx[:a.nq_per_row]
        print(f"row {row} [{cell}]: {len(ftext)} facts -> {len(idx)} Q", flush=True)
        for qi in idx:
            q = str(qs[qi]); g = golds_of(ans[qi]); rec = {"row": int(row), "cell": cell, "qi": int(qi)}
            qv = np.asarray(model.encode([q], normalize_embeddings=True), dtype=np.float32)[0]
            vtop = sorted(np.argsort(-(fvecs @ qv))[:a.topk])
            btop = sorted(np.argsort(-bm25.get_scores(q.lower().split()))[:a.topk])
            coll_raw = multihop(entities(q), fact_ents, ent_facts, hubs, a.H, a.cap)
            if not coll_raw: coll_raw = set(int(i) for i in vtop)
            coll = sorted(coll_raw, key=lambda i: -(fvecs[i] @ qv))[:a.cap]
            # prune-to-newest (ours)
            newest = {}
            for i in coll:
                if fslot[i] is not None and (fslot[i] not in newest or forder[i] > forder[newest[fslot[i]]]): newest[fslot[i]] = i
            keep_new = [i for i in coll if fslot[i] is None or newest.get(fslot[i]) == i]
            dropped = len(coll) - len(keep_new)
            # keep-oldest
            oldest = {}
            for i in coll:
                if fslot[i] is not None and (fslot[i] not in oldest or forder[i] < forder[oldest[fslot[i]]]): oldest[fslot[i]] = i
            keep_old = [i for i in coll if fslot[i] is None or oldest.get(fslot[i]) == i]
            # random size-matched
            if dropped > 0 and dropped <= len(coll):
                drop_set = set(rrng.sample(list(coll), dropped)); rand_keep = [i for i in coll if i not in drop_set]
            else: rand_keep = list(coll)
            # rank_recency (SCM stand-in): keep ALL, order newest-first (recency rank, NO prune)
            rank_rec = sorted(coll, key=lambda i: -forder[i])
            outs = {
                "vecRAG": (vtop, ask("\n".join(ftext[i] for i in vtop), q)),
                "BM25": (btop, ask("\n".join(ftext[i] for i in btop), q)),
                "graph_mh": (coll, ask("\n".join(ftext[i] for i in sorted(coll)), q)),
                "graph_mh_super": (keep_new, ask("\n".join(ftext[i] for i in sorted(keep_new)), q)),
                "rank_recency": (rank_rec, ask("\n".join(ftext[i] for i in rank_rec), q)),  # newest-first order preserved
                "keep_oldest": (keep_old, ask("\n".join(ftext[i] for i in sorted(keep_old)), q)),
                "random_match": (rand_keep, ask("\n".join(ftext[i] for i in sorted(rand_keep)), q)),
            }
            for m in arms:
                sel, out = outs[m]; s = subem(out, g)
                per[m].append(s); nfed[m].append(len(sel)); cell_scores[cell][m].append(s); rec[m] = s; rec[m + "_nfed"] = len(sel)
            dump.write(json.dumps(rec) + "\n"); dump.flush(); n += 1
            if n % 25 == 0: print(f"  {n} Q done", flush=True)
    dump.close()
    M = lambda x: sum(x) / len(x) if x else 0.0
    print(f"\n=== PER-CELL (nq/row={a.nq_per_row}) ===", flush=True)
    print("cell".ljust(26) + "".join(m[:12].rjust(15) for m in arms), flush=True)
    for cell in sorted(cell_scores):
        cs = cell_scores[cell]
        print(f"{cell.ljust(26)}" + "".join(f"{M(cs[m]):.3f}".rjust(15) for m in arms), flush=True)
    print(f"\n=== POOLED (n={n}) ===", flush=True)
    for m in arms: print(f"  {m:16s} = {M(per[m]):.3f}  (facts fed {M(nfed[m]):.1f})", flush=True)
    for base in ["graph_mh", "rank_recency", "keep_oldest", "random_match"]:
        d = [per["graph_mh_super"][i] - per[base][i] for i in range(n)]; c = paired_bootstrap(d, 5000, 1)
        note = {"rank_recency": " (PRUNE vs RANK: SCM stand-in — keep both, recency-ordered)"}.get(base, "")
        print(f"  Δ(prune − {base:13s}) = {c[0]:+.3f}[{c[1]:+.3f},{c[2]:+.3f}]{' *' if significant(c) else ''}{note}", flush=True)
    print(f"\nITEM DUMP -> {a.dump} (cluster bootstrap via reanalyze_bundle.py — same schema minus a few arms)", flush=True)


if __name__ == "__main__":
    main()
