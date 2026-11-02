"""LoCoMo-Refined -> the harness parquet. The store is the paper's: every row's `context` is copied byte for byte from beam/locomo.parquet (built by
make_locomo_pq.py from locomo10.json: "<dia_id> [<date>] <speaker>: <text>" plus " [image: <blip_caption>]"; no image search query).
Only the questions change: LoCoMo-Refined's public questions.jsonl (github.com/mem-eval-suite/LoCoMo_refined, commit 887091190789), in qa_index order per conversation, with
all acceptable golds as the answers and rubric. Asserts: same 10 conversations in the same row order; Refined evidence dia_ids (entries holding two ids are split) occur in
that row's context, at most 1% missing (LoCoMo's own evidence has a few slips), and the missing ones are printed.
usage: python -I make_locomo_refined_pq.py locomo_refined/questions.jsonl"""
import hashlib, json, os, sys
MUSTER_HOME = os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, MUSTER_HOME); os.chdir(MUSTER_HOME)
import pandas as pd
NAME = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop"}
src = pd.read_parquet("beam/locomo.parquet")
Q = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = {}
for q in Q:
    by.setdefault(q["sample_id"], []).append(q)
rows = []; CHECK = [0]; MISS = []
for i in range(len(src)):
    m = src["metadata"].iloc[i]; cid = m["conv_id"]; ctx = src["context"].iloc[i]
    qs = sorted(by[cid], key=lambda q: int(q["qa_index"]))
    for q in qs:
        for e in q.get("evidence") or []:
            for one in [x.strip() for x in str(e).replace(",", ";").split(";") if x.strip()]:   # some entries hold two ids ("D8:6; D9:17")
                CHECK[0] += 1
                if (one + " [") not in ctx:
                    MISS.append((cid, q["qa_id"], one))
    rows.append(dict(context=ctx,
                     questions=[q["question"] for q in qs],
                     answers=[[str(a) for a in q["answer"]] for q in qs],
                     metadata=dict(conv_id=cid, source="beam_locomo",
                                   cat_index=[int(q["category"]) for q in qs],
                                   question_types=[NAME[int(q["category"])] for q in qs],
                                   rubric=[[str(a) for a in q["answer"]] for q in qs],
                                   evidence=[json.dumps(q.get("evidence", [])) for q in qs],
                                   qa_id=[q["qa_id"] for q in qs])))
assert [r["metadata"]["conv_id"] for r in rows] == list(by) or set(by) == {r["metadata"]["conv_id"] for r in rows}
df = pd.DataFrame(rows); df.to_parquet("beam/locomo_refined.parquet")
assert all(df["context"].iloc[i] == src["context"].iloc[i] for i in range(len(df)))
print("evidence ids checked", CHECK[0], "not found in the context", len(MISS), MISS[:8])
assert len(MISS) <= 0.01 * CHECK[0], "too many evidence ids missing from the contexts"
print("rows", len(df), "questions", sum(len(q) for q in df["questions"]), "per row", [len(q) for q in df["questions"]],
      "sha256", hashlib.sha256(open("beam/locomo_refined.parquet", "rb").read()).hexdigest()[:16])
