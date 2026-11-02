"""Build a reader-swap bundle: one JSON line per question with the byte-identical rendered context the reader saw.

  python make_bundle.py <tag> <parquet> <out.jsonl>        e.g. xb10_beam100k_noxc beam/beam100k.parquet bundle_beam100k_noxc.jsonl

Reads results/<tag>_s*.jsonl (written with ROUTER_ONLY=1 DUMP_CTX=1), joins the parquet metadata (question, gold, rubric,
category, cat_index, conv_id) and writes: row, qi, conv_id, category, cat_index, question, gold, rubric, ctxhash, context,
n_units, tok. Refuses a partial export. Prints the sha256 of the bundle for BEAM_PINS.json.
"""
import sys, json, glob, hashlib, os
import pandas as pd
os.chdir(os.environ.get("MUSTER_HOME", os.path.dirname(os.path.abspath(__file__))))
RES = os.environ.get("MUSTER_RESULTS", "results")   # release: directory holding results/<tag>_s*.jsonl
tag, parq, out = sys.argv[1], sys.argv[2], sys.argv[3]
ARM = "flr_4_3_3|prune_annot"
df = pd.read_parquet(parq)
recs = {}
for f in sorted(glob.glob(os.path.join(RES, f"{tag}_s*.jsonl")) or glob.glob(os.path.join(RES, f"{tag}_r*.jsonl"))):
    for l in open(f):
        r = json.loads(l); recs[(r["row"], r["qi"])] = r
nq = sum(len(df.iloc[i]["questions"]) for i in range(len(df)))
assert len(recs) == nq, f"partial export: {len(recs)}/{nq}"
n = 0
with open(out, "w") as fo:
    for row in range(len(df)):
        m = dict(df.iloc[row]["metadata"]); qs = list(df.iloc[row]["questions"]); ans = df.iloc[row]["answers"]
        for qi in range(len(qs)):
            r = recs[(row, qi)]
            ctx = r["ctx|" + ARM]   # raw-key-ok: context text, not a score
            assert hashlib.sha1(ctx.encode()).hexdigest()[:12] == r["ctxhash|" + ARM]
            a = ans[qi]; gold = [str(x) for x in a] if hasattr(a, "__len__") and not isinstance(a, str) else [str(a)]
            fo.write(json.dumps(dict(row=row, qi=qi, conv_id=str(m["conv_id"]), category=str(m["question_types"][qi]),
                                     cat_index=int(m["cat_index"][qi]), question=str(qs[qi]), gold=gold,
                                     rubric=[str(x) for x in m["rubric"][qi]], ctxhash=r["ctxhash|" + ARM],
                                     n_units=r.get("n_units"), tok=(r.get("tok_by_arm") or {}).get(ARM),
                                     context=ctx), ensure_ascii=False) + "\n"); n += 1
h = hashlib.sha256(open(out, "rb").read()).hexdigest()
print(f"{out}: {n} questions, {len(df)} conversations, {os.path.getsize(out)/1e6:.1f} MB, sha256={h}")
