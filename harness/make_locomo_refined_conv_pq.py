"""LoCoMo-Refined parquet with the store built from LoCoMo-Refined's OWN conversations.jsonl. LoCoMo-Refined's 5,882 turns equal the original
locomo10.json turns except 209 texts that lost leading/trailing whitespace (speakers, captions, queries, dates identical). This
builder renders the Refined turns with the paper's store format (make_locomo_pq.py): one blank-line-separated record per turn,
"<dia_id> [<session date_time>] <speaker>: <text>" and, when the turn has an image, " [image: <blip_caption>]" (no URL, no image
search query). Questions/answers/metadata are those of beam/locomo_refined.parquet (make_locomo_refined_pq.py).
usage: python -I make_locomo_refined_conv_pq.py locomo_refined/conversations.jsonl"""
import hashlib, json, os, re, sys
MUSTER_HOME = os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, MUSTER_HOME); os.chdir(MUSTER_HOME)
import pandas as pd
src = pd.read_parquet("beam/locomo_refined.parquet")
C = {c["sample_id"]: c for c in (json.loads(l) for l in open(sys.argv[1]) if l.strip())}
O = {d["sample_id"]: d for d in json.load(open("locomo/locomo10.json"))}
def orig_recs(cid):   # make_locomo_pq.py, verbatim logic: the paper store's records
    c = O[cid]["conversation"]; out = []
    for k in sorted([k for k in c if re.fullmatch(r"session_\d+", k)], key=lambda k: int(k.split("_")[1])):
        for t in c[k]:
            s = "%s [%s] %s: %s" % (t["dia_id"], c[k + "_date_time"], t["speaker"], t["text"])
            if t.get("blip_caption"): s += " [image: %s]" % t["blip_caption"]
            out.append(s)
    return out
ctxs, changed, diff_recs = [], 0, 0
for i in range(len(src)):
    cid = src["metadata"].iloc[i]["conv_id"]; c = C[cid]
    o = orig_recs(cid)
    assert "\n\n".join(o) == src["context"].iloc[i], ("paper store not reproduced", cid)
    recs = []
    for s in sorted(c["sessions"], key=lambda s: int(s["session_index"])):
        for t in sorted(s["messages"], key=lambda t: int(t["message_index"])):
            r = "%s [%s] %s: %s" % (t["dia_id"], s["date_time"], t["speaker"], t["text"])
            if t.get("blip_caption"):
                r += " [image: %s]" % t["blip_caption"]
            recs.append(r)
    assert len(o) == len(recs), (cid, len(o), len(recs))
    for a, b in zip(o, recs):   # every difference from the paper store must be whitespace only
        if a != b:
            diff_recs += 1
            assert re.sub(r"\s+", " ", a).strip() == re.sub(r"\s+", " ", b).strip(), (cid, a[:80], b[:80])
    ctx = "\n\n".join(recs); ctxs.append(ctx)
    changed += ctx != src["context"].iloc[i]
df = src.copy(); df["context"] = ctxs
df.to_parquet("beam/locomo_refined_conv.parquet")
print("rows", len(df), "rows whose context changed", changed, "turn records that differ from the paper store", diff_recs,
      "sha256", hashlib.sha256(open("beam/locomo_refined_conv.parquet", "rb").read()).hexdigest()[:16])
