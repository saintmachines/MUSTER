"""LoCoMo (snap-research/locomo, data/locomo10.json) -> the parquet layout the harness reads (context/questions/answers/metadata),
2026-10-07. Same construction as the BEAM store we ship (beam_ser.py S1): a faithful dump of every field a turn carries, one
blank-line-separated record per turn, the writer's own id first: "<dia_id> [<session date_time>] <speaker>: <text>" and, when the
turn has an image, " [image: <blip_caption>]". Nothing is added that the json does not hold.
Questions: categories 1-4 (category 5 is adversarial and has no gold answer; excluded as in Mem0's eval) -> 1,540.
Category names follow the LoCoMo convention: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop.
source='beam_locomo' so the harness applies its generic (BEAM) path: no LoCoMo-specific code.
usage: python -I make_locomo_pq.py (reads locomo/locomo10.json, writes beam/locomo.parquet)"""
import json, os, re, sys, hashlib, pandas as pd, numpy as np
MUSTER_HOME = os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, MUSTER_HOME); os.chdir(MUSTER_HOME)
NAME = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop"}
D = json.load(open("locomo/locomo10.json"))
rows = []
for d in D:
    c = d["conversation"]
    ks = sorted([k for k in c if re.fullmatch(r"session_\d+", k)], key=lambda k: int(k.split("_")[1]))
    recs = []
    for k in ks:
        dt = c[k + "_date_time"]
        for t in c[k]:
            s = "%s [%s] %s: %s" % (t["dia_id"], dt, t["speaker"], t["text"])
            if t.get("blip_caption"): s += " [image: %s]" % t["blip_caption"]
            recs.append(s)
    qa = [q for q in d["qa"] if q["category"] in NAME]
    rows.append(dict(context="\n\n".join(recs),
                     questions=[q["question"] for q in qa],
                     answers=[[str(q["answer"])] for q in qa],
                     metadata=dict(conv_id=d["sample_id"], source="beam_locomo",
                                   cat_index=[q["category"] for q in qa],
                                   question_types=[NAME[q["category"]] for q in qa],
                                   rubric=[[str(q["answer"])] for q in qa],
                                   evidence=[json.dumps(q.get("evidence", [])) for q in qa])))
df = pd.DataFrame(rows); df.to_parquet("beam/locomo.parquet")
print("rows", len(df), "questions", sum(len(q) for q in df["questions"]), "sha256", hashlib.sha256(open("beam/locomo.parquet","rb").read()).hexdigest()[:16])
print(df["context"].iloc[0][:400])
