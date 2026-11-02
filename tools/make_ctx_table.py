#!/usr/bin/env python3
"""Write gate/ctx_table.jsonl: one line per question of every published context the paper's reads consumed, with
(leg, row, qi, ctxhash, tok) and no text. With it, a run of the library anywhere can be checked question by question.

  EXPERIMENT_DIR=<tree with bundle_out/> python tools/make_ctx_table.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

M = os.environ["EXPERIMENT_DIR"]
SOURCES = {   # leg -> the bundle the paper's reads consumed (relative to the experiment tree)
    "lz_shqa": "bundle_out/len0928/full_lz_shqa.jsonl", "lz_mhqa": "bundle_out/len0928/full_lz_mhqa.jsonl",
    "lz_lme": "bundle_out/len0928/full_lz_lme.jsonl", "lz_rec": "bundle_out/len0928/full_lz_rec.jsonl",
    "lz_evq": "bundle_out/rules/r4_mab_evq.jsonl", "lz_det": "bundle_out/rules/r4_mab_det.jsonl",
    "lz_summ": "bundle_out/rules/r4_mab_summ.jsonl",
    "mg_fcsh_struct": "bundle_out/mab_glm/mg_fcsh_struct_notag.jsonl",
    "mg_fcmh_struct": "bundle_out/mab_glm/mg_fcmh_struct_notag.jsonl",
    "lz_u20_beam100k": "bundle_out/rules/r4_u20_100k.jsonl", "lz_u40_beam100k": "bundle_out/rules/r4_u40_100k.jsonl",
    "lz_u20_beam1m": "bundle_out/rules/r4_u20_1m.jsonl", "lz_u40_beam1m": "bundle_out/rules/r4_u40_1m.jsonl",
    "lz_u20_beam10m": "bundle_out/rules/r4_u20_10m.jsonl", "lz_u40_beam10m": "bundle_out/rules/r4_u40_10m.jsonl",
    "lcm_u10": "bundle_out/locomo/lcm_u10_r4.jsonl", "lcmrefc_u10": "bundle_out/locomo/lcmrefc_u10_r4.jsonl",
    "lcm_u20": "bundle_out/locomo/lcm_u20_r4.jsonl", "lcm_u5": "bundle_out/locomo/lcm_u5_r4.jsonl",
}
ICL = {"lz_b77": "b77", "lz_clin": "clin", "lz_nlu": "nlu", "lz_trcC": "trcC", "lz_trcF": "trcF"}
n = 0
missing = []
with open(os.path.join(ROOT, "gate", "ctx_table.jsonl"), "w") as fo:
    for tag, rel in sorted(SOURCES.items()):
        p = os.path.join(M, rel)
        if not os.path.exists(p):
            missing.append(tag)
            continue
        for line in open(p):
            r = json.loads(line)
            fo.write(json.dumps({"leg": tag, "row": r["row"], "qi": r["qi"], "ctxhash": r["ctxhash"],
                                 "tok": r.get("tok")}) + "\n")
            n += 1
    for tag, c in sorted(ICL.items()):   # read from bundles packed under another label: the read's own hashes
        p = os.path.join(M, "results", "nx_ra_%s.jsonl" % c)
        for line in open(p):
            r = json.loads(line)
            fo.write(json.dumps({"leg": tag, "row": r["row"], "qi": r["qi"],
                                 "ctxhash": r["ctxhash|flrs_4_3_3|prune_annot"], "tok": None}) + "\n")
            n += 1
print("wrote gate/ctx_table.jsonl: %d questions; legs without a bundle on this machine: %s" % (n, missing or "none"))
