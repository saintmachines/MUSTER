#!/usr/bin/env python3
"""G0d (+ G0c for LoCoMo): rebuild, with library code only, every bundle the paper's MemoryAgentBench board and the
LoCoMo / LoCoMo-Refined U=10 reads consumed, from the published dumps; compare each file's sha256 with the published
oracle and each question's context hash with every board read that recorded one. A bundle with no published oracle
file (the five ICL bundles) is checked through its questions only.

  EXPERIMENT_DIR=<read-only tree> OUT_DIR=<fresh dir> python tests/studio/g0d_render_chain.py
"""
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT]
import pandas as pd

from muster import presets
from muster.compat import bundles
from muster.cut import cut
from muster.plan import Plan

M, W = os.environ["EXPERIMENT_DIR"], os.environ["OUT_DIR"]
os.makedirs(W, exist_ok=True)
CELLS = ["shqa", "mhqa", "lme", "evq", "b77", "clin", "nlu", "trcC", "trcF", "rec", "summ", "det", "fcsh", "fcmh"]
ICL = ("b77", "clin", "nlu", "trcC", "trcF")
READERS = (("4om", "4omini"), ("41m", "41mini"), ("5m", "5mini"))
_DF = {}


def df(p):
    if p not in _DF:
        _DF[p] = pd.read_parquet(os.path.join(M, p))
    return _DF[p]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def board_legs(c):
    out = []
    for m, fc in READERS:
        if c in ("shqa", "mhqa", "lme", "rec"): out.append("gOA_l2k_%s_%s" % (m, c))
        elif c in ("evq", "summ"): out.append("gOA_r4_%s_%s" % (m, c))
        elif c == "det": out += ["gOA_r4_%s_det" % m, "gOA_r4rep_%s_det" % m, "gOA_r4rep2_%s_det" % m]
        elif c in ICL: out.append("nx_ra_%s" % c if m == "4om" else "gOA_nx_%s_%s" % (m, c))
        else: out.append("gOA_%s_%s_struct_notag" % (fc, c))
    return out


def units_fn(parq, atom, chunk, mscale):
    plan = Plan(branch="record" if atom == "blank" else "prose", atom=atom, chunk=chunk, mscale=mscale, U=10,
                policy="flr", render="r4", mode="exact")

    def f(row):
        c = cut(str(df(parq)["context"].iloc[row]), plan)
        return list(c.units), list(c.parent)
    return f


def main():
    t0 = time.time()
    legs = presets.legs()
    files = []          # (name, ours, oracle)
    finals = {}
    for c in CELLS:
        if c.startswith("fc"):
            tag, arm = "mg_%s_struct" % c, "attprio_sess"
            leg = legs[tag]
            p = os.path.join(W, tag + ".jsonl")
            bundles.bundle_mab(os.path.join(M, "results", tag + ".jsonl"), tag=tag, arm=arm, df=df(leg["parq"]),
                               nq=leg["nq"], out=p, dsname="factconsolidation")
            q = os.path.join(W, tag + "_notag.jsonl")
            bundles.notag(p, q)
            files += [(tag, p, os.path.join(M, "bundle_out/mab_glm/%s.jsonl" % tag)),
                      (tag + "_notag", q, os.path.join(M, "bundle_out/mab_glm/%s_notag.jsonl" % tag))]
            finals[c] = q
            continue
        tag = "lz_" + c
        leg = legs[tag]
        env = presets.get(tag).compat["env"]
        p = os.path.join(W, "full_%s.jsonl" % tag)
        bundles.bundle_mab(os.path.join(M, "len0928_in", tag + "_s0.jsonl"), tag=tag, arm=leg["arm"].split("|")[0],
                           df=df(leg["parq"]), nq=leg["nq"], out=p, dsname=leg["ds"])
        files.append(("full_" + tag, p, os.path.join(M, "bundle_out/len0928/full_%s.jsonl" % tag)))
        finals[c] = p
        if c in ("evq", "det", "summ"):
            q = os.path.join(W, "r4_%s.jsonl" % tag)
            bundles.r4_bundle(p, q, units_fn(leg["parq"], env["ATOM"], int(env["CHUNK"]), int(env["MSCALE"])))
            files.append(("r4_" + tag, q, os.path.join(M, "bundle_out/rules/r4_mab_%s.jsonl" % c)))
            finals[c] = q
        print("[%4.0fs] %s built" % (time.time() - t0, c), flush=True)
    for tag, pq in (("lcm_u10", "beam/locomo.parquet"), ("lcmrefc_u10", "beam/locomo_refined_conv.parquet")):
        r0 = os.path.join(W, tag + "_r0.jsonl")
        n, dups = bundles.bundle_beam(bundles.shards(os.path.join(M, "results"), tag), df=df(pq), out=r0)
        r4 = os.path.join(W, tag + "_r4.jsonl")
        bundles.r4_bundle(r0, r4, units_fn(pq, "blank", 512, 2048))
        files += [(tag + "_r0", r0, os.path.join(M, "bundle_out/locomo/%s_r0.jsonl" % tag)),
                  (tag + "_r4", r4, os.path.join(M, "bundle_out/locomo/%s_r4.jsonl" % tag))]
        print("[%4.0fs] %s built (%d questions, %d repeated keys)" % (time.time() - t0, tag, n, len(dups)), flush=True)
    ok_f, no_orc = 0, []
    for name, ours, orc in files:
        if not os.path.exists(orc):
            no_orc.append(name)
            print("%-22s no oracle file, checked by question" % name)
            continue
        same = sha(ours) == sha(orc)
        ok_f += same
        print("%-22s %s" % (name, "MATCH" if same else "DIFF"))
    unchecked = [name for name, ours, orc in files if name in no_orc and ours not in finals.values()]
    tq = tok = tl = lok = 0
    for c in CELLS:
        gh = {(r["row"], r["qi"]): r["ctxhash"] for r in map(json.loads, open(finals[c]))}
        k = "ctxhash|%s|prune_annot" % ("flr_4_3_3" if c in ("evq", "summ", "det") else "attprio_sess" if c.startswith("fc") else "flrs_4_3_3")
        for t in board_legs(c):
            rs = [json.loads(x) for x in open(os.path.join(M, "results", t + ".jsonl")) if x.strip()]
            n_eq = sum(gh.get((r["row"], r["qi"])) == r[k] for r in rs)
            tl += 1; lok += n_eq == len(rs) == len(gh); tq += len(rs); tok += n_eq
    print("files byte-identical %d/%d, %d without an oracle file and checked by question (%s); board reads with every "
          "question equal %d/%d; questions %d/%d; %.0f s" % (ok_f, len(files) - len(no_orc), len(no_orc),
                                                             " ".join(no_orc) or "none", lok, tl, tok, tq, time.time() - t0))
    if unchecked:
        print("no oracle file and no board read: " + " ".join(unchecked))
    sys.exit(0 if (ok_f == len(files) - len(no_orc) and lok == tl and not unchecked) else 1)


if __name__ == "__main__":
    main()
