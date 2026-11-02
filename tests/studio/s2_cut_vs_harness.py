#!/usr/bin/env python3
"""S1 + S2 on the Studio: for every row of every final leg, the library's cut under the leg's preset against
(a) the harness's own load_row under the leg's env, (b) the parent labels and unit count the published dump recorded,
and (c) the reranker-cache file the published run wrote, which is named by a hash of the exact unit list.

  EXPERIMENT_DIR=<read-only experiment tree> CE_DIRS=<dir1:dir2> [LEGS=tag,tag] python tests/studio/s2_cut_vs_harness.py
"""
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]
import pandas as pd

from muster import presets
from muster.compat.ce_cache import ce_sig
from muster.cut import cut
from oracle import harness_cut

M = os.environ["EXPERIMENT_DIR"]
CE_DIRS = [d for d in os.environ.get("CE_DIRS", "").split(":") if d]
ONLY = set(filter(None, os.environ.get("LEGS", "").split(",")))
SUFFIX = {"2048": "2ec7c5cc", "1024": "61fdceea"}
MEMO = {}


def dump_par(tag, row):
    for f in sorted(glob.glob(os.path.join(M, "len0928_in", tag + "_s*.jsonl"))) + \
             sorted(glob.glob(os.path.join(M, "len0928_in", tag + ".jsonl"))):
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("row") == row and "par" in d:
                    return d["par"], d.get("n_rounds"), os.path.basename(f)
    return None, None, None


def main():
    legs = presets.legs()
    fails, lines = 0, []
    t0 = time.time()
    for tag in sorted(legs):
        if ONLY and tag not in ONLY:
            continue
        leg = legs[tag]
        plan = presets.get(leg["preset"])
        env = plan.compat["env"]
        df = pd.read_parquet(os.path.join(M, leg["parq"]))
        settings = dict(ATOM=env.get("ATOM", "sentence"), CHUNK=int(env.get("CHUNK", "4096")),
                        MSCALE=int(env.get("MSCALE", "0") or 0), AUTOSTRUCT=int(env.get("AUTOSTRUCT", "0") or 0),
                        AS_SLOT=env.get("AS_SLOT", "1"))
        for row in leg["rows"]:
            key = (leg["parq"], row, tuple(sorted(settings.items())))
            if key not in MEMO:     # the u20 and u40 legs of one store share its cut
                text = str(df["context"].iloc[row])
                c = cut(text, plan)
                hu, hk, hp, ho = harness_cut(df, row, **settings)
                MEMO[key] = (c, list(c.units) == list(hu) and list(c.keys) == list(hk) and list(c.parent) == list(hp))
            c, ok_h = MEMO[key]
            dpar, dn, dfile = dump_par(tag, row)
            # dumps write parent labels as strings (LongMemEval sessions are ints in memory)
            ok_d = None if dpar is None else ([str(x) for x in c.parent] == [str(x) for x in dpar] and dn == len(c.units))
            sig = ce_sig(list(c.units))
            suf = SUFFIX.get(env.get("CEML", ""), "")
            hit = [d for d in CE_DIRS if os.path.exists(os.path.join(d, "%s__%s.json" % (sig, suf)))] if suf else []
            ok = ok_h and ok_d is not False
            fails += 0 if ok else 1
            lines.append("%s %-16s row %-3d units %6d parents %6d | harness %s | dump %s | cache %s" % (
                "PASS" if ok else "FAIL", tag, row, len(c.units), len(set(c.parent)), "equal" if ok_h else "DIFF",
                {None: "none", True: "equal", False: "DIFF"}[ok_d], ("hit " + sig) if hit else ("miss " + sig)))
            print(lines[-1], flush=True)
    print("rows %d, failed %d, %.0f s" % (len(lines), fails, time.time() - t0))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
