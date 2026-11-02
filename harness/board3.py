#!/usr/bin/env python3
"""The three-backbone board. Source is named PER (backbone, cell) and scored through cellscore,
which owns the metric map -- Recsys is recall@5 and LME/Summ are judged, so reading the raw
substring key gives 0.000 or the wrong scale."""
import sys, os
MUSTER_HOME = os.environ.get("MUSTER_HOME", os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, MUSTER_HOME); os.chdir(MUSTER_HOME)
import cellscore as CS

ICL = ["b77", "trcC", "clin", "nlu", "trcF"]
ORDER = ["shqa", "mhqa", "lme", "evq"] + ICL + ["rec", "summ", "det", "fcsh", "fcmh"]
BOARD, GATE = "flrs_4_3_3", "attprio_sess"          # gate fires only on the two FC cells
ARM = {c: (GATE if c in ("fcsh", "fcmh") else BOARD) for c in ORDER}

# ra_* is the rule's 512 refinement (the paper's board); at_* is the 4096 protocol CONTROL.
SRC = {
 "4o-mini":  dict(shqa="at_shqa", mhqa="at_mhqa", lme="p24_lme_m", evq="at_evq",
                  b77="ra_b77", trcC="ra_trcC", clin="ra_clin", nlu="ra_nlu", trcF="ra_trcF",
                  rec="ra_rec", summ="p24_summ_m", det="at_det", fcsh="as6_fcsh", fcmh="as6_fcmh"),
 # PROVENANCE FIX 2026-09-05. shqa/mhqa/det here were gp4p1_rulSH / gp4p1_rulMH / gp4p1_detqa,
 # built by queue_gaps.sh under lme_pf12/13.py with HDETECT=1 and NO MSCALE, so they carried a FLAT
 # parent layer (rulSH 412 units / 412 parents, detqa 25/25) while the 4o-mini row carried the
 # MSCALE hierarchy the rule derives. s6a_* are the same cells under lme_pf27.py with HDETECT=0 and
 # the derived MSCALE, already complete on disk. evq and lme have no conformant leg yet and are
 # still the old sources -- see EXPERIMENT_PLAN_a4.md.
 "4.1-mini": dict(shqa="s6a_shqa", mhqa="s6a_mhqa", lme="rd4p1-mini_lme", evq="gp4p1_evq",
                  b77="m41_b77", trcC="m41_trcC", clin="m41_clin", nlu="m41_nlu", trcF="m41_trcF",
                  rec="m41_rec", summ="m41_summ", det="s6a_det",
                  fcsh="m41_fcsh", fcmh="m41_fcmh"),
 "5-mini":   dict(shqa="s6b_shqa", mhqa="s6b_mhqa", lme="rd5-mini_lme", evq="gp5m_evq",
                  b77="m5m_b77", trcC="m5m_trcC", clin="m5m_clin", nlu="m5m_nlu", trcF="m5m_trcF",
                  rec="m5m_rec", summ="m5m_summ", det="gp5m_detqa",
                  fcsh="m5m_fcsh", fcmh="m5m_fcmh"),
}
BKS = list(SRC)
print("%-7s" % "cell" + "".join("%11s" % b for b in BKS))
print("-" * (7 + 11 * len(BKS)))
S = {b: {} for b in BKS}
for c in ORDER:
    line = "%-7s" % c
    for b in BKS:
        try:
            v, n = CS.score(c, SRC[b][c], ARM[c], "prune_annot")
        except Exception as e:
            v, n = None, 0
        if v is None:
            line += "%11s" % ("(%d)" % n if n else "-")
        else:
            S[b][c] = v; line += "%11.3f" % v
    print(line)
print("-" * (7 + 11 * len(BKS)))


def comp(s):
    have = lambda ks: all(k in s for k in ks)
    out = {}
    if have(["shqa", "mhqa", "lme", "evq"]): out["AR"] = sum(s[k] for k in ["shqa","mhqa","lme","evq"])/4
    if have(ICL): out["MCC"] = sum(s[k] for k in ICL)/5
    if "MCC" in out and "rec" in s: out["TTL"] = (out["MCC"] + s["rec"])/2
    if have(["summ","det"]): out["LRU"] = (s["summ"] + s["det"])/2
    if have(["fcsh","fcmh"]): out["SF"] = (s["fcsh"] + s["fcmh"])/2
    if all(k in out for k in ("AR","TTL","LRU","SF")):
        out["All"] = (out["AR"] + out["TTL"] + out["LRU"] + out["SF"])/4
    return out

for name in ("AR", "MCC", "TTL", "LRU", "SF", "All"):
    line = "%-7s" % name
    for b in BKS:
        d = comp(S[b])
        line += "%11.1f" % (100*d[name]) if name in d else "%11s" % "—"
    print(line)
print()
for b in BKS:
    miss = [c for c in ORDER if c not in S[b]]
    print("%-9s missing: %s" % (b, miss or "none"))
