"""ONE OWNER for how a cell is scored. Every tool must call this and nothing else.

WHY THIS EXISTS. The rule "Recsys is recall@5, InfBench-Sum is LLM-judge f1, LongMemEval is a
GPT-4o judge, everything else is substring match" was established, implemented in tools_grid.py,
and then RE-BROKEN the same night in four freshly-written scripts (audit_overall, board_composed,
board_gated, fast_table) -- each read the raw arm|view key and got 0.000 for Recsys, a number that
would have carried straight into TTL. A rule that lives inside one tool does not protect the next
tool. So the mapping lives here, the tools import it, and `score()` REFUSES rather than returning a
plausible wrong number.

    cell   metric                     where it comes from
    lme    GPT-4o judge               results/<tag>_judged.jsonl  (judge_lme.py)
    summ   LLM-as-judge f1            results/<tag>_judged.jsonl  (judge_infbench.py)
    rec    recsys_recall@5            mab_scorers.score_file over the stored predictions
    other  substring_exact_match      the recorded arm|view key
"""
import json, os, importlib.util, hashlib

ROOT = os.environ.get("MUSTER_HOME", os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
NQ = {"lme": 300, "rec": 200, "det": 71}
JUDGED = {"lme", "summ"}                       # scored only from a *_judged.jsonl
BENCH = {"rec": "recsys_redial_full"}          # rescored from predictions by the benchmark module
ICL = ["b77", "trcC", "clin", "nlu", "trcF"]
ALL = ["shqa", "mhqa", "lme", "evq"] + ICL + ["rec", "summ", "det", "fcsh", "fcmh"] + \
      ["fcmh6k", "fcmh32k", "fcmh64k", "fcsh6k", "fcsh32k", "fcsh64k"]
_ms = None


class Refused(Exception):
    pass


def vkey(arm, view):
    return ("router|" + arm) if view == "router" else (arm + "|" + view)


def _read(p):
    if not os.path.exists(p):
        return None
    out = []
    for i, l in enumerate(open(p), 1):
        if not l.strip():
            continue
        try:
            out.append(json.loads(l))
        except Exception as e:
            raise Refused("truncated record in %s line %d (%s)" % (p, i, e))
    return out or None




# ---- IDENTITY: the file's own recorded row decides what it is, never the tag ------------------
TTL_ROW = {0: "rec", 1: "b77", 2: "clin", 3: "nlu", 4: "trcC", 5: "trcF"}   # from the parquet's
# metadata["qa_pair_ids"], corroborated by class counts (row 4 has SIX classes = trec_coarse) and
# by each name's shot count matching its blank-line record count exactly.
AR_ROW = {0: "shqa", 1: "mhqa", 2: "evq", 17: "lme", 18: "lme", 19: "lme", 20: "lme", 21: "lme"}
FC_ROW = {3: "fcmh", 7: "fcsh", 0: "fcmh6k", 1: "fcmh32k", 2: "fcmh64k",
          4: "fcsh6k", 5: "fcsh32k", 6: "fcsh64k"}


# THE DATASET SIGNATURE, from the file's own records. The ROW ALONE IS AMBIGUOUS ACROSS PARQUETS:
# FactConsolidation row 3 is FC-MH and Test_Time_Learning row 3 is nlu, so a row-only check passes
# an nlu file tagged fcmh. Every record carries `qt`, the question type the harness wrote.
QT_OF = {"factconsolidation": {3: "fcmh", 7: "fcsh", 0: "fcmh6k", 1: "fcmh32k",
                              2: "fcmh64k", 4: "fcsh6k", 5: "fcsh32k", 6: "fcsh64k"},
         "in_context_learning": {1: "b77", 2: "clin", 3: "nlu", 4: "trcC", 5: "trcF"},
         "ruler_qa": {0: "shqa", 1: "mhqa"},
         "eventqa": "evq", "detective_qa": "det", "recsys_redial": "rec",
         "infbench_sum": "summ"}
LME_QT = {"single-session-assistant", "single-session-user", "single-session-preference",
          "multi-session", "knowledge-update", "temporal-reasoning"}


def cell_of(recs):
    """What this file ACTUALLY is, from its own records. None when it cannot be determined."""
    if not recs:
        return None
    qts = {r.get("qt") for r in recs if r.get("qt")}
    if not qts:
        return None
    if qts & LME_QT:
        return "lme"
    if len(qts) != 1:
        return None
    q = qts.pop()
    m = QT_OF.get(q)
    if m is None:
        return None
    if isinstance(m, str):
        return m
    return m.get(recs[0].get("row"))


def check_identity(cell, tag, recs):
    """Raise if the file is not what it is tagged as. Uses (question type, row), never the row
    alone -- rows collide across parquets."""
    if not recs:
        return
    true = cell_of(recs)
    if true is not None and true != cell:
        raise Refused(
            "IDENTITY MISMATCH: %s is tagged '%s' but its records say '%s' "
            "(qt=%r, row=%s). The tag is a claim; the data is the fact."
            % (tag, cell, true, recs[0].get("qt"), recs[0].get("row")))
    row = recs[0].get("row")
    if row is None:
        return
    for table in (TTL_ROW, AR_ROW, FC_ROW):
        if cell in table.values():
            true = table.get(row)
            if true is None:
                continue
            if true != cell:
                raise Refused(
                    "IDENTITY MISMATCH: %s is tagged '%s' but holds parquet row %s, which is '%s'. "
                    "The tag is a claim; the row is the fact. Fix the queue or rename the file -- "
                    "do not remap it in the caller." % (tag, cell, row, true))
            return


_BENCH_CACHE = os.path.join(ROOT, ".bench_cache.json")
_VEC_CACHE = os.path.join(ROOT, ".vec_cache.json")
_SCORER_SIG = [None]


def _scorer_sig():
    """What is doing the scoring, not just what is being scored.

    A cache keyed on the result file alone survives a change to the scorer, and the entry it keeps
    is then a number the current code would never produce. mab_scorers.py has already been repaired
    once in a way that moved a cell from 0.000 to 0.900 (_fix_gold), so this is not hypothetical.
    Digest our wrapper and the benchmark's own module underneath it.
    """
    if _SCORER_SIG[0] is None:
        h = hashlib.sha1()
        for rel in ("mab_scorers.py", os.path.join("mab_repo", "utils", "eval_other_utils.py")):
            p = os.path.join(ROOT, rel)
            h.update(rel.encode())
            h.update(open(p, "rb").read() if os.path.exists(p) else b"<absent>")
        _SCORER_SIG[0] = h.hexdigest()[:12]
    return _SCORER_SIG[0]


def _key(tag, ds, arm, view):
    """The one key both caches use. None when the file is unreadable -- then nothing is cached."""
    try:
        stt = os.stat(os.path.join(R, tag + ".jsonl"))
    except OSError:
        return None
    return "%s|%s|%s|%s|%d|%d|%s" % (tag, ds, arm, view, stt.st_size,
                                     int(stt.st_mtime), _scorer_sig())


def _jload(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}


def _jsave(path, c):
    tmp = path + ".tmp"
    json.dump(c, open(tmp, "w"))
    os.replace(tmp, path)                      # atomic: a crash never leaves a half-written cache


def _cache_load():
    return _jload(_BENCH_CACHE)


def _bench(tag, ds, arm, view):
    """Cached. The key carries the result file's size and mtime, so re-running the leg
    invalidates the entry and nothing else can go stale."""
    key = _key(tag, ds, arm, view)
    if key:
        c = _jload(_BENCH_CACHE)
        if key in c:
            m, n = c[key]
            return m, n, None
    out = _bench_raw(tag, ds, arm, view)
    if key:
        c = _jload(_BENCH_CACHE)
        c[key] = [out[0], out[1]]
        _jsave(_BENCH_CACHE, c)
    return out


def _bench_vec(tag, ds, arm, view):
    """Same key, same discipline, the vector instead of the mean. Paired statistics live here."""
    key = _key(tag, ds, arm, view)
    if key:
        c = _jload(_VEC_CACHE)
        if key in c:
            m, n, v = c[key]
            return m, n, v
    out = _bench_raw(tag, ds, arm, view)
    if key:
        c = _jload(_VEC_CACHE)
        c[key] = [out[0], out[1], out[2]]
        _jsave(_VEC_CACHE, c)
    return out


def _bench_raw(tag, ds, arm, view):
    global _ms
    if _ms is None:
        sp = importlib.util.spec_from_file_location("ms", os.path.join(ROOT, "mab_scorers.py"))
        _ms = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(_ms)
    return _ms.score_file(os.path.join(R, tag + ".jsonl"), ds, arm, view)


def score(cell, tag, arm, view, require_full=True):
    """(mean, n) for one cell / file / arm / view -- the ONLY admissible path.

    Raises Refused rather than returning a wrong-metric number."""
    if cell not in ALL:
        raise Refused("unknown cell %r" % cell)
    want = NQ.get(cell, 100)
    if cell in JUDGED:
        rs = _read(os.path.join(R, tag + "_judged.jsonl"))
        if rs is None:
            return None, 0                      # not judged yet -> incomplete, never guessed
        if cell == "summ":
            # The judged file names its own (arm, view). Trust THAT, not the request: a caller
            # asking for flrs used to be handed a flr file's numbers silently, because this gate
            # only checked the request. b4_summ_{4o,41m,5m} are judged on flr_4_3_3.
            got = {(r.get("arm"), r.get("view")) for r in rs}
            if got != {(arm, view)}:
                raise Refused("summ judged file %s holds %s, not the requested %r"
                              % (tag, sorted(got), (arm, view)))
            x = [r["f1"] for r in rs if isinstance(r.get("f1"), (int, float))]
        else:
            k = vkey(arm, view)
            x = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
    elif cell in BENCH:
        if view == "router":
            m, n = _bench_router(tag, BENCH[cell], arm)
            if m is None or (require_full and n < want):
                return None, n
            return m, n
        try:
            m, n, _ = _bench(tag, BENCH[cell], arm, view)
        except Exception as e:
            raise Refused("benchmark scorer failed on %s: %s" % (tag, e))
        if require_full and n < want:
            return None, n
        return m, n
    else:
        rs = _read(os.path.join(R, tag + ".jsonl"))
        if rs is None:
            return None, 0
        check_identity(cell, tag, rs)
        k = vkey(arm, view)
        x = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
    if not x:
        return None, 0
    if require_full and len(x) < want:
        return None, len(x)
    return sum(x) / len(x), len(x)


def selftest():
    """Fails loudly if any cell would be scored by the wrong path."""
    bad = []
    for c in ALL:
        try:
            v, n = score(c, "gr_%s_m" % c, "flrs_4_3_3", "prune_annot")
        except Refused as e:
            bad.append("%s: %s" % (c, e)); continue
        if c in BENCH and v is not None and v == 0.0:
            bad.append("%s scored exactly 0.000 -- the wrong-metric signature" % c)
    return bad


if __name__ == "__main__":
    print("cells:", " ".join(ALL))
    print("judged:", " ".join(sorted(JUDGED)), " bench:", BENCH)
    for c in ALL:
        try:
            v, n = score(c, "gr_%s_m" % c, "flrs_4_3_3", "prune_annot")
            print("  %-6s %-8s n=%-4s %s" % (c, ("%.4f" % v) if v is not None else "-", n,
                                             "JUDGE" if c in JUDGED else ("BENCH" if c in BENCH else "substring")))
        except Refused as e:
            print("  %-6s REFUSED: %s" % (c, e))
    b = selftest()
    print("SELFTEST:", "clean" if not b else b)


def vector(cell, tag, arm, view):
    """Per-question vector for one cell/file/arm/view -- the paired-statistics counterpart of
    score(). Same metric map, same identity check, same refusals."""
    if cell in JUDGED:
        rs = _read(os.path.join(R, tag + "_judged.jsonl"))
        if rs is None:
            return None
        if cell == "summ":
            got = {(r.get("arm"), r.get("view")) for r in rs}
            if got != {(arm, view)}:
                raise Refused("summ judged file %s holds %s, not the requested %r"
                              % (tag, sorted(got), (arm, view)))
            return [r["f1"] for r in rs if isinstance(r.get("f1"), (int, float))] or None
        k = vkey(arm, view)
        return [r[k] for r in rs if isinstance(r.get(k), (int, float))] or None
    if cell in BENCH:
        if view == "router":
            return None
        _m, _n, vec = _bench_vec(tag, BENCH[cell], arm, view)  # cached on file identity + scorer digest
        return vec or None
    rs = _read(os.path.join(R, tag + ".jsonl"))
    if rs is None:
        return None
    check_identity(cell, tag, rs)
    k = vkey(arm, view)
    return [r[k] for r in rs if isinstance(r.get(k), (int, float))] or None


def _bench_router(tag, ds, arm):
    """recall@5 for the ROUTER view, rebuilt from `router_view` + the per-view predictions.

    The harness records the view the router chose for each question but not the router's own
    prediction string, so the prediction is looked up rather than inferred: question i takes
    `<arm>|<router_view[i]>_pred`, which is exactly what the router emitted."""
    global _ms
    if _ms is None:
        sp = importlib.util.spec_from_file_location("ms", os.path.join(ROOT, "mab_scorers.py"))
        _ms = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(_ms)
    rs = _read(os.path.join(R, tag + ".jsonl"))
    if rs is None:
        return None, 0
    key = _ms.COLUMN.get(ds, (ds, "substring_exact_match"))[1]
    v = []
    for r in rs:
        rv = r.get("router_view")
        if not rv:
            continue
        pred = r.get("%s|%s_pred" % (arm, rv))
        if pred is None:
            continue
        v.append(float(_ms.score_one(ds, pred, r["gold"], None)[key]))
    if not v:
        return None, 0
    return sum(v) / len(v), len(v)
