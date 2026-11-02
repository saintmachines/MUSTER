"""Score LongMemEval the way MemoryAgentBench scores it: a GPT-4o judge, not substring match.

Every LongMemEval number this project has produced used SubEM, and the benchmark does not. Its README is
explicit - Accurate Retrieval is substring_exact_match for event_qa and ruler_qa, Conflict Resolution is
substring_exact_match, but longmemeval is LLM-as-judge - and llm_based_eval/longmem_qa_evaluate.py carries
the protocol: gpt-4o at temperature 0 with max_tokens 10, a per-question-type prompt, and a label that is
true when the reply contains "yes".

The templates are not interchangeable with a substring test, and that is the whole point. knowledge-update
counts an answer correct when it carries the updated value even if older values appear alongside it, which
is exactly the case a prune-versus-composite comparison turns on. temporal-reasoning forgives off-by-one
day counts. single-session-preference is scored against a rubric rather than a string, so SubEM on that
type was measuring almost nothing. The templates are the benchmark's own, read from its file at run time.

Judging is applied to predictions already on disk, so the reader is not re-run. Identical (question,
prediction) pairs are judged once and reused, which matters because sixteen arms times two views produce
many identical short answers.
"""
import ast, json, os, sys, collections, hashlib
import pandas as pd
MUSTER_HOME = os.environ.get("MUSTER_HOME", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MUSTER_HOME)
from budget_guard import Guard

SRC = os.environ.get("SRC", os.path.join(MUSTER_HOME, "results", "LMEO_subem.jsonl"))
OUT = os.environ.get("JOUT", os.path.join(MUSTER_HOME, "results", "LMEO_judged.jsonl"))
# The benchmark's file says model="gpt-4o", which is a moving alias; this pins the version LongMemEval's
# own protocol specifies, so the run is reproducible and still the model the benchmark asks for.
# DEFAULT CHANGED to the alias (2026-08-20). The pin `gpt-4o-2024-08-06` 403s on this project key
# ("does not have access to model"), and NO run on the board ever used it: every judging call in
# every shell script passes JUDGE_MODEL=gpt-4o, and 26 of 27 judge log lines report `gpt-4o`. A
# default that crashes - and that would silently judge new results with a different model from
# the historical band they are compared against if it ever stopped crashing - is the CEML=320
# mistake again. `gpt-4o` is also the string the benchmark's own longmem_qa_evaluate.py names.
JUDGE = os.environ.get("JUDGE_MODEL", "gpt-4o")
PARQ = os.path.expanduser("~/.cache/huggingface/hub/datasets--ai-hyz--MemoryAgentBench/snapshots/"
        "7ea066982b140a19337e17e60d45d4076e042faf/data/Accurate_Retrieval-00000-of-00001.parquet")
ROWS = [17, 18, 19, 20, 21]


_MAB_EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mab_repo", "llm_based_eval", "longmem_qa_evaluate.py")
_ANSCHECK_SHA1 = "82adff5f4d9c0fc8"     # the function every published LongMemEval judgement used
_ANSCHECK = []


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    """The benchmark's own judge prompt (llm_based_eval/longmem_qa_evaluate.py), read from its file at first use: only
    this function is executed, so the module's other imports are not needed, and no copy lives here."""
    if not _ANSCHECK:
        import ast
        if not os.path.exists(_MAB_EVAL):
            raise SystemExit("judge_lme.py needs MemoryAgentBench's llm_based_eval/longmem_qa_evaluate.py: "
                             "run scripts/fetch_mab_repo.sh first")
        src = open(_MAB_EVAL, encoding="utf-8").read()
        fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "get_anscheck_prompt"]
        code = ast.get_source_segment(src, fn[0]) if fn else ""
        if hashlib.sha1(code.encode()).hexdigest()[:16] != _ANSCHECK_SHA1:
            raise SystemExit("get_anscheck_prompt in %s is not the one the published judgements used" % _MAB_EVAL)
        ns = {}
        exec(code, ns)
        _ANSCHECK.append(ns["get_anscheck_prompt"])
    return _ANSCHECK[0](task, question, answer, response, abstention=abstention)


def main():
    df = pd.read_parquet(PARQ)
    # question text and question_id per (row, qi); the id carries the _abs flag the judge needs
    qtext, qid = {}, {}
    for r in ROWS:
        m = df["metadata"].iloc[r]
        qs = list(df["questions"].iloc[r])
        ids = list(m["question_ids"]) if m.get("question_ids") is not None else [""] * len(qs)
        for i, q in enumerate(qs):
            qtext[(r, i)] = str(q)
            qid[(r, i)] = str(ids[i]) if i < len(ids) else ""

    rows = [json.loads(l) for l in open(SRC)]
    arms = sorted({k.rsplit("_pred", 1)[0] for k in rows[0] if k.endswith("_pred")})
    print("judging %d questions x %d predictions with %s" % (len(rows), len(arms), JUDGE), flush=True)

    g = Guard(os.environ.get("JGUARD", "judge_" + os.path.basename(OUT).replace(".jsonl", "")),
              cap_usd=float(os.environ.get("CAP_USD", "15")))
    cache, out = {}, open(OUT, "w")
    hits = collections.defaultdict(float)
    bytype = collections.defaultdict(lambda: collections.defaultdict(float))
    n = 0
    from concurrent.futures import ThreadPoolExecutor
    _work = {}
    for r in rows:
        key = (r["row"], r["qi"])
        q, gold, qt = qtext[key], r["gold"], r["qt"]
        abst = "_abs" in qid.get(key, "")
        for a in arms:
            pred = (r.get(a + "_pred") or "").strip()
            ck = hashlib.sha1(("%s|%s|%s" % (key, qt, pred)).encode()).hexdigest()
            if pred and ck not in _work:
                _work[ck] = get_anscheck_prompt(qt, q, gold, pred, abstention=abst)

    def _judge_one(item):
        ck, prompt = item
        resp = g.openai(prompt, model=JUDGE, max_tokens=10, temperature=0.0,
                        note="judge|par|" + ck[:10])
        return ck, float("yes" in (resp or "").lower())

    with ThreadPoolExecutor(max_workers=8) as _ex:
        for _ck, _v in _ex.map(_judge_one, sorted(_work.items())):
            cache[_ck] = _v
    print("parallel pre-pass done: %d unique judgements  $%.3f" % (len(cache), g.spent), flush=True)
    for r in rows:
        key = (r["row"], r["qi"])
        q, gold, qt = qtext[key], r["gold"], r["qt"]
        abst = "_abs" in qid.get(key, "")
        # stamp the judge into every record: the model is the instrument, and an instrument
        # that is only recorded in a log cannot be checked once the log is gone
        rec = {"row": r["row"], "qi": r["qi"], "qt": qt, "router_view": r["router_view"],
               "judge_model": JUDGE}
        for a in arms:
            pred = (r.get(a + "_pred") or "").strip()
            ck = hashlib.sha1(("%s|%s|%s" % (key, qt, pred)).encode()).hexdigest()
            if ck not in cache:
                if not pred:
                    cache[ck] = 0.0
                else:
                    p = get_anscheck_prompt(qt, q, gold, pred, abstention=abst)
                    resp = g.openai(p, model=JUDGE, max_tokens=10, temperature=0.0,
                                    note="judge|%s|r%dq%d" % (a, r["row"], r["qi"]))
                    cache[ck] = float("yes" in (resp or "").lower())
            rec[a] = cache[ck]
            hits[a] += cache[ck]
            bytype[qt][a] += cache[ck]
        # the router is a lookup over the two views, exactly as in the reader run
        for base in sorted({a.split("|")[0] for a in arms}):
            k = "%s|%s" % (base, r["router_view"])
            if k in rec:
                rec["router|" + base] = rec[k]
                hits["router|" + base] += rec[k]
                bytype[qt]["router|" + base] += rec[k]
        bytype[qt]["n"] += 1
        n += 1
        out.write(json.dumps(rec) + "\n")
        if n % 25 == 0:
            out.flush()
            print("   %3d/%d  $%.3f  (%d unique judgements)" % (n, len(rows), g.spent, len(cache)),
                  flush=True)
    out.close()

    bases = sorted({a.split("|")[0] for a in arms})
    print("\n=== LongMemEval S*, MemoryAgentBench protocol (GPT-4o judge), n=%d, $%.3f ===" % (n, g.spent))
    print("   %-22s %-11s %-11s %-11s" % ("arm", "prune", "composite", "router"))
    for b in sorted(bases, key=lambda b: -max(hits.get(b + "|prune_annot", 0),
                                              hits.get(b + "|composite", 0), hits.get("router|" + b, 0))):
        print("   %-22s %-11.3f %-11.3f %-11.3f"
              % (b, hits.get(b + "|prune_annot", 0) / n, hits.get(b + "|composite", 0) / n,
                 hits.get("router|" + b, 0) / n))
    print("\n   by question type (router column)")
    for t in sorted(bytype):
        c = bytype[t]
        print("   %-26s n=%-4d %s" % (t, int(c["n"]),
              "  ".join("%s %.3f" % (b, c["router|" + b] / c["n"]) for b in bases[:5])))
    print("\n   MemoryAgentBench Table 3, LongMemEval S*, n=300: GPT-4.1-mini(1M) .557 | HippoRAG-v2 .507")
    print("   | text-embed-3-large .503 | text-embed-3-small .483 | BM25 .453 | Qwen3-Emb-4B .433")
    print("   | Zep .383 | MIRIX .373 | Mem0 .360 | GraphRAG .350 | RAPTOR .343 | GPT-4o-mini(128K) .307")
    print("   | Cognee .293 | Self-RAG .257 | Contriever .157")
    print("LME_JUDGE_DONE", flush=True)


if __name__ == "__main__":
    main()
