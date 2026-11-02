#!/usr/bin/env python3
"""Score our result files with MemoryAgentBench's OWN scorer. Zero reimplementation.

WHY. Three cells the benchmark reports have never been scored here because their metric is not
substring match: MCC (the five in-context-learning sets), Recom. (Recall@5 over a movie-entity
mapping) and Summ. (token F1 over InfBench summaries). Writing our own version of any of those
invites the objection that we published a metric under the benchmark's name, which is exactly the
mistake already recorded for DetectiveQA. So this module imports mab_repo/utils/eval_other_utils.py
and calls its post_process() - the same function the benchmark's own harness calls.

The metric each column of the paper's Table 3 reports (from its caption):
    SH-QA, MH-QA, LME, EventQA, DetQA, FC-SH, FC-MH   accuracy   -> substring_exact_match
    MCC   (mean of the five ICL sets)                 accuracy   -> substring_exact_match
    Recom.                                            Recall@5   -> recsys_recall@5
    Summ. (InfBench)                                  F1         -> f1
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_EV = None

# sub_dataset string -> (routing key the benchmark dispatcher matches, metric the paper reports)
COLUMN = {
    "ruler_qa1_197K":                   ("ruler_qa1_197K", "substring_exact_match"),
    "ruler_qa2_421K":                   ("ruler_qa2_421K", "substring_exact_match"),
    "eventqa_full":                     ("eventqa_full", "substring_exact_match"),
    "eventqa_65536":                    ("eventqa_65536", "substring_exact_match"),
    "eventqa_131072":                   ("eventqa_131072", "substring_exact_match"),
    "longmemeval_s*":                   ("longmemeval_s*", "substring_exact_match"),
    "detective_qa":                     ("detective_qa", "substring_exact_match"),
    "factconsolidation_mh_262k":        ("factconsolidation_mh_262k", "substring_exact_match"),
    "factconsolidation_sh_262k":        ("factconsolidation_sh_262k", "substring_exact_match"),
    "icl_banking77_5900shot_balance":   ("icl_banking77", "substring_exact_match"),
    "icl_clinic150_7050shot_balance":   ("icl_clinic150", "substring_exact_match"),
    "icl_nlu_8296shot_balance":         ("icl_nlu", "substring_exact_match"),
    "icl_trec_coarse_6600shot_balance": ("icl_trec_coarse", "substring_exact_match"),
    "icl_trec_fine_6400shot_balance":   ("icl_trec_fine", "substring_exact_match"),
    "infbench_sum_eng_shots2":          ("infbench_sum_eng_shots2", "f1"),
    "recsys_redial_full":               ("recsys_redial_full", "recsys_recall@5"),
}
MCC_SETS = [k for k in COLUMN if k.startswith("icl_")]


def _ev(repo=None):
    """The benchmark's own scorer module, imported once."""
    global _EV
    if _EV is None:
        repo = repo or os.path.join(HERE, "mab_repo")
        p = os.path.join(repo, "utils", "eval_other_utils.py")
        if not os.path.exists(p):
            raise SystemExit("mab_repo not beside us: %s" % p)
        # recsys reads ./processed_data/Recsys_Redial/entity2id.json relative to CWD
        sys.path.insert(0, repo)
        sp = importlib.util.spec_from_file_location("_mabev", p)
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
        _EV = m
    return _EV


def _fix_gold(answer):
    """Undo a serialisation artefact in OUR files, not in the benchmark.

    The parquet answer column is a numpy array. Some result files stored str(array), so the gold
    arrives as the single string "['France' 'France' 'France']" - one ground truth that no prediction
    can match, which scored ruler at 0.000 against our own 0.900. Parse it back to the list it was.
    """
    if isinstance(answer, str) and answer.startswith("[") and answer.endswith("]"):
        import re as _re
        parts = _re.findall(r"'([^']*)'|\"([^\"]*)\"", answer)
        got = [a or b for a, b in parts]
        if got:
            return got
    return answer


JUDGE_ONLY = {
    # sub_dataset -> what actually scores it. eval_other_utils has a same-named metric that is NOT
    # the benchmark's, so routing here silently returns a plausible wrong number.
    "infbench_sum_eng_shots2": "LLM-as-judge f1 -- use judge_infbench.py "
                               "(mab_repo/llm_based_eval/summarization_evaluate.py). "
                               "Token f1 gives 0.2983 where the judge gives 0.1904.",
}


def score_one(sub_dataset, prediction, answer, repo=None):
    """The benchmark's full metric dict for one prediction."""
    if sub_dataset in JUDGE_ONLY:
        raise SystemExit("REFUSED: %s is not scored by this module. %s"
                         % (sub_dataset, JUDGE_ONLY[sub_dataset]))
    ev = _ev(repo)
    answer = _fix_gold(answer)
    cfg = {"sub_dataset": sub_dataset, "debug": False}
    metrics, _info = ev.post_process({"output": prediction}, answer, cfg)
    return metrics


def score_file(path, sub_dataset, arm, view="prune_annot", repo=None):
    """(mean of the paper's metric, n, per-question vector) for one arm in one result file."""
    key = COLUMN.get(sub_dataset, (sub_dataset, "substring_exact_match"))[1]
    pk = "%s|%s_pred" % (arm, view)
    v = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        pred = r.get(pk)
        if pred is None:
            pred = r.get(arm + "_pred")
        if pred is None:
            continue
        m = score_one(sub_dataset, pred, r["gold"], repo)
        v.append(float(m[key]))
    return (sum(v) / len(v) if v else float("nan")), len(v), v


def mcc(scores):
    """Table 3's MCC column: the mean over the five in-context-learning sets."""
    have = [scores[k] for k in MCC_SETS if k in scores and scores[k] == scores[k]]
    return (sum(have) / len(have)) if len(have) == len(MCC_SETS) else float("nan"), len(have)


if __name__ == "__main__":
    # self-test: the benchmark's scorer must reproduce our substring match on a cell we already run
    import glob
    ev = _ev()
    print("benchmark scorer imported:", ev.__file__)
    for sd, (rk, met) in sorted(COLUMN.items()):
        try:
            m = score_one(sd, "the answer is paris", ["Paris"])
            print("  %-34s -> %-22s %s" % (sd, met, "%.3f" % m[met] if met in m else "KEY MISSING: %s" % sorted(m)[:4]))
        except Exception as e:
            print("  %-34s -> ERROR %s" % (sd, str(e)[:70]))
