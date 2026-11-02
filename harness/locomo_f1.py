"""LoCoMo token F1 with the official scorer (snap-research/locomo task_eval/evaluation.py, transcribed: normalize_answer with the
comma removal and article/'and' removal, Porter-stemmed token F1; category 1 multi-hop = mean over comma-split gold of the max over
comma-split predictions; category 3 open-domain gold cut at the first ';'; 2/3/4 plain F1). Categories as LoCoMo numbers them:
1 multi-hop (282), 2 temporal (321), 3 open-domain (96), 4 single-hop (841); the fifth has no gold answer and is not in our set.
Reads existing answer files only ($0). usage (from the harness directory): python locomo_f1.py results/<leg>.jsonl ..."""
import sys, json, re, string, collections
from nltk.stem import PorterStemmer
ps = PorterStemmer()
CAT = {"multi_hop": 1, "temporal": 2, "open_domain": 3, "single_hop": 4}


def normalize_answer(s):
    s = s.replace(',', "")
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the|and)\b', ' ', s)
    return ' '.join(s.split())


def f1_score(prediction, ground_truth):
    p = [ps.stem(w) for w in normalize_answer(prediction).split()]
    g = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    same = sum((collections.Counter(p) & collections.Counter(g)).values())
    if same == 0:
        return 0
    pr, rc = same / len(p), same / len(g)
    return 2 * pr * rc / (pr + rc)


def f1_multi(prediction, ground_truth):
    preds = [x.strip() for x in prediction.split(',')]
    golds = [x.strip() for x in ground_truth.split(',')]
    return sum(max(f1_score(p, g) for p in preds) for g in golds) / len(golds)


def gold_of(r):
    g = r["gold"]
    if isinstance(g, str) and g.startswith("["):
        try: g = json.loads(g.replace("'", '"'))
        except Exception: pass
    return str(g[0] if isinstance(g, list) else g)


for f in sys.argv[1:]:
    R = [json.loads(l) for l in open(f)]
    assert len(R) == len({(r["row"], r["qi"]) for r in R}) == 1540, (f, len(R))
    by = collections.defaultdict(list)
    for r in R:
        c = CAT[r["category"]]; ans = gold_of(r); out = str(r.get("answer") or "")
        if c == 3: ans = ans.split(';')[0].strip()
        by[c].append(f1_multi(out, ans) if c == 1 else f1_score(out, ans))
    allv = [x for v in by.values() for x in v]
    print("%-36s F1 %.2f | multi-hop %.2f temporal %.2f open-domain %.2f single-hop %.2f | n %d" % (
        f.split("/")[-1][:-6], 100 * sum(allv) / len(allv), *(100 * sum(by[c]) / len(by[c]) for c in (1, 2, 3, 4)), len(allv)))
