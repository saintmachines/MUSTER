#!/usr/bin/env python3
"""Score InfBench-Sum with the benchmark's OWN judge, not with token F1.

mab_repo/README.md:175-183 reports the Summ. column as F1 (LLM-as-judge), following HELMET, and the
scorer is mab_repo/llm_based_eval/summarization_evaluate.py - NOT the token f1_score in
eval_other_utils.py that mab_scorers.py routes to. Those two are different numbers on different
scales: the benchmark's own table gives BM25 19.0 where our token F1 gives 0.0775. Reporting the
token version under the name the paper uses is the DetectiveQA mistake again, so this module imports
the judge's three prompts, its json parser and its model settings and reimplements nothing.

  f1 = fluency * 2 * (recall * precision) / (recall + precision)      summarization_evaluate.py:444
  recall    = r["recall"] / len(keypoints)                            :442
  precision = p["precision"] / p["sentence_count"]                    :443
  judge     = gpt-4o-2024-05-13, temperature 0.1, max 4096            :493

Two deliberate departures, both stated rather than hidden:
  * keypoints and the expert summary are read from the LOCAL parquet instead of re-downloading the
    HuggingFace copy. Verified to be the same fields: metadata['keypoints'] and answers[0].
  * generate_batch is replaced by a thread pool over generate, because the campaign does not use the
    Batch API. Same model, same temperature, same prompts, same parser.
"""
import importlib.util, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, 'mab_repo')
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'llm_based_eval'))
_p = os.path.join(REPO, 'llm_based_eval', 'summarization_evaluate.py')
_sp = importlib.util.spec_from_file_location('_mabsum', _p)
SUM = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(SUM)
from model_utils_openai import OpenAIModel

SRC   = os.environ.get('SRC', 'results/infbench_sum.jsonl')
ARM   = os.environ.get('ARM', 'flrs_4_3_3')
VIEW  = os.environ.get('VIEW', 'prune_annot')
OUT   = os.environ.get('OUT', 'results/infbench_judged_%s_%s.jsonl' % (ARM, VIEW))
PARQ  = os.environ.get('PARQ', 'data_all/data/Long_Range_Understanding-00000-of-00001.parquet')
MAXQ  = int(os.environ.get('MAXQ', '0') or 0)
WORK  = int(os.environ.get('WORKERS', '8') or 8)

import pandas as pd
df = pd.read_parquet(PARQ)
# summarization_evaluate.py:493 pins gpt-4o-2024-05-13 and this account 403s on it - and on
# gpt-4o-2024-08-06 too. judge_lme.py hit the same wall on 2026-08-20 and settled on the alias
# , which is also the string the benchmark's own longmem_qa_evaluate.py names. Use the same
# alias here so both LLM-judged columns of this campaign are judged by one model. It currently
# resolves to gpt-4o-2024-11-20. Written into every record's judge_model field, not a footnote.
JUDGE = os.environ.get('JUDGE', 'gpt-4o')
model = OpenAIModel(JUDGE, temperature=0.1, generation_max_length=4096)

# _METERED, 2026-09-05. This file had NO Guard while judge_lme.py:74 has one, so every
# InfBench-Sum judging pass this project ever ran is absent from cost_ledger and was uncapped --
# CAP_USD and GUARD were read by the caller and silently ignored. That is why `judge_r1_summ` has
# no ledger file. The charge below is MEASURED, not estimated: model_utils_openai.py:229-230
# returns the API's own `input_len` / `output_len` on every call.
import budget_guard as _BG
_G = _BG.Guard(os.environ.get('GUARD', 'judge_' + os.path.basename(OUT).replace('.jsonl', '')),
               cap_usd=float(os.environ.get('CAP_USD', '10')))
_raw_generate = model.generate
_meter_lock = threading.Lock()
def _metered_generate(*a, **kw):
    o = _raw_generate(*a, **kw)
    if o is not None and o.get('input_len') is not None:
        with _meter_lock:
            _G._charge(JUDGE, int(o.get('input_len') or 0), int(o.get('output_len') or 0),
                       0, 'infbench_sum|' + os.path.basename(OUT))
    return o
model.generate = _metered_generate
print('metered: GUARD=%s cap=$%.2f' % (_G.name, _G.cap), flush=True)

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        if l.strip():
            try: done.add((json.loads(l)['row'], json.loads(l)['qi']))
            except Exception: pass
print('resume: %d already judged' % len(done), flush=True)

recs = []
for l in open(SRC):
    if not l.strip(): continue
    r = json.loads(l)
    k = '%s|%s_pred' % (ARM, VIEW)
    if k not in r: continue
    if (r['row'], r['qi']) in done: continue
    recs.append((r['row'], r['qi'], r[k]))
if MAXQ: recs = recs[:MAXQ]
print('to judge: %d' % len(recs), flush=True)

lock = threading.Lock()
fh = open(OUT, 'a')
n_ok = [0]

def one(t):
    row, qi, pred = t
    md = df['metadata'].iloc[row]
    kps = list(md['keypoints'])
    expert = str(list(df['answers'].iloc[row])[0])
    pred = (pred or '').strip()
    fp = SUM.fluency_prompt_book.format(text=pred)
    rp = SUM.recall_prompt_book.format(
        keypoints='\n'.join('%d. %s' % (i + 1, kp) for i, kp in enumerate(kps)), summary=pred)
    pp = SUM.precision_prompt_book.format(expert_summary=expert, summary=pred)
    outs = [model.generate(prompt=x) for x in (fp, rp, pp)]
    if any(o is None or o.get('output') is None for o in outs):
        return
    f = SUM.parse_json(outs[0]['output']); r_ = SUM.parse_json(outs[1]['output']); p = SUM.parse_json(outs[2]['output'])
    if f is None or r_ is None or p is None: return
    try:
        rec_ = r_['recall'] / len(kps) if len(kps) > 0 else 0
        pre_ = p['precision'] / p['sentence_count'] if p['sentence_count'] > 0 else 0
        f1 = f['fluency'] * 2 * (rec_ * pre_) / (rec_ + pre_) if rec_ + pre_ > 0 else 0
    except Exception:
        return
    with lock:
        fh.write(json.dumps({'row': row, 'qi': qi, 'arm': ARM, 'view': VIEW,
                             'judge_model': JUDGE, 'metric': 'llm_judge_f1',
                             'fluency': f['fluency'], 'recall': rec_, 'precision': pre_,
                             'f1': f1}) + '\n')
        fh.flush(); n_ok[0] += 1
        if n_ok[0] % 10 == 0: print('  judged %d' % n_ok[0], flush=True)

with ThreadPoolExecutor(max_workers=WORK) as ex:
    list(ex.map(one, recs))
fh.close()

v = [json.loads(l)['f1'] for l in open(OUT) if l.strip()]
print('InfBench Summ. (LLM-judge F1)  arm=%s view=%s  n=%d  F1=%.4f  x100=%.1f'
      % (ARM, VIEW, len(v), sum(v) / max(len(v), 1), 100 * sum(v) / max(len(v), 1)), flush=True)
