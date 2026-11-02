"""Score a MUSTER answer file on BEAM with the benchmark authors' own judge code.

Nothing about the scoring is ours. The prompts (src/prompts.py), the per-category evaluate_*
functions and the event-ordering Kendall tau-b path are imported unmodified from
mohammadtavakoli78/BEAM (commit b2da22e). What this file adds is (1) an OpenAI client that presents
the .invoke(...).content interface those functions expect, so the shared venv needs no langchain,
(2) the cost ledger and CAP_USD the guard hook requires, and (3) resumable I/O keyed by (row, qi).

    SRC=results/<tag>.jsonl JUDGE=gpt-4.1-mini OUT=results/<tag>_bj_<judge>.jsonl \
    CAP_USD=2 GUARD=judge_beam_<tag> WORKERS=8 APPROVED_BY_USER=1 python judge_beam.py

The aggregate is computed by beam_score.py exactly as the authors' report_results.py does it:
per question the rubric-item mean (event_ordering: tau_norm), per category the mean over the two
questions of a conversation, then the mean over the twenty conversations, then the ten-category mean.
"""
import os, sys, json, time, threading, types, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
MUSTER_HOME = os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__)))
os.chdir(MUSTER_HOME)
sys.path.insert(0, os.environ.get('BEAM_REPO', os.path.join(MUSTER_HOME, 'beam', 'BEAM')))   # BEAM repository clone at b2da22e

SRC, OUT = os.environ['SRC'], os.environ['OUT']
JUDGE = os.environ.get('JUDGE', 'gpt-4.1-mini')
CAP = float(os.environ.get('CAP_USD', '0'))
GUARD = os.environ.get('GUARD', 'judge_beam')
WORKERS = int(os.environ.get('WORKERS', '8'))
ARM, VIEW = os.environ.get('ARM', 'flr_4_3_3'), os.environ.get('VIEW', 'prune_annot')
PARQ = os.environ.get('PARQ', 'beam/beam100k.parquet')
assert os.environ.get('APPROVED_BY_USER') == '1', 'paid run needs APPROVED_BY_USER=1'
assert CAP > 0, 'CAP_USD must be set'
PRICE = {'gpt-4.1-mini': (0.40, 1.60), 'gpt-4o-mini': (0.15, 0.60), 'gpt-4.1-nano': (0.10, 0.40)}[JUDGE]

# ---- the judge client: same interface as langchain's ChatOpenAI.invoke, same model, temperature 0 ----
from openai import OpenAI
_client = OpenAI()
_lock = threading.Lock()
_spent = {'usd': 0.0, 'in': 0, 'out': 0, 'calls': 0}
_ledger = open('cost_ledger/%s.jsonl' % GUARD, 'a')

class _Msg:
    def __init__(self, c): self.content = c

class JudgeLLM:
    def __init__(self, model): self.model = model
    def invoke(self, prompt):
        msgs = prompt if isinstance(prompt, list) else [{'role': 'user', 'content': prompt}]
        for attempt in range(6):
            try:
                r = _client.chat.completions.create(model=self.model, messages=msgs, temperature=0)
                break
            except Exception as e:
                if attempt == 5: raise
                time.sleep(2 ** attempt)
        u = r.usage; usd = u.prompt_tokens / 1e6 * PRICE[0] + u.completion_tokens / 1e6 * PRICE[1]
        with _lock:
            _spent['usd'] += usd; _spent['in'] += u.prompt_tokens; _spent['out'] += u.completion_tokens; _spent['calls'] += 1
            _ledger.write(json.dumps({'t': int(time.time()), 'model': self.model, 'in': u.prompt_tokens,
                                      'out': u.completion_tokens, 'thoughts': 0, 'usd': round(usd, 7),
                                      'note': 'beam_judge|%s' % os.path.basename(SRC)}) + '\n'); _ledger.flush()
            if _spent['usd'] > CAP:
                raise RuntimeError('CAP_USD %.2f exceeded: %.4f' % (CAP, _spent['usd']))
        return _Msg(r.choices[0].message.content or '')

# ---- stub src.llm so compute_metrics imports without langchain; nothing in it is used by the scorers ----
stub = types.ModuleType('src.llm'); stub.gpt_llm = stub.llama_llm = stub.qwen_llm = None
sys.modules['src.llm'] = stub
import src.evaluation.compute_metrics as CM
CM.initialize_models()
EVAL = {
    'abstention': CM.evaluate_abstention, 'contradiction_resolution': CM.evaluate_contradiction_resolution,
    'event_ordering': CM.evaluate_event_ordering, 'information_extraction': CM.evaluate_information_extraction,
    'instruction_following': CM.evaluate_instruction_following, 'knowledge_update': CM.evaluate_knowledge_update,
    'multi_session_reasoning': CM.evaluate_multi_session_reasoning, 'preference_following': CM.evaluate_preference_following,
    'summarization': CM.evaluate_summarization, 'temporal_reasoning': CM.evaluate_temporal_reasoning,
}

# ---- inputs ----
import pandas as pd
df = pd.read_parquet(PARQ)
META = {i: dict(df.iloc[i]['metadata']) for i in range(len(df))}
QS = {i: list(df.iloc[i]['questions']) for i in range(len(df))}
PRED = '%s|%s_pred' % (ARM, VIEW)
recs = {}
for l in open(SRC):
    r = json.loads(l); recs[(r['row'], r['qi'])] = r
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        d = json.loads(l); done.add((d['row'], d['qi']))
todo = sorted(k for k in recs if k not in done)
print('judge=%s src=%s records=%d already=%d todo=%d cap=$%.2f' % (JUDGE, SRC, len(recs), len(done), len(todo), CAP), flush=True)

llm = JudgeLLM(JUDGE)
def one(key):
    row, qi = key; r = recs[key]; m = META[row]
    cat = m['question_types'][qi]; rubric = list(m['rubric'][qi]); q = QS[row][qi]
    resp = r.get(PRED) or ''  # raw-key-ok: PRED is the prediction TEXT handed to the BEAM judge, not a score
    res = EVAL[cat](rubric=rubric, llm_response=resp, probing_question=q, model=llm)
    score = res['tau_norm'] if cat == 'event_ordering' else res['llm_judge_score']
    return dict(row=row, qi=qi, category=cat, cat_index=int(m['cat_index'][qi]), score=float(score),
                judge=JUDGE, arm=ARM, view=VIEW, n_rubric=len(rubric), result=res)

with open(OUT, 'a') as fo, ThreadPoolExecutor(WORKERS) as ex:
    futs = {ex.submit(one, k): k for k in todo}
    n = 0
    for f in as_completed(futs):
        try:
            d = f.result()
        except Exception as e:
            print('FAIL %s: %s' % (futs[f], str(e)[:120]), flush=True)
            if 'CAP_USD' in str(e): break
            continue
        with _lock:
            fo.write(json.dumps(d) + '\n'); fo.flush()
        n += 1
        if n % 25 == 0: print('  %d/%d  $%.3f  calls=%d' % (n, len(todo), _spent['usd'], _spent['calls']), flush=True)
print('DONE judged=%d  usd=%.4f  calls=%d  in=%d out=%d' % (n, _spent['usd'], _spent['calls'], _spent['in'], _spent['out']), flush=True)
