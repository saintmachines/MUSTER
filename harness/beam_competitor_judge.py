"""Score BEAM-100K answers with a COMPETITOR's own judging code (2026-09-22, at the author's request: score with the competitors' own judges as well).

The scoring code runs unmodified from a pinned clone of the competitor's repo (neither repo has a licence, so it is run from the
clone at run time and never vendored). What this file adds: module stubs so the package __init__ files are not executed, the
(row, qi) -> query-id join, the transport of the judge LLM through or_client (one pinned non-training provider, answering provider
verified), resume by (row, qi), and each runner's aggregation.

  INSTR=amb5d5e8db  AMB @5d5e8db = the code behind Hindsight's leaderboard 73.4: one call per rubric item (BEAM-style unified
                    prompt), event_ordering by Kendall tau-b with LLM item alignment, judge gemini-2.5-flash-lite through the
                    GeminiLLM contract (JSON response schema, temperature not set, retry on unparseable JSON).
  INSTR=valorbrain  ValorBrain fork @48f980d (beam.py/runner.py identical to the 100K run commit d6bd8da): one batched rubric call
                    per question, judge glm-5.2 through the GlmLLM contract (JSON instruction appended, temperature 0,
                    max_tokens 16384, JSON salvaged from reasoning when content is empty), retry-on-zero once (runner.py:233-240).
  Both runners: accuracy = mean score over questions; correct = score >= 0.5 (the pass rate ValorBrain headlines).

  SRC=results/<tag>.jsonl (harness rows: row, qi, prediction)   BUNDLE=beam/bundle_beam100k_xc.jsonl (question/category/ids)
  INSTR=... CLONE=<clone root> OUT=results/<tag>_cj_<INSTR>.jsonl CAP_USD=.. WORKERS=8 APPROVED_BY_USER=1 [DRY=1]
"""
import os, sys, json, gzip, types, importlib, collections, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
MUSTER_HOME = os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__))); os.chdir(MUSTER_HOME); sys.path.insert(0, MUSTER_HOME)
import or_client

INSTR, CLONE, SRC, OUT = os.environ['INSTR'], os.environ['CLONE'], os.environ['SRC'], os.environ['OUT']
BUNDLE = os.environ.get('BUNDLE', 'beam/bundle_beam100k_xc.jsonl')
CAP = float(os.environ.get('CAP_USD', '0')); WORKERS = int(os.environ.get('WORKERS', '8')); DRY = os.environ.get('DRY') == '1'
PRED = 'flr_4_3_3|prune_annot_pred'   # raw-key-ok: prediction text, not a score
JUDGE = {'amb5d5e8db': 'google/gemini-2.5-flash-lite', 'valorbrain': 'z-ai/glm-5.2'}[INSTR]
assert DRY or (os.environ.get('APPROVED_BY_USER') == '1' and CAP > 0), 'paid judge run needs APPROVED_BY_USER=1 and CAP_USD'
LEDGER = f'cost_ledger/cj_{os.path.basename(OUT)}'
spent = {'usd': 0.0}; calls = collections.Counter(); lock = threading.Lock()
# Both competitors' scorers turn ANY exception inside a judge call into a 0.0 score (AMB _rubric_item_score, ValorBrain's
# batch -> per-item fallback). A transport failure on OUR side (rate limit, network, 5xx after or_client's retries) is not a
# verdict, so every failed call marks the question (thread-local, one question per worker thread) and one() refuses to write it;
# a resume re-scores it. (2026-09-22: without this, 24 workers turned rate-limited calls into 62/118 silent zeros.)
TLS = threading.local()

# ---- import the competitor's beam.py (and LLM wrappers) unmodified, without running the package __init__ files ----
src = os.path.join(CLONE, 'src', 'memory_bench')
for name, path in (('memory_bench', src), ('memory_bench.dataset', os.path.join(src, 'dataset')), ('memory_bench.llm', os.path.join(src, 'llm'))):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m
for n in ('rich', 'rich.console', 'rich.table'):
    if n not in sys.modules:
        try: importlib.import_module(n)
        except ImportError: sys.modules[n] = types.ModuleType(n)
if not hasattr(sys.modules['rich.console'], 'Console'): sys.modules['rich.console'].Console = object
if not hasattr(sys.modules['rich.table'], 'Table'): sys.modules['rich.table'].Table = object
beam = importlib.import_module('memory_bench.dataset.beam'); models = importlib.import_module('memory_bench.models')
llm_base = importlib.import_module('memory_bench.llm.base')

def _chat(prompt, **kw):
    if DRY:
        with lock: calls['dry'] += 1
        return ('{"score": 1.0, "reason": "dry", "answer": "YES", "scores": []}', {'reasoning_text': ''})
    try:
        text, info = or_client.chat(JUDGE, [{'role': 'user', 'content': prompt}], ledger=LEDGER, note=INSTR, spent=spent, cap=CAP, **kw)
    except Exception as e:
        TLS.failed = f'{type(e).__name__}: {str(e)[:200]}'
        with lock: calls['transport_fail'] += 1
        raise
    with lock: calls['api'] += 1
    return text, info

if INSTR == 'amb5d5e8db':
    class Judge(llm_base.LLM):
        """AMB 5d5e8db GeminiLLM.generate contract: JSON response schema, temperature unset, up to 6 attempts on unparseable
        output (5 s doubling), last resort {field: raw text for every required field} (which makes float() fail -> 0.0)."""
        @property
        def model_id(self): return 'openrouter:' + JUDGE
        def generate(self, prompt, schema):
            js = {'type': 'object', 'properties': {k: {'type': v.get('type', 'string')} for k, v in schema.properties.items()},
                  'required': list(schema.required), 'additionalProperties': False}
            rf = {'type': 'json_schema', 'json_schema': {'name': 'judge', 'strict': True, 'schema': js}}
            delay, last = 5, ''
            for attempt in range(6):
                text, _ = _chat(prompt, temperature=None, seed=None, response_format=rf)
                last = text
                try:
                    d = json.loads(text)
                    if all(k in d for k in schema.required): return d
                except Exception:
                    import re
                    m = re.search(r'\{.*\}', text or '', re.S)
                    if m:
                        try:
                            d = json.loads(m.group())
                            if all(k in d for k in schema.required): return d
                        except Exception: pass
                if attempt < 5: time.sleep(delay); delay *= 2
            if last and schema.required: return {k: last for k in schema.required}
            raise RuntimeError('unparseable judge output after 6 attempts')
else:
    glm = importlib.import_module('memory_bench.llm.glm')
    class Judge(glm.GlmLLM):
        """ValorBrain GlmLLM: generate() (GatewayLLM: JSON instruction + 4 retries + _parse_json) is inherited unmodified; only
        the transport of _call changes (Z.ai coding endpoint -> OpenRouter, pinned to Z.AI). Same body: temperature 0,
        max_tokens 16384, no reasoning setting (provider default), 3 attempts with 5/10 s backoff, JSON salvaged from reasoning."""
        def __init__(self): self._model = 'glm-5.2'
        @property
        def model_id(self): return 'openrouter:' + JUDGE
        def _call(self, prompt):
            # ValorBrain returns '' after 3 failed attempts, which the batch judge turns into 0.0. That is kept for an EMPTY
            # model reply (the judge answered nothing). A TRANSPORT failure on our side (rate limit, network, 5xx) is not the
            # judge's verdict: it raises, the question is not written, and a resume re-scores it. (2026-09-22: at 24 workers
            # rate-limited calls became 62/118 silent zeros; those rows were quarantined.)
            last = None; transport = 0
            for attempt in range(3):
                try:
                    content, info = _chat(prompt, temperature=0, max_tokens=16384, seed=None, retries=4)
                    if content: return content
                    salv = glm._extract_json_from_text(info.get('reasoning_text') or '')
                    if salv: return salv
                    last = 'empty content'
                except Exception as e:
                    last = f'{type(e).__name__}: {str(e)[:200]}'; transport += 1
                    if 'cap $' in last: raise
                if attempt < 2: time.sleep(5 * (attempt + 1))
            return ''

# ---- queries: the instrument's own exported Query list; answers: ours joined through the bundle ----
Q = {q['id']: q for q in json.load(gzip.open(os.path.join(CLONE, 'data', 'beam', '100k', 'queries.json.gz')))}
B = {(b['row'], b['qi']): b for b in map(json.loads, filter(str.strip, open(BUNDLE)))}
ans = [json.loads(l) for l in open(SRC) if l.strip()]
assert len(Q) == 400 and len(B) == 400 and len(ans) == 400, (len(Q), len(B), len(ans))
LIMIT = int(os.environ.get('LIMIT', '0'))   # smoke test: score only the first LIMIT answers
if LIMIT: ans = ans[:LIMIT]
done = {}
if os.path.exists(OUT):
    for l in open(OUT):
        if l.strip(): d = json.loads(l); done[(d['row'], d['qi'])] = d
ds = beam.BEAMDataset(); judge = Judge()
def one(a):
    r, qi = int(a['row']), int(a['qi']); b = B[(r, qi)]
    qid = '%s_%s_%d' % (b['conv_id'], b['category'], int(b['cat_index'])); q = Q[qid]
    assert b['question'].strip() == q['query'].strip(), (qid, b['question'][:60], q['query'][:60])
    res = models.QueryResult(query_id=qid, query=q['query'], answer=a[PRED] or '', reasoning='', context='(external)',
                             context_tokens=0, retrieve_time_ms=0.0, gold_answers=q['gold_answers'], correct=False,
                             judge_reason='', meta=q['meta'])
    TLS.failed = None
    s = float(ds.score_result(res, judge)); retried = False
    if INSTR == 'valorbrain' and s == 0.0 and (a[PRED] or '').strip():   # runner.py:233-240, fork-added
        time.sleep(3); s = float(ds.score_result(res, judge)); retried = True
    if TLS.failed:
        raise RuntimeError(f'{qid}: a judge call failed on transport ({TLS.failed}) -- not written, resume re-scores it')
    return dict(row=r, qi=qi, query_id=qid, category=q['meta']['question_category'], score=s, correct=s >= 0.5,
                n_rubric=len(q['meta'].get('rubric') or []), retried=retried, instr=INSTR, judge=JUDGE)
todo = [a for a in ans if (int(a['row']), int(a['qi'])) not in done]
print(f'{INSTR}: judge={JUDGE} pin={or_client.PINS[JUDGE]["slug"]} src={SRC} todo={len(todo)} done={len(done)} dry={DRY}', flush=True)
t0 = time.time(); fails = 0
with open(OUT, 'a') as fo, ThreadPoolExecutor(WORKERS) as ex:
    futs = [ex.submit(one, a) for a in todo]
    for i, f in enumerate(as_completed(futs), 1):
        try: d = f.result()
        except Exception as e:
            fails += 1; print('FAIL', type(e).__name__, str(e)[:160], flush=True)
            if isinstance(e, or_client.PinViolation) or 'cap $' in str(e): os._exit(4)
            continue
        with lock: fo.write(json.dumps(d) + '\n'); fo.flush(); done[(d['row'], d['qi'])] = d
        if i % 50 == 0: print(f'  {i}/{len(todo)} ${spent["usd"]:.3f} calls={calls["api"]} {time.time() - t0:.0f}s', flush=True)
rows = list(done.values()); n = len(rows)
bycat = collections.defaultdict(list)
for r in rows: bycat[r['category']].append(r['score'])
print(json.dumps(dict(instr=INSTR, judge=JUDGE, src=SRC, n=n, complete=(n == 400), fails=fails, spent_usd=round(spent['usd'], 4), calls=dict(calls),
                      accuracy_mean=round(100 * sum(r['score'] for r in rows) / max(1, n), 2),
                      pass_rate=round(100 * sum(r['correct'] for r in rows) / max(1, n), 2),
                      retried=sum(r.get('retried', False) for r in rows),
                      per_category={k: round(100 * sum(v) / len(v), 1) for k, v in sorted(bycat.items())})), flush=True)
