"""BEAM 100K reader legs on OpenRouter models, 2026-09-22. MUSTER's shipped contexts are kept byte-identical and only the
reader changes: the xc bundle (= the contexts of bX10 / bC10, 512x10, checked per question against bC10's ctxhash), prompt C
(system beam_prompt_C_sys.txt, tail beam_prompt_B_qtpl.txt), the user message assembled exactly as lme_pf29.py:4016-4020 does
(context + "\\n\\n" + tail with the question), temperature 0, seed 42, answer budget 1200 (the harness MAXOUT for beam).
Every call goes through or_client.chat (one pinned non-training provider, allow_fallbacks=False, answering provider checked).

Output: results/<TAG>.jsonl in the harness row format (template = results/bC10_beam_41m.jsonl with the prune_annot and
composite predictions replaced), so judge_beam.py, beam_score.py and the paired analysis read it like the table rows.

  source .keys.sh; MODEL=google/gemini-3.1-pro-preview EFFORT=low N_PER_CAT=4 TAG=bOR_g31p_low_p40 CAP_USD=1.5 \\
      $PY or_beam_reader.py            (DRY=1: no call; prints the selection, input tokens and a cost estimate)

EFFORT = 'off' -> or_client.PINS[MODEL]['off'] (the lowest setting the endpoint accepts: the main-table convention); otherwise none / minimal / low / medium / high as reasoning.effort. It is never left
to the provider default. With reasoning on, max_tokens = 1200 + THINK_ALLOW (thinking tokens count against max_tokens);
visible answers longer than 1200 tokens are counted and reported, not cut.
N_PER_CAT = 0 -> all 400 questions; k -> k questions per category, spread over the conversations (pilot).
"""
import os, sys, re, json, time, hashlib, threading, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
MUSTER_HOME = os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__))); os.chdir(MUSTER_HOME); sys.path.insert(0, MUSTER_HOME)
import or_client
import tiktoken

MODEL = os.environ['MODEL']; TAG = os.environ['TAG']
EFFORT = os.environ.get('EFFORT', 'off'); NPC = int(os.environ.get('N_PER_CAT', '0'))
CAP = float(os.environ.get('CAP_USD', '0')); DRY = os.environ.get('DRY') == '1'
WORKERS = int(os.environ.get('WORKERS', '6')); THINK_ALLOW = int(os.environ.get('THINK_ALLOW', '8000'))
BUNDLE = os.environ.get('BUNDLE', 'beam/bundle_beam100k_xc.jsonl')
# CTXSRC=comp:<competitor result file>: answer from THEIR retrieved context (the file records exactly what their reader saw),
# with our prompt and reader. author, 2026-09-22: run ValorBrain and Hindsight under our leak-free prompt as well.
CTXSRC = os.environ.get('CTXSRC', '')
TEMPLATE = os.environ.get('TEMPLATE', 'results/bC10_beam_41m.jsonl')   # U=40: results/bC40_beam_41m.jsonl + its bundle
BUNDLE_SHA256 = os.environ.get('BUNDLE_SHA256', '7a36edddb7e960ad33460f0ac11bb978d44228184350f21521a7d15b004ca2cb')   # default = muster_beam_pkg/BEAM_PINS.json (U=10 xc)
ARM = 'flr_4_3_3|prune_annot'; MAXOUT = 1200
PRICE = {   # $ per 1M tokens (in, out), OpenRouter catalog 2026-09-22; DRY estimate only -- spend comes from usage.cost
    'z-ai/glm-5.3-flash': (0.15, 0.50), 'xiaomi/mimo-v2.6-pro': (0.435, 0.87), 'deepseek/deepseek-v4.1-flash': (0.15, 0.60),
    'qwen/qwen3.8-flash': (0.15, 0.47), 'google/gemini-3.1-pro-preview': (2.00, 12.00),
    'openai/gpt-4.1-mini': (0.40, 1.60), 'openai/gpt-5-mini': (0.25, 2.00), 'qwen/qwen3.5-9b': (0.10, 0.15), 'anthropic/claude-haiku-5.5': (0.10, 0.50)}
THINK = re.compile(r'<think>.*?</think>\s*', re.S)

assert MODEL in or_client.PINS, f'{MODEL} not pinned in or_client.PINS'
REASONING = dict(or_client.PINS[MODEL]['off']) if EFFORT == 'off' else {'effort': EFFORT}
EFF = REASONING['effort']
MAX_TOKENS = MAXOUT if EFF == 'none' else MAXOUT + THINK_ALLOW
if os.environ.get('NO_ANSWER_CAP') == '1':
    # BEAM's own driver sets max_tokens 1000000 (answer_generation.py:17) and ValorBrain's answers run past our 1,200-token cap
    # (median 383, summarization median 1,732). For a matched comparison the cap is lifted: no max_tokens at all.
    MAX_TOKENS = None
assert DRY or CAP > 0, 'CAP_USD must be set'
assert DRY or os.environ.get('OPENROUTER_API_KEY'), 'source .keys.sh first'

PROMPT = os.environ.get('PROMPT', 'C')   # C = our prompt C (bC10/bC40); A = BEAM Listing 44 exactly as bX10 read it
# amb5d5e8db / vb48f980d = the competitor's OWN answer prompt, built by its build_rag_prompt imported unmodified from the pinned clone
# amb5d5e8db = the prompt behind Hindsight's 73.4 (per-category gold metadata);
# vb48f980d = ValorBrain's (category-conditioned hints). Their reader-call contract is mirrored in answer_competitor() below.
COMP = PROMPT in ('amb5d5e8db', 'vb48f980d')
if COMP:
    import types, importlib, gzip
    CLONE = {'amb5d5e8db': 'competitor_clones/amb_5d5e8db', 'vb48f980d': 'competitor_clones/vb_48f980d'}[PROMPT]
    _src = os.path.join(CLONE, 'src', 'memory_bench')
    for _n, _p in (('memory_bench', _src), ('memory_bench.dataset', os.path.join(_src, 'dataset')), ('memory_bench.llm', os.path.join(_src, 'llm'))):
        _m = types.ModuleType(_n); _m.__path__ = [_p]; sys.modules[_n] = _m
    for _n in ('rich', 'rich.console', 'rich.table'):
        if _n not in sys.modules:
            try: importlib.import_module(_n)
            except ImportError: sys.modules[_n] = types.ModuleType(_n)
    if not hasattr(sys.modules['rich.console'], 'Console'): sys.modules['rich.console'].Console = object
    if not hasattr(sys.modules['rich.table'], 'Table'): sys.modules['rich.table'].Table = object
    COMP_DS = importlib.import_module('memory_bench.dataset.beam').BEAMDataset()
    COMP_Q = {q['id']: q for q in json.load(gzip.open(os.path.join(CLONE, 'data', 'beam', '100k', 'queries.json.gz')))}
    COMP_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['reasoning', 'answer'],
                   'properties': {'reasoning': {'type': 'string', 'description': 'Step-by-step explanation of how you used the context to arrive at the answer.'},
                                  'answer': {'type': 'string', 'description': 'The final concise answer to the question. If the context lacks the information, say so.'}}}
    SYS, TAIL = None, None
elif PROMPT == 'C':
    SYS = open('beam_prompt_C_sys.txt').read(); TAIL = open('beam_prompt_B_qtpl.txt').read(); PIN = ('ac515b52', '1e75fa61')
elif PROMPT == 'Cg2':   # C-general v2 (2026-09-23): prompt C byte-for-byte + a 5-line conditional addendum (form / options /
    # worked examples / records over world knowledge); frozen after the BEAM-500K dev split, same question tail as C
    SYS = open('beam_prompt_Cg2_sys.txt').read(); TAIL = open('beam_prompt_B_qtpl.txt').read(); PIN = ('2d386107', '1e75fa61')
else:   # system = fbeam_x10m.sh SYSTEM_MESSAGE, tail = lme_pf29.py RAG_QUERY["beam"] (lines 1285-1286), extracted byte for byte
    SYS = open('beam_prompt_L44_sys.txt').read(); TAIL = open('beam_prompt_L44_qtpl.txt').read(); PIN = ('c131977a', '78b941f6')
assert COMP or (PROMPT in ('A', 'C', 'Cg2') and hashlib.sha1(SYS.encode()).hexdigest().startswith(PIN[0]) and hashlib.sha1(TAIL.encode()).hexdigest().startswith(PIN[1])), f'prompt {PROMPT} bytes drifted'
assert CTXSRC or hashlib.sha256(open(BUNDLE, 'rb').read()).hexdigest() == BUNDLE_SHA256, 'bundle bytes drifted'
items = [json.loads(l) for l in open(BUNDLE) if l.strip()]
if CTXSRC.startswith('comp:'):
    import gzip as _gz
    _p = CTXSRC[5:]
    _d = json.load(_gz.open(_p) if _p.endswith('.gz') else open(_p))
    _B = {(b['conv_id'], b['category'], int(b['cat_index'])): b for b in items}
    comp_items = []
    for _r in _d['results']:
        _c, _rest = _r['query_id'].split('_', 1); _cat, _i = _rest.rsplit('_', 1)
        _b = dict(_B[(_c, _cat, int(_i))])
        assert _b['question'].strip() == _r['query'].strip(), _r['query_id']
        _b['context'] = _r.get('context') or ''
        _b['ctxhash'] = hashlib.sha1(_b['context'].encode()).hexdigest()[:12]
        _b['comp_ctx_source'] = os.path.basename(_p); _b['comp_query_id'] = _r['query_id']
        comp_items.append(_b)
    # ALLOW_SUBSET=1: a competitor measured on part of the split (past.dev on 17 of 20 conversations, 340 questions, within its
    # free credits). The subset is scored against OUR rows restricted to the same questions (subset_score.py), never against 400.
    assert len(comp_items) == int(os.environ.get('NQ_TOTAL', '400')) or os.environ.get('ALLOW_SUBSET') == '1', len(comp_items)
    items = comp_items
tmpl = {}
for l in open(TEMPLATE):
    r = json.loads(l); tmpl[(int(r['row']), int(r['qi']))] = r
# NQ_TOTAL (2026-09-23): the split's question count -- 400 for BEAM 100K (default), 700 for BEAM 1M, whose template is the
# merged ROUTER_ONLY export rows of that split (results/xa20_beam1m_template.jsonl).
NQ_TOTAL = int(os.environ.get('NQ_TOTAL', '400'))
assert (len(items) == NQ_TOTAL or os.environ.get('ALLOW_SUBSET') == '1') and len(tmpl) == NQ_TOTAL, (len(items), len(tmpl), NQ_TOTAL)
CHECK_TMPL = os.environ.get('CTX_TEMPLATE_CHECK', '1') == '1' and not CTXSRC   # 0 for a U with no earlier leg, and for competitor contexts
bad = [(b['row'], b['qi']) for b in items if (CHECK_TMPL and tmpl[(b['row'], b['qi'])]['ctxhash|' + ARM] != b['ctxhash'])
       or hashlib.sha1(b['context'].encode()).hexdigest()[:12] != b['ctxhash']]
assert not bad, f'context differs from {TEMPLATE} on {len(bad)} questions: {bad[:5]}'

if NPC:
    bycat = collections.defaultdict(list)
    for b in sorted(items, key=lambda b: (b['row'], b['qi'])): bycat[b['category']].append(b)
    sel = [v[int(i * len(v) / NPC)] for v in bycat.values() for i in range(NPC)]
else:
    sel = items
OUT = f'results/{TAG}.jsonl'; LEDGER = f'cost_ledger/{TAG}.jsonl'
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        if l.strip(): r = json.loads(l); done.add((int(r['row']), int(r['qi'])))
todo = [b for b in sel if (b['row'], b['qi']) not in done]
enc = tiktoken.get_encoding('cl100k_base')
def comp_query(b):
    q = COMP_Q['%s_%s_%d' % (b['conv_id'], b['category'], int(b['cat_index']))]
    assert q['query'].strip() == b['question'].strip(), (b['row'], b['qi'])
    return q
def user_msg(b):
    if COMP:   # runner.py _prompt_fn: dataset.build_rag_prompt(query, context, task_type, split, category=None, meta); our context as-is
        q = comp_query(b)
        return COMP_DS.build_rag_prompt(q['query'], b['context'], 'open', '100k', None, {**q['meta'], '_raw_response': None})
    return b['context'] + '\n\n' + TAIL.replace('{question}', b['question'])
tin = sum(len(enc.encode(SYS or '')) + len(enc.encode(user_msg(b))) for b in todo)
pi, po = PRICE[MODEL]
print(f'{TAG}: model={MODEL} provider={or_client.PINS[MODEL]["slug"]} prompt={PROMPT} reasoning={REASONING} max_tokens={MAX_TOKENS} '
      f'selected={len(sel)} done={len(done)} todo={len(todo)} contexts={os.path.basename(TEMPLATE)} ({len(items)}/400 ctxhash equal)', flush=True)
print(f'  input ~{tin:,} tok (cl100k) -> ${tin * pi / 1e6:.3f}; output at 232 visible tok/q (bC10 gpt-4.1-mini) -> '
      f'${len(todo) * 232 * po / 1e6:.3f} + reasoning tokens x ${po}/M (measured in the pilot)', flush=True)
if DRY:
    sys.exit(0)

spent = {'usd': 0.0}; lock = threading.Lock()
def answer_competitor(b):
    """The competitor's reader-call contract: one user message, JSON schema {reasoning, answer} (strict), temperature, seed and
    max_tokens NOT set. amb5d5e8db = GeminiLLM.generate: parse JSON, retry up to 6 on unparseable, last resort the raw text.
    vb48f980d = OpenAILLM.generate: _extract_json(text) from the clone; not JSON -> text[:8000] (what all 400 of their answers were);
    JSON missing keys -> one repair call without the schema, else text[:8000]."""
    msgs = [{'role': 'user', 'content': user_msg(b)}]
    rf = {'type': 'json_schema', 'json_schema': {'name': 'response', 'schema': COMP_SCHEMA, 'strict': True}}
    note = f'{TAG}|r{b["row"]}q{b["qi"]}'; infos = []
    def call(m, fmt):
        t, i = or_client.chat(MODEL, m, reasoning=REASONING, temperature=None, max_tokens=None, seed=None, ledger=LEDGER, note=note,
                              spent=spent, cap=CAP, response_format=fmt)
        infos.append(i); return t
    if PROMPT == 'amb5d5e8db':
        last = ''
        for attempt in range(6):
            last = call(msgs, rf)
            try:
                d = json.loads(last)
                if all(k in d for k in ('reasoning', 'answer')): return d['answer'], infos
            except Exception:
                m = re.search(r'\{.*\}', last or '', re.S)
                if m:
                    try:
                        d = json.loads(m.group())
                        if all(k in d for k in ('reasoning', 'answer')): return d['answer'], infos
                    except Exception: pass
            time.sleep(5 * 2 ** attempt)
        return last, infos
    xj = importlib.import_module('memory_bench.llm.openai')._extract_json
    text = call(msgs, rf)
    try: d = xj(text)
    except Exception: d = None
    if d is None: return (text or '').strip()[:8000], infos
    if not all(k in d for k in ('reasoning', 'answer')):
        fix = msgs[0]['content'] + ('\n\nYour previous reply did not follow the required format (missing keys: '
              + ', '.join(k for k in ('reasoning', 'answer') if k not in d) + '). Reply ONLY with a single JSON object containing EXACTLY these keys: reasoning, answer.')
        t2 = call([{'role': 'user', 'content': fix}], None)
        try:
            d2 = xj(t2)
            if d2 and all(k in d2 for k in ('reasoning', 'answer')): return d2['answer'], infos
        except Exception: pass
        return (text or '').strip()[:8000], infos
    return d['answer'], infos

def one(b):
    if COMP:
        text, infos = answer_competitor(b)
        info = dict(infos[-1]); info['cost'] = sum(i['cost'] for i in infos); info['reasoning_tokens'] = sum(i['reasoning_tokens'] or 0 for i in infos)
        info['completion_tokens'] = sum(i['completion_tokens'] or 0 for i in infos); info['calls'] = len(infos)
        text = text if isinstance(text, str) else json.dumps(text)
    else:
        msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': user_msg(b)}]
        text, info = or_client.chat(MODEL, msgs, reasoning=REASONING, temperature=0.0, max_tokens=MAX_TOKENS, seed=42,
                                    ledger=LEDGER, note=f'{TAG}|r{b["row"]}q{b["qi"]}', spent=spent, cap=CAP)
    ans = THINK.sub('', text or '').strip()
    rec = dict(tmpl[(b['row'], b['qi'])])
    rec[ARM + '_pred'] = ans; rec['flr_4_3_3|composite_pred'] = ans   # raw-key-ok: prediction text, not a score
    rec.update({'or_model': MODEL, 'or_provider': info['provider'], 'or_reasoning': REASONING, 'or_max_tokens': MAX_TOKENS,
                'or_prompt_tokens': info['prompt_tokens'], 'or_completion_tokens': info['completion_tokens'],
                'or_reasoning_tokens': info['reasoning_tokens'], 'or_cost': info['cost'], 'or_finish': info.get('finish_reason'),
                'or_visible_tok': len(enc.encode(ans)), 'reader_prompt': PROMPT, 'or_calls': info.get('calls', 1),
                'ctx_source': b.get('comp_ctx_source', 'MUSTER'), 'ctxhash_used': b['ctxhash'],
                'or_max_tokens_sent': None if COMP else MAX_TOKENS})
    return rec
t0 = time.time(); n = 0; fails = 0
with open(OUT, 'a') as fo, ThreadPoolExecutor(WORKERS) as ex:
    futs = {ex.submit(one, b): b for b in todo}
    for f in as_completed(futs):
        b = futs[f]
        try: rec = f.result()
        except Exception as e:
            fails += 1; print('FAIL r%sq%s' % (b['row'], b['qi']), type(e).__name__, str(e)[:600], flush=True)
            if isinstance(e, or_client.PinViolation): os._exit(4)
            continue
        with lock: fo.write(json.dumps(rec, ensure_ascii=False) + '\n'); fo.flush(); n += 1
        if n % 25 == 0: print(f'  {n}/{len(todo)} ${spent["usd"]:.3f} {time.time() - t0:.0f}s', flush=True)
rows = [json.loads(l) for l in open(OUT) if l.strip()]
print(f'DONE {TAG}: +{n} (fails {fails}) rows={len(rows)}/{len(sel)} spent=${spent["usd"]:.4f} {time.time() - t0:.0f}s | '
      f'empty={sum(1 for r in rows if not r[ARM + "_pred"])} length_stop={sum(1 for r in rows if r.get("or_finish") == "length")} '   # raw-key-ok: prediction text, not a score
      f'visible>1200={sum(1 for r in rows if r["or_visible_tok"] > MAXOUT)} '
      f'reasoning_tok_mean={sum(r["or_reasoning_tokens"] or 0 for r in rows) / max(1, len(rows)):.0f} '
      f'providers={sorted({r["or_provider"] for r in rows})}', flush=True)
