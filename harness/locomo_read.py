"""LoCoMo reader (2026-10-07). One context per question from BUNDLE (jsonl: row, qi, question, gold, category, context, ctxhash).
PROMPT=ours : our answer prompt, byte-identical to BEAM (system beam_prompt_C_sys.txt, user = context + "\n\n" + tail
              beam_prompt_B_qtpl.txt with the question; the assembly of or_beam_reader.py), via or_client.chat (pinned provider).
PROMPT=mem0 : Mem0's LoCoMo answer prompt, one-memory-list variant (mem0 evaluation/prompts.py ANSWER_PROMPT_ZEP @aae5989,
              verbatim), rendered as their jinja template and sent as the system message, temperature 0 -- their reader contract.
              Model via the OpenAI API directly (gpt-4o-mini is the model the Mem0 paper and the field use).
Resumable: rows already in OUT are skipped. CAP_USD is checked before every call.
  BUNDLE=... OUT=... PROMPT=ours|mem0 MODEL=z-ai/glm-5.3-flash EFFORT=max CAP_USD=3 WORKERS=12 python locomo_read.py"""
import os, sys, re, json, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
MUSTER_HOME = os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__))); os.chdir(MUSTER_HOME); sys.path.insert(0, MUSTER_HOME)
BUNDLE, OUT, PROMPT, MODEL = os.environ['BUNDLE'], os.environ['OUT'], os.environ['PROMPT'], os.environ['MODEL']
EFFORT = os.environ.get('EFFORT', 'off'); CAP = float(os.environ['CAP_USD']); WORKERS = int(os.environ.get('WORKERS', '8'))
THINK = re.compile(r'<think>.*?</think>\s*', re.S)
items = [json.loads(l) for l in open(BUNDLE)]
assert len({(b['row'], b['qi']) for b in items}) == len(items), 'duplicate (row, qi) in bundle'
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: r = json.loads(l); done.add((r['row'], r['qi']))
        except Exception: pass
todo = [b for b in items if (b['row'], b['qi']) not in done]
spent = {'usd': 0.0}; lock = threading.Lock()
if PROMPT == 'ours':
    import or_client
    SYS = open('beam_prompt_C_sys.txt').read(); TAIL = open('beam_prompt_B_qtpl.txt').read()
    import hashlib
    assert hashlib.sha1(SYS.encode()).hexdigest().startswith('ac515b52') and hashlib.sha1(TAIL.encode()).hexdigest().startswith('1e75fa61'), 'prompt bytes drifted'
    pin = or_client.PINS[MODEL]
    REASONING = pin['off'] if EFFORT == 'off' else {'effort': EFFORT}
    MAXT = 1200 + (int(os.environ.get('THINK_ALLOW', '8000')) if EFFORT not in ('off', 'none', 'minimal') else 0)
    def user_msg(b): return b['context'] + '\n\n' + TAIL.replace('{question}', b['question'])
    def call(b):
        msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': user_msg(b)}]
        t, info = or_client.chat(MODEL, msgs, reasoning=REASONING, temperature=0.0, max_tokens=MAXT, seed=42,
                                 note=f'{os.path.basename(OUT)}|r{b["row"]}q{b["qi"]}', spent=spent, cap=CAP)
        return t, dict(provider=info['provider'], cost=info['cost'], ptok=info['prompt_tokens'], ctok=info['completion_tokens'],
                       rtok=info.get('reasoning_tokens'), finish=info.get('finish_reason'))
elif PROMPT in ('mem0', 'everm'):
    # everm: EverMemOS's LoCoMo answer call (evaluation/ @fe436d3e, ACL 2026 paper era): ANSWER_PROMPT.format(context=TEMPLATE.format(
    # speaker_1, speaker_2, speaker_memories=<context>), question) as ONE user message, temperature 0, max_tokens 32768; answer =
    # text after "FINAL ANSWER:" (else the whole text), an empty answer is retried.
    from openai import OpenAI
    cli = OpenAI()
    PRICE = {'gpt-4o-mini-2024-07-18': (0.15, 0.60), 'gpt-4.1-mini-2025-04-14': (0.40, 1.60)}[MODEL]
    if PROMPT == 'mem0':
        TPL = open('locomo_prompt_mem0zep.txt').read()
        assert TPL.count('{{memories}}') == 1 and TPL.count('{{question}}') == 1
        def build(b): return [{'role': 'system', 'content': TPL.replace('{{memories}}', b['context']).replace('{{question}}', b['question'])}], None
    else:
        ANS = open('everm_answer.txt').read(); ETPL = open('everm_template.txt').read()
        SPK = {d['sample_id']: (d['conversation']['speaker_a'], d['conversation']['speaker_b']) for d in json.load(open('locomo/locomo10.json'))}
        def build(b):
            s1, s2 = SPK[b['conv_id']]
            ctx = ETPL.format(speaker_1=s1, speaker_2=s2, speaker_memories=b['context'])
            # EverMemOS sends max_tokens 32768 (its 4.1-mini setting); gpt-4o-mini's completion limit is 16384, so the cap is the model's own
            return [{'role': 'user', 'content': ANS.format(context=ctx, question=b['question'])}], (16384 if MODEL.startswith('gpt-4o-mini') else 32768)
    def call(b):
        with lock:
            if spent['usd'] >= CAP: raise RuntimeError(f'cap ${CAP} reached')
        msgs, mt = build(b)
        for a in range(6):
            try:
                kw = dict(model=MODEL, messages=msgs, temperature=0.0)
                if mt: kw['max_tokens'] = mt
                r = cli.chat.completions.create(**kw)
                if PROMPT == 'everm' and not (r.choices[0].message.content or '').strip(): raise ValueError('empty answer')
                break
            except Exception as e:
                if a == 5: raise
                time.sleep(5 * 2 ** a)
        u = r.usage; cached = getattr(getattr(u, 'prompt_tokens_details', None), 'cached_tokens', 0) or 0
        c = (u.prompt_tokens * PRICE[0] + u.completion_tokens * PRICE[1]) / 1e6   # list price; cached tokens recorded separately
        with lock: spent['usd'] += c
        return r.choices[0].message.content, dict(provider='openai', cost=c, ptok=u.prompt_tokens, ctok=u.completion_tokens,
                                                  rtok=0, cached=cached, finish=r.choices[0].finish_reason)
else:
    raise SystemExit('PROMPT must be ours or mem0')
print(f'{os.path.basename(OUT)}: model={MODEL} prompt={PROMPT} effort={EFFORT} bundle={BUNDLE} n={len(items)} done={len(done)} todo={len(todo)}', flush=True)
def one(b):
    t, info = call(b)
    if PROMPT == 'ours':
        with lock: pass
    raw = THINK.sub('', t or '').strip(); ans = raw
    if PROMPT == 'everm' and 'FINAL ANSWER:' in raw: ans = raw.split('FINAL ANSWER:')[1].strip()
    return dict(row=b['row'], qi=b['qi'], conv_id=b.get('conv_id'), category=b['category'], question=b['question'], gold=b['gold'],
                ctxhash=b['ctxhash'], ctx_tok=b.get('tok'), answer=ans, raw=raw if PROMPT == 'everm' else None, model=MODEL, prompt=PROMPT, **info)
t0 = time.time(); n = fails = 0
with open(OUT, 'a') as fo, ThreadPoolExecutor(WORKERS) as ex:
    futs = {ex.submit(one, b): b for b in todo}
    for f in as_completed(futs):
        try: rec = f.result()
        except Exception as e:
            fails += 1; print('  FAIL', futs[f]['row'], futs[f]['qi'], repr(e)[:200], flush=True); continue
        if PROMPT == 'ours':
            with lock: spent['usd'] = spent['usd']   # or_client adds to spent itself
        fo.write(json.dumps(rec, ensure_ascii=False) + '\n'); fo.flush(); n += 1
        if n % 50 == 0: print(f'  {n}/{len(todo)} ${spent["usd"]:.3f} {time.time()-t0:.0f}s', flush=True)
print(f'  done {n}/{len(todo)} fails={fails} spent=${spent["usd"]:.4f} {time.time()-t0:.0f}s', flush=True)
