"""LoCoMo LLM-as-a-Judge (J), Mem0's evaluation/metrics/llm_judge.py @aae5989: ACCURACY_PROMPT verbatim as the user message,
gpt-4o-mini, temperature 0, response_format json_object, label CORRECT -> 1 else 0. A failed call is retried and, if it still
fails, NOT written (a transport failure is not a zero); rows == unique == n is checked before any score is printed.
  SRC=<reader out> OUT=<judged> CAP_USD=0.5 WORKERS=8 python locomo_judge.py"""
import os, sys, re, json, time, threading, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
os.chdir(os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
SRC, OUT, CAP = os.environ['SRC'], os.environ['OUT'], float(os.environ['CAP_USD']); W = int(os.environ.get('WORKERS', '8'))
JM = os.environ.get('JUDGE_MODEL', 'gpt-4o-mini-2024-07-18'); PR = {'anthropic/claude-haiku-5.5': (0.10, 0.50), 'gpt-4o-mini-2024-07-18': (0.15, 0.60), 'gpt-4.1-2025-04-14': (2.00, 8.00), 'gpt-4.1-mini-2025-04-14': (0.40, 1.60), 'gpt-5-mini-2025-08-07': (0.25, 2.00)}[JM]
G5 = JM.startswith('gpt-5')
OR = JM.startswith('anthropic/')   # via or_client (pinned provider); no temperature/seed on this endpoint, reasoning off
if OR:
    sys.path.insert(0, os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__)))); import or_client   # gpt-5-mini takes no temperature: reasoning_effort minimal instead
STYLE = os.environ.get('JUDGE_STYLE', 'mem0')   # mem0: Mem0 llm_judge.py (user msg, json_object); everm: EverMemOS llm_judge (system + user, no response_format)
P = open('locomo_prompt_judge.txt' if STYLE == 'mem0' else 'everm_judge_user.txt').read(); SYSP = open('everm_judge_sys.txt').read() if STYLE == 'everm' else None
cli = OpenAI(); lock = threading.Lock(); spent = {'usd': 0.0}
src = [json.loads(l) for l in open(SRC)]
done = set()
if os.path.exists(OUT): done = {(r['row'], r['qi']) for r in map(json.loads, open(OUT))}
todo = [r for r in src if (r['row'], r['qi']) not in done]
def label(txt):
    try: return json.loads(txt)['label']
    except Exception:
        m = re.search(r'\{.*\}', txt or '', re.S)
        return json.loads(m.group())['label'] if m else None
def one(r):
    with lock:
        if spent['usd'] >= CAP: raise RuntimeError('cap')
    gold = r['gold'][0] if isinstance(r['gold'], list) else r['gold']
    for a in range(6):
        try:
            if OR:
                msgs = ([{'role': 'user', 'content': P.format(question=r['question'], gold_answer=gold, generated_answer=r['answer'])}] if STYLE == 'mem0'
                        else [{'role': 'system', 'content': SYSP}, {'role': 'user', 'content': P.format(question=r['question'], golden_answer=gold, generated_answer=r['answer'])}])
                t, info = or_client.chat(JM, msgs, reasoning={'effort': 'none'}, temperature=None, max_tokens=400, seed=None,
                                         response_format={'type': 'json_object'} if STYLE == 'mem0' else None)
                lab = label(t)
                if lab is None: raise ValueError('no label')
                with lock: spent['usd'] += info['cost']
                return dict(row=r['row'], qi=r['qi'], category=r['category'], label=lab, score=1 if lab.strip().upper() == 'CORRECT' else 0, judge=JM, style=STYLE)
            if STYLE == 'mem0':
                x = cli.chat.completions.create(model=JM, **({'reasoning_effort': 'minimal'} if G5 else {'temperature': 0.0}), response_format={'type': 'json_object'},
                    messages=[{'role': 'user', 'content': P.format(question=r['question'], gold_answer=gold, generated_answer=r['answer'])}])
            else:
                x = cli.chat.completions.create(model=JM, **({'reasoning_effort': 'minimal'} if G5 else {'temperature': 0.0}), messages=[{'role': 'system', 'content': SYSP},
                    {'role': 'user', 'content': P.format(question=r['question'], golden_answer=gold, generated_answer=r['answer'])}])
            lab = label(x.choices[0].message.content)
            if lab is None: raise ValueError('no label')
            u = x.usage
            with lock: spent['usd'] += (u.prompt_tokens * PR[0] + u.completion_tokens * PR[1]) / 1e6
            return dict(row=r['row'], qi=r['qi'], category=r['category'], label=lab, score=1 if lab.strip().upper() == 'CORRECT' else 0, judge=JM, style=STYLE)
        except Exception as e:
            if a == 5: raise
            time.sleep(3 * 2 ** a)
n = 0
with open(OUT, 'a') as fo, ThreadPoolExecutor(W) as ex:
    for f in as_completed([ex.submit(one, r) for r in todo]):
        try: fo.write(json.dumps(f.result()) + '\n'); n += 1
        except Exception as e: print('  FAIL', repr(e)[:150], flush=True)
J = [json.loads(l) for l in open(OUT)]; keys = {(j['row'], j['qi']) for j in J}
print(f'{os.path.basename(OUT)}: judged {n} now, spent ${spent["usd"]:.4f}; rows {len(J)} unique {len(keys)} n {len(src)}')
if len(J) == len(keys) == len(src):
    c = collections.defaultdict(list)
    for j in J: c[j['category']].append(j['score'])
    print('  J overall %.2f | ' % (100 * sum(j['score'] for j in J) / len(J)) + ' '.join('%s %.2f (%d)' % (k, 100 * sum(v) / len(v), len(v)) for k, v in sorted(c.items())))
else:
    print('  NOT SCORED: rows/unique/n mismatch')
