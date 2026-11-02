"""Honcho (managed cloud, api.honcho.dev) on BEAM 100K, following Honcho's own BEAM runner (plastic-labs/honcho@a1d689b
tests/bench/beam.py: execute_conversation / _process_single_question) with the cloud SDK (honcho-ai 2.5.0).

Per conversation, as the runner does: one workspace, peers "user" (observe_me=True) and "assistant" (observe_me=False), one session,
every BEAM turn added as that peer's message (split at 25,000 characters), batches of 100, then wait for the deriver queue to empty,
then one "consolidate" dream observed by "user" (100K: dreaming ON in Honcho's report) and wait again. Per question, the runner's
get_context branch: session.context(summary=True, peer_target="user", search_query=question) -> SessionContext.to_openai(); that
context is what we hand to OUR reader (the C1/C2/Mem0/past.dev design), so every system is read by the same answer LLM and judge.

One deliberate difference from the runner, stated in results: each message carries its BEAM date as created_at (the session's
time anchor at 09:00Z plus the turn index in minutes), as Mem0 and past.dev received dates. The runner sends no timestamps.

  python honcho_beam_cloud.py ingest <convs e.g. 1-20>   (idempotent per conversation: a marker file records completion)
  python honcho_beam_cloud.py context <convs>             -> results/honcho_ctx_beam100k.jsonl (appends)
  python honcho_beam_cloud.py render                      -> results/honcho_ctx_beam100k.json (comp format for or_beam_reader)

SPLIT=1m (2026-09-23): BEAM 1M (35 conversations, 700 q) with Honcho's reported 1M configuration, dreaming OFF (their blog table:
100K dreaming ON, 500K/1M/10M OFF). Separate workspaces (muster-beam1m-cNN), markers (results/honcho_beam1m/) and outputs
(results/honcho_ctx_beam1m.*); questions from the 1M bundle. Queue wait raised to 6 h per conversation (about 2,100 messages each).
Default SPLIT=100k is unchanged.

SPLIT=10m (2026-09-24): BEAM 10M (10 conversations, 200 q; beam/10M.parquet). Same 1M configuration (dreaming OFF, Honcho's own
10M setting), workspaces muster-beam10m-cNN, markers results/honcho_beam10m/, outputs results/honcho_ctx_beam10m.*, questions from
bundle_out/bundle_beam10m_noxc_u20.jsonl (the MUSTER bOR_glm53f_max_10m_{,noxc_}c_u20 legs). The 10M chat container differs:
chat[i] = {'plan-1'..'plan-10': None except plan-(i+1)}, each plan a list of batches {batch_number, time_anchor, turns=[[message..]]};
messages are taken in the order beam/build_parquet_10m.py serializes them (plans numeric, batches/turns stored, ids 0..N-1 asserted),
then exactly the 1M per-message logic (role -> peer, message time_anchor resets the 09:00Z date + minute counter, 25,000-char split,
batches of 100). About 20,000 messages per conversation (1M: ~2,100), so the queue wait is 24 h per conversation.

  python honcho_beam_cloud.py dry [convs]   ($0, no network, no key, no ledger) builds every batch exactly as ingest would and
                                            prints messages / batches / o200k tokens / list-price $ / first-last created_at per
                                            conversation, with totals; writes DONE_DIR/dry_plan.json.
"""
import os, sys, json, time, datetime, ast, re
os.chdir(os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__))))
from honcho import Honcho
from honcho.api_types import SessionPeerConfig
KEY = os.environ.get('HONCHO_API_KEY', '') if sys.argv[1:2] == ['dry'] else os.environ['HONCHO_API_KEY']
MON = {m: i + 1 for i, m in enumerate(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                                        'October', 'November', 'December'])}
SPLIT = os.environ.get('SPLIT', '100k')
assert SPLIT in ('100k', '1m', '10m'), SPLIT
PARQ = {'100k': 'beam/100K.parquet', '1m': 'beam/1M.parquet', '10m': 'beam/10M.parquet'}[SPLIT]
QBUNDLE = {'100k': 'beam/bundle_beam100k_xc.jsonl', '1m': 'bundle_out/bundle_beam1m_noxc_a20.jsonl',
           '10m': 'bundle_out/bundle_beam10m_noxc_u20.jsonl'}[SPLIT]
DREAM = SPLIT == '100k'
QWAIT = {'100k': 7200, '1m': 6 * 3600, '10m': 24 * 3600}[SPLIT]
DONE_DIR = {'100k': 'results/honcho_beam', '1m': 'results/honcho_beam1m', '10m': 'results/honcho_beam10m'}[SPLIT]
os.makedirs(DONE_DIR, exist_ok=True)
CTX = f'results/honcho_ctx_beam{SPLIT}.jsonl'
# Spend guard (author's rule, 2026-09-23: stop unconditionally once the credit is below $10). Honcho has no balance API, so we meter what we send:
# o200k tokens of every message content x HONCHO_USD_PER_M (list price $2/M; raise it if the probe shows a higher real rate). Before
# each batch of 100 messages: if the ledger plus that batch would exceed HONCHO_SPEND_CAP_USD, write STOP and exit. The cap is set by
# the caller as (balance - $10). A file DONE_DIR/STOP also stops the run before the next batch (manual kill switch).
SPEND_CAP = float(os.environ.get('HONCHO_SPEND_CAP_USD', '0') or 0)
USD_PER_M = float(os.environ.get('HONCHO_USD_PER_M', '2.0'))
LEDGER = f'{DONE_DIR}/spend_ledger.json'

def _ledger():
    return json.load(open(LEDGER)) if os.path.exists(LEDGER) else {'tokens': 0, 'usd': 0.0, 'batches': 0}

def _guard(batch_tokens, label):
    import fcntl   # parallel ingest workers share one ledger: read-check-write under an exclusive lock
    with open(LEDGER + '.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        _guard_locked(batch_tokens, label)

def _guard_locked(batch_tokens, label):
    L = _ledger(); add = batch_tokens / 1e6 * USD_PER_M
    if os.path.exists(f'{DONE_DIR}/STOP'):
        print(label, 'STOP file present; exiting', flush=True); sys.exit(3)
    if SPLIT != '100k' and SPEND_CAP <= 0:
        print('HONCHO_SPEND_CAP_USD not set; refusing to ingest', flush=True); sys.exit(2)
    if SPEND_CAP > 0 and L['usd'] + add > SPEND_CAP:
        msg = f"{label}: ledger ${L['usd']:.2f} + batch ${add:.2f} > cap ${SPEND_CAP:.2f}; STOP"
        open(f'{DONE_DIR}/STOP', 'w').write(msg + '\n'); print(msg, flush=True); sys.exit(3)
    L['tokens'] += batch_tokens; L['usd'] = round(L['usd'] + add, 4); L['batches'] += 1
    json.dump(L, open(LEDGER, 'w'))

def convs(spec):
    lo, hi = [int(x) for x in spec.split('-')]; return list(range(lo, hi + 1))

def load():
    import pandas as pd
    src = pd.read_parquet(PARQ)   # the HF split, chat as sessions of turn dicts
    return {int(r['conversation_id']): r for _, r in src.iterrows()}

def client(cid):
    return Honcho(api_key=KEY, environment='production', workspace_id=f'muster-beam{SPLIT}-c{cid:02d}', timeout=120, max_retries=5)

def wait_queue(h, label, limit=7200):
    t0 = time.time()
    while True:
        try:
            st = h.queue_status()
            pend = getattr(st, 'pending_work_units', 0) or 0; prog = getattr(st, 'in_progress_work_units', 0) or 0
            if pend == 0 and prog == 0: return True
        except Exception as e:
            print(label, 'queue_status error', str(e)[:120], flush=True)
        if time.time() - t0 > limit: return False
        time.sleep(15)

def chat_messages(chat):
    # BEAM messages in stored order. 100K/1M: chat = sessions, each a list of message dicts. 10M: see the docstring; flattened
    # exactly as beam/build_parquet_10m.py flat10m() does, including its two order assertions.
    if SPLIT != '10m':
        return [t for s in chat for t in s]
    plans = []
    for d in chat:
        ks = [k for k in d if d[k] is not None]
        assert len(ks) == 1, ks
        plans.append((int(ks[0].split('-')[1]), d[ks[0]]))
    assert [n for n, _ in plans] == list(range(1, len(plans) + 1)), [n for n, _ in plans]
    msgs = [m for _, pl in plans for b in pl for t in b['turns'] for m in t]
    assert [int(m['id']) for m in msgs] == list(range(len(msgs))), 'message ids not 0..N-1 in stored order'
    return msgs

def build(chat):
    # (role, text, created_at) per Honcho message, in send order: a message's time_anchor sets the date (09:00Z) and restarts the
    # minute counter; created_at = date + k minutes; content split at 25,000 characters.
    chat = ast.literal_eval(chat) if isinstance(chat, str) else chat
    out, date, k = [], None, 0
    msgs = chat_messages(chat)
    if SPLIT == '10m' and msgs and not msgs[0].get('time_anchor'):
        # 10M conversation 7 opens with 244 undated messages (plan-1 batch 1). Left undated, Honcho would stamp them with the
        # ingest day (2026) and the oldest material would sort last; they get the conversation's first anchor date instead
        # (2026-09-24 decision; 100K/1M never take this branch -- their first message is always anchored).
        a0 = next(m['time_anchor'] for m in msgs if m.get('time_anchor'))
        m0 = re.match(r'([A-Za-z]+)-(\d+)-(\d{4})', a0)
        date = datetime.datetime(int(m0.group(3)), MON[m0.group(1)], int(m0.group(2)), 9, 0, tzinfo=datetime.timezone.utc)
    for t in msgs:
        a = t.get('time_anchor')
        if a:
            m = re.match(r'([A-Za-z]+)-(\d+)-(\d{4})', a)
            date = datetime.datetime(int(m.group(3)), MON[m.group(1)], int(m.group(2)), 9, 0, tzinfo=datetime.timezone.utc); k = 0
        k += 1
        when = (date + datetime.timedelta(minutes=k)).isoformat() if date else None
        content = t['content']
        for i in range(0, max(1, len(content)), 25000):
            out.append((t['role'], content[i:i + 25000], when))
    return out

def dry(spec=None):
    # $0: no client, no network, no ledger. Same build() and the same 100-message batches and o200k count the guard charges.
    import tiktoken; enc = tiktoken.get_encoding('o200k_base')
    D = load(); rows = []
    for cid in (convs(spec) if spec else sorted(D)):
        chat = D[cid]['chat']; chat = ast.literal_eval(chat) if isinstance(chat, str) else chat
        raw = chat_messages(chat)
        ids_ok = [int(m['id']) for m in raw] == list(range(len(raw)))
        plan = build(chat); ntok = [len(enc.encode(txt, disallowed_special=())) for _, txt, _ in plan]
        bt = [sum(ntok[i:i + 100]) for i in range(0, len(plan), 100)]
        ws = [w for _, _, w in plan if w]
        r = dict(conv=cid, raw_messages=len(raw), messages=len(plan), split_parts=len(plan) - len(raw), batches=len(bt),
                 o200k_tokens=sum(ntok), max_batch_tokens=max(bt), usd_list=round(sum(ntok) / 1e6 * USD_PER_M, 2),
                 anchors=sum(1 for m in raw if m.get('time_anchor')), undated=len(plan) - len(ws), first=min(ws), last=max(ws),
                 first_sent=ws[0], last_sent=ws[-1], out_of_order=sum(1 for a, b in zip(ws, ws[1:]) if b < a), ids_0_to_n=ids_ok,
                 roles={x: sum(1 for m in raw if m['role'] == x) for x in sorted({m['role'] for m in raw})})
        rows.append(r)
        print(f"c{cid:02d} msgs {r['messages']:6d} (raw {r['raw_messages']}, +{r['split_parts']} split) batches {r['batches']:4d} "
              f"o200k {r['o200k_tokens']:>11,} ${r['usd_list']:7.2f}  {r['first_sent'][:16]} .. {r['last_sent'][:16]} "
              f"out_of_order {r['out_of_order']} ids0..N-1 {ids_ok} undated {r['undated']}", flush=True)
    T = dict(conversations=len(rows), messages=sum(r['messages'] for r in rows), batches=sum(r['batches'] for r in rows),
             o200k_tokens=sum(r['o200k_tokens'] for r in rows), usd_list=round(sum(r['o200k_tokens'] for r in rows) / 1e6 * USD_PER_M, 2),
             all_ids_0_to_n=all(r['ids_0_to_n'] for r in rows))
    print('TOTAL', json.dumps(T), flush=True)
    json.dump(dict(split=SPLIT, usd_per_m=USD_PER_M, per_conv=rows, total=T), open(f'{DONE_DIR}/dry_plan.json', 'w'), indent=1)

def ingest(spec):
    D = load()
    for cid in convs(spec):
        mark = f'{DONE_DIR}/ingested_c{cid:02d}.json'
        if os.path.exists(mark): print('have', cid); continue
        h = client(cid); t0 = time.time()
        user, asst = h.peer('user'), h.peer('assistant')
        sess = h.session(f'beam{SPLIT}-c{cid:02d}-session')
        sess.add_peers([(user, SessionPeerConfig(observe_me=True, observe_others=False)),
                        (asst, SessionPeerConfig(observe_me=False, observe_others=False))])
        plan = build(D[cid]['chat'])
        msgs = [(user if role == 'user' else asst).message(txt, created_at=when) for role, txt, when in plan]
        texts = [txt for _, txt, _ in plan]
        import tiktoken; enc = tiktoken.get_encoding('o200k_base')
        for i in range(0, len(msgs), 100):
            _guard(sum(len(enc.encode(txt, disallowed_special=())) for txt in texts[i:i + 100]), f'c{cid} batch {i // 100}')
            sess.add_messages(msgs[i:i + 100])
        ok1 = wait_queue(h, f'c{cid}', QWAIT)
        ok2 = None
        if DREAM:
            h.schedule_dream(observer='user', session=sess)
            time.sleep(5); ok2 = wait_queue(h, f'c{cid} dream', QWAIT)
        json.dump(dict(conv=cid, messages=len(msgs), deriver_ok=ok1, dream_ok=ok2, seconds=round(time.time() - t0)), open(mark, 'w'))
        print(f'c{cid}: {len(msgs)} messages, deriver {ok1}, dream {ok2}, {time.time() - t0:.0f}s', flush=True)

def context(spec):
    B = {(b['row'], b['qi']): b for b in map(json.loads, open(QBUNDLE))}
    byconv = {}
    for b in B.values(): byconv.setdefault(int(b['conv_id']), []).append(b)
    done = set()
    if os.path.exists(CTX):
        for l in open(CTX): d = json.loads(l); done.add((d['row'], d['qi']))
    with open(CTX, 'a') as fo:
        for cid in convs(spec):
            h = client(cid); sess = h.session(f'beam{SPLIT}-c{cid:02d}-session')
            for b in sorted(byconv[cid], key=lambda x: x['qi']):
                if (b['row'], b['qi']) in done: continue
                t0 = time.time()
                ctx = sess.context(summary=True, peer_target='user', search_query=b['question'])
                msgs = ctx.to_openai(assistant='assistant')
                fo.write(json.dumps(dict(row=b['row'], qi=b['qi'], conv=cid, latency_s=round(time.time() - t0, 2), messages=msgs)) + '\n')
                fo.flush()
            print('context', cid, flush=True)

def render():
    B = {(b['row'], b['qi']): b for b in map(json.loads, open(QBUNDLE))}
    res = []
    for l in open(CTX):
        d = json.loads(l); b = B[(d['row'], d['qi'])]
        text = '\n\n'.join(f"{m['role'].upper()}: {m['content']}" for m in d['messages'])
        res.append(dict(query_id='%s_%s_%d' % (b['conv_id'], b['category'], int(b['cat_index'])), query=b['question'], context=text,
                        n_messages=len(d['messages'])))
    json.dump({'results': res, 'source': 'Honcho cloud (honcho-ai 2.5.0) session.context(summary, peer_target=user, search_query)'},
              open(f'results/honcho_ctx_beam{SPLIT}.json', 'w'))
    import tiktoken, statistics
    enc = tiktoken.get_encoding('cl100k_base'); toks = [len(enc.encode(r['context'])) for r in res]
    print(len(res), 'contexts; tokens mean', round(statistics.mean(toks)), 'median', round(statistics.median(toks)), 'empty', sum(1 for r in res if not r['context'].strip()))

if __name__ == '__main__':
    {'ingest': lambda: ingest(sys.argv[2]), 'context': lambda: context(sys.argv[2]), 'render': render,
     'dry': lambda: dry(sys.argv[2] if len(sys.argv) > 2 else None)}[sys.argv[1]]()
