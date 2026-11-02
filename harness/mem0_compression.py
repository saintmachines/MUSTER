"""How much does Mem0 2.1.0 compress each BEAM conversation? ($0: Qdrant payloads + parquet, cl100k tokens.)
Per conversation: source tokens (BEAM context string), stored memories (count, tokens), write-time LLM input tokens (usage meter
totals from the cost ledger, per conversation), and what one question reads (Mem0 top-200 context, MUSTER a20 context)."""
import json, glob, os, collections, statistics, urllib.request, re
import pandas as pd, tiktoken
enc = tiktoken.get_encoding('cl100k_base')
def scroll(coll):
    c = collections.defaultdict(list); off = None
    while True:
        body = {'limit': 20000, 'with_payload': ['user_id', 'data'], 'with_vector': False}
        if off is not None: body['offset'] = off
        r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://localhost:6333/collections/{coll}/points/scroll',
                      data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})))['result']
        for p in r['points']: c[p['payload'].get('user_id')].append(p['payload'].get('data') or '')
        off = r.get('next_page_offset')
        if off is None: return c
WRITE_IN = {'100K': 27_585_803 / 20, '1M': 359_818_532 / 35}      # measured Mem0 chat input tokens per conversation (ledger)
for sz, coll, pq, ctxf, bund in [('100K', 'mem0_beam100k', 'beam/beam100k.parquet', 'results/mem0_ctx_m0v210_smoke_top200.json', 'bundle_out/bundle_beam100k_noxc_a20.jsonl'),
                                  ('1M', 'mem0_beam100k', 'beam/beam1m.parquet', 'results/mem0_ctx_m0v210_1m_top200.json', 'bundle_out/bundle_beam1m_noxc_a20.jsonl'),
                                  ('10M', 'mem0_beam10m', 'beam/beam10m.parquet', 'results/mem0_ctx_m0v210_10m_top200.json', 'bundle_out/bundle_beam10m_noxc_a20.jsonl')]:
    mem = scroll(coll); df = pd.read_parquet(pq)
    users = {u: v for u, v in mem.items() if u and u.startswith(f'beam_{sz}_')}
    src, stored_n, stored_t = [], [], []
    for u, v in users.items():
        i = int(u.split('_')[2]); src.append(len(enc.encode(df.iloc[i]['context'], disallowed_special=())))
        stored_n.append(len(v)); stored_t.append(sum(len(enc.encode(x, disallowed_special=())) for x in v))
    rd = [r['ctx_tokens_cl100k'] for r in json.load(open(ctxf))['results']]
    mu = [json.loads(l)['tok'] for l in open(bund)]
    S, T = statistics.mean(src), statistics.mean(stored_t)
    print(f'{sz}: convs {len(users)} | source tokens/conv {S:,.0f} | stored memories {statistics.mean(stored_n):,.0f} '
          f'= {T:,.0f} tokens ({T/S:.1%} of source, {S/T:.1f}x compression, {T/statistics.mean(stored_n):.0f} tok/memory) | '
          f'read per question: Mem0 top-200 {statistics.mean(rd):,.0f} tok ({statistics.mean(rd)/T:.1%} of its store, {statistics.mean(rd)/S:.2%} of source); '
          f'MUSTER a20 {statistics.mean(mu):,.0f} tok ({statistics.mean(mu)/S:.2%} of source)'
          + (f' | Mem0 write-time LLM input {WRITE_IN[sz]:,.0f} tok/conv = {WRITE_IN[sz]/S:.1f}x source' if sz in WRITE_IN else ''))
