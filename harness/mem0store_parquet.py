"""Composability arm "MUSTER over Mem0's store" (PREREG v41 A3): the BEAM parquet with each conversation's context replaced by
# A6: "combined" mode = the BEAM conversation followed by all Mem0 memories (MUSTER runs on source + Mem0 store).
the memories Mem0 2.1.0 built for it (all of them, oldest first, one "[date] memory" record per paragraph), questions and
metadata unchanged. MUSTER then runs unchanged on that store (same BEAM configuration), so the reader sees MUSTER's selection
from Mem0's memories instead of Mem0's own top-k search. $0.  Usage: python mem0store_parquet.py 100K|1M|10M"""
import json, sys, collections, urllib.request
import pandas as pd
sz = sys.argv[1]; COMB = len(sys.argv) > 2 and sys.argv[2] == "combined"; coll = 'mem0_beam10m' if sz == '10M' else 'mem0_beam100k'
pq = {'100K': 'beam/beam100k.parquet', '1M': 'beam/beam1m.parquet', '10M': 'beam/beam10m.parquet'}[sz]
mem = collections.defaultdict(list); off = None
while True:
    body = {'limit': 20000, 'with_payload': ['user_id', 'data', 'date'], 'with_vector': False}
    if off is not None: body['offset'] = off
    r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://localhost:6333/collections/{coll}/points/scroll',
                  data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})))['result']
    for p in r['points']:
        u = p['payload'].get('user_id') or ''
        if u.startswith(f'beam_{sz}_'): mem[int(u.split('_')[2])].append((p['payload'].get('date') or '', p['payload'].get('data') or ''))
    off = r.get('next_page_offset')
    if off is None: break
df = pd.read_parquet(pq)
assert sorted(mem) == list(range(len(df))), (sorted(mem)[:5], len(df))
MEMTXT = [('\n\n'.join(f'[{d}] {t}' for d, t in sorted(mem[i]))) for i in range(len(df))]
df['context'] = [(df.iloc[i]['context'].rstrip() + '\n\n' + MEMTXT[i]) if COMB else MEMTXT[i] for i in range(len(df))]
out = pq.replace('.parquet', '_srcmem0.parquet' if COMB else '_mem0store.parquet'); df.to_parquet(out)
print(out, 'rows', len(df), 'memories/conv mean', round(sum(len(v) for v in mem.values()) / len(mem)))
