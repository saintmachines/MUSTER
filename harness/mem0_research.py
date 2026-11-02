"""E1 (PREREG v41): re-run Mem0 2.1.0's own search for every BEAM question with a larger limit, NO ingestion.
Same call the authors' runner makes (memory-benchmarks@4b61c5d common/mem0_client.py _search_oss: POST /search
{query, user_id, limit}, rerank off), same query text and user_id as the saved prediction file. Only `limit` changes (600).
Cost: one query embedding (+ entity embeddings) per question on text-embedding-3-small; no LLM call.
Usage: python mem0_research.py <proj> <port> <limit>   -> results/mem0_beam/predicted_<proj>_k<limit>/"""
import json, glob, os, sys, urllib.request, concurrent.futures as cf, time
proj, port, lim = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
src = f'results/mem0_beam/predicted_{proj}'; dst = f'{src}_k{lim}'; os.makedirs(dst, exist_ok=True)
files = [p for p in sorted(glob.glob(f'{src}/*.json')) if not os.path.basename(p).startswith('_')]
def one(p):
    out = os.path.join(dst, os.path.basename(p))
    if os.path.exists(out): return 'skip'
    d = json.load(open(p)); q = d['retrieval']['search_query']
    body = json.dumps({'query': q, 'user_id': d['user_id'], 'limit': lim}).encode()
    for a in range(5):
        try:
            t = time.time()
            r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/search', data=body,
                          headers={'Content-Type': 'application/json'}), timeout=300))
            break
        except Exception as e:
            if a == 4: raise
            time.sleep(2 ** a)
    res = r.get('results', r) if isinstance(r, dict) else r
    old = [m['id'] for m in d['retrieval']['search_results']]
    new = [m['id'] for m in res]
    d['retrieval'] = {'search_query': q, 'search_results': [{k: m.get(k) for k in ('memory', 'score', 'id', 'created_at')} for m in res],
                      'search_latency_ms': round(1000 * (time.time() - t), 1), 'total_results': len(res),
                      'overlap_top200_vs_saved': len(set(new[:len(old)]) & set(old)) / max(1, len(old))}
    json.dump(d, open(out, 'w')); return len(res)
with cf.ThreadPoolExecutor(8) as ex:
    rs = list(ex.map(one, files))
ov = [json.load(open(os.path.join(dst, os.path.basename(p))))['retrieval']['overlap_top200_vs_saved'] for p in files]
n = [x for x in rs if x != 'skip']
print(f'{proj} k{lim}: {len(files)} questions; returned mean {sum(n)/max(1,len(n)):.0f} min {min(n) if n else 0}; '
      f'top-200 overlap with saved: mean {sum(ov)/len(ov):.3f} min {min(ov):.3f}; share < 0.95: {sum(o < .95 for o in ov)/len(ov):.3f}')
