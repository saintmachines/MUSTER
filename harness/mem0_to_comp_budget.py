"""E1 / hybrid (PREREG v41): Mem0 re-searched results -> competitor context cut to a TOKEN BUDGET (cl100k), then rendered exactly
as mem0_to_comp.py renders top-k: selected memories sorted oldest-first by BEAM date, "i. [YYYY-MM-DD] memory".
Selection = relevance order, add while the rendered context stays <= budget.
Hybrid mode (HYBRID_BUNDLE=<muster bundle>): context = the MUSTER context for the same question, a blank line, then the Mem0
list filling the remaining budget (budget minus the MUSTER context's tokens).
Usage: [MEM0_BUNDLE=..] [COLLECTION=..] [HYBRID_BUNDLE=..] python mem0_to_comp_budget.py <proj>_k600 <budget_tokens> <outtag>"""
import json, glob, os, sys, statistics, collections, urllib.request
import tiktoken
proj, BUD, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
enc = tiktoken.get_encoding('cl100k_base')
bundle = [json.loads(l) for l in open(os.environ['MEM0_BUNDLE']) if l.strip()]
by_q = collections.defaultdict(list)
for b in bundle: by_q[b['question'].strip()].append(b)
HYB = os.environ.get('HYBRID_BUNDLE')
hctx = {}
if HYB:
    for l in open(HYB):
        b = json.loads(l); hctx[(b['question'].strip(), b['category'])] = b['context']
QDRANT = os.environ.get('QDRANT', 'http://127.0.0.1:6333'); COLL = os.environ.get('COLLECTION', 'mem0_beam100k'); DATE = {}
def fetch_dates(ids):
    ids = [i for i in ids if i not in DATE]
    for k in range(0, len(ids), 500):
        req = urllib.request.Request(f'{QDRANT}/collections/{COLL}/points', method='POST',
              data=json.dumps({'ids': ids[k:k + 500], 'with_payload': True}).encode(), headers={'Content-Type': 'application/json'})
        for pt in json.load(urllib.request.urlopen(req))['result']:
            DATE[pt['id']] = (pt.get('payload') or {}).get('date')
def render(mems):
    mems = sorted(mems, key=lambda m: DATE.get(m['id']) or '')
    return '\n'.join((f"{i}. [{DATE.get(m['id'])}] {m['memory']}" if DATE.get(m['id']) else f"{i}. {m['memory']}")
                     for i, m in enumerate(mems, 1))
res, lens, nsel, undated, exhausted = [], [], [], 0, 0
for p in sorted(glob.glob(f'results/mem0_beam/predicted_{proj}/*.json')):
    if os.path.basename(p).startswith('_'): continue
    d = json.load(open(p)); cands = by_q.get(d['question'].strip(), [])
    hit = [b for b in cands if b['category'] == d['question_type']] or cands
    assert len(hit) == 1, p
    b = hit[0]; mems = d['retrieval']['search_results']; fetch_dates([m['id'] for m in mems])
    head = hctx[(b['question'].strip(), b['category'])] if HYB else ''
    budget = BUD - (len(enc.encode(head)) + 2 if HYB else 0)
    sel, used = [], 0
    for m in mems:                                     # relevance order; per-line tokens + newline
        t = len(enc.encode(f"{len(sel)+1}. [{DATE.get(m['id']) or ''}] {m['memory']}")) + 1
        if used + t > budget: break
        sel.append(m); used += t
    else:
        exhausted += 1
    body = render(sel) if sel else '(No memories available)'
    ctx = (head + '\n\n' + body) if HYB else body
    undated += sum(1 for m in sel if not DATE.get(m['id']))
    lens.append(len(enc.encode(ctx))); nsel.append(len(sel))
    res.append(dict(query_id='%s_%s_%d' % (b['conv_id'], b['category'], int(b['cat_index'])), query=b['question'], context=ctx,
                    n_memories=len(sel), ctx_tokens_cl100k=lens[-1], mem0_question_id=d['question_id']))
keys = [r['query_id'] for r in res]; assert len(keys) == len(set(keys))
out = f'results/mem0_ctx_{tag}.json'
json.dump({'results': res, 'source': f'mem0 2.1.0 OSS re-search {proj}, token budget {BUD}' + (f', hybrid over {HYB}' if HYB else '')}, open(out, 'w'))
print(f'{out}: {len(res)} ctx; memories mean {statistics.mean(nsel):.0f} (min {min(nsel)} max {max(nsel)}); tokens mean {statistics.mean(lens):.0f} '
      f'median {statistics.median(lens):.0f}; all candidates used (store exhausted) in {exhausted}; undated {undated}')
