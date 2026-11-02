"""Render a harness bundle under an explicit container rule, for EVERY question (2026-09-29, at the author's request: render every question exactly).
The selected units are recovered exactly from the harness context (render_parent.py decomposition, duplicate passages
resolved); P = distinct parents among them.
  R2: P < 10 -> every chunk its own block;  P >= 10 -> one block per parent.
  R3: every chunk its own block.
  R4: one block per parent.
Units inside a block and blocks themselves are in id order (the harness order, validated byte for byte on BEAM).
Usage: [ATOM CHUNK MSCALE] python render_rule.py <R2|R3|R4> <parquet> <bundle in> <bundle out>"""
import sys, os, json, re, hashlib, collections
RULE, PQ, SRC, OUT = sys.argv[1:5]
sys.argv = ['x', PQ, SRC, '/dev/null']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'render_parent.py')).read().split('def render(')[0])   # (defines pq/bundle/out of its own: ours are upper-case)


def per_chunk(ids, ch): return '\n'.join('Memory %d:\n%s' % (n + 1, ch[i]) for n, i in enumerate(sorted(ids)))


def per_parent(ids, ch, par):
    g = {}
    for i in sorted(ids): g.setdefault(par[i], []).append(i)
    return '\n'.join('Memory %d:\n%s' % (n + 1, '\n'.join(ch[i] for i in b)) for n, b in enumerate(sorted(g.values(), key=lambda b: b[0])))


stat = collections.Counter()
with open(OUT, 'w') as fo:
    for l in open(SRC):
        r = json.loads(l); ch, par, idx = units(r['row'])
        parts = HDR.split(r['context'])[1:]
        blocks = [b[:-1] if (n < len(parts) - 1 and b.endswith('\n')) else b for n, b in enumerate(parts)]
        dec = [decompose(b, ch, idx) for b in blocks]
        assert all(d is not None for d in dec), (r['row'], r['qi'], 'unit decomposition failed')
        dec = resolve(dec, ch, par)
        ids = [i for d in dec for i in d]
        assert len(set(ids)) == len(ids), (r['row'], r['qi'], 'a unit id is used twice')
        P = len({par[i] for i in ids})
        if RULE == 'R3' or (RULE == 'R2' and P < 10): new = per_chunk(ids, ch); stat['per_chunk'] += 1
        else: new = per_parent(ids, ch, par); stat['per_parent'] += 1
        stat['P<10'] += P < 10; stat['same_as_input'] += new == r['context']
        r['context'] = new; r['ctxhash'] = hashlib.sha1(new.encode()).hexdigest()[:12]
        fo.write(json.dumps(r, ensure_ascii=False) + '\n'); stat['n'] += 1
print(json.dumps(dict(rule=RULE, src=SRC, out=OUT, **stat)))
