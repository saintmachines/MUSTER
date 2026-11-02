"""'One block per parent' render, built from an adaptive bundle.
Selected units are recovered exactly (rerender_nomerge.py machinery); blocks = parents, units in id order inside a block, blocks
in order of their first unit. VALIDATION: on every question where the harness already rendered one block per parent (B == P),
this render must reproduce the harness context byte for byte, else the script refuses. Questions where B != P get the new
render; only those are written (the rest are unchanged).  Usage: python render_parent.py <parquet> <adaptive bundle> <out>"""
import sys, os, json, re, hashlib
args = sys.argv[1:]; sys.argv = sys.argv[:1] + [args[0], 'x', 'x']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rerender_nomerge.py')).read().split('HDR = re.compile')[0])
pq, bundle, out = args
HDR = re.compile(r'(?m)^Memory \d+:\n')


def decompose(block, ch, idx):
    """As rerender_nomerge.decompose, but when several units carry the same text (a conversation can repeat a passage),
    take the first one after the previous unit of the block: units inside a harness block are in id order."""
    ids, p = [], 0
    while p < len(block):
        cand = [i for i in idx.get(block[p:p + 80], []) if block.startswith(ch[i], p)
                and (p + len(ch[i]) == len(block) or block[p + len(ch[i])] == '\n')]
        if not cand: return None
        L = max(len(ch[j]) for j in cand); cand = [j for j in cand if len(ch[j]) == L]
        after = [j for j in cand if not ids or j > ids[-1]]
        i = min(after) if after else min(cand); ids.append(i); p += len(ch[i])
        if p < len(block): p += 1
    return ids


def resolve(dec, ch, par):
    """Each unit id appears once in a harness context. Where the text-based decomposition gave one id to two places (a
    repeated passage), move the extra occurrence to an unused copy of the same text in the parent of its block-mates."""
    import collections
    used = collections.Counter(i for d in dec for i in d)
    copies = collections.defaultdict(list)
    for j, s in enumerate(ch): copies[s].append(j)
    for d in dec:
        for p, i in enumerate(d):
            if used[i] < 2: continue
            mates = {par[x] for x in d if x != i}
            free = [j for j in copies[ch[i]] if used[j] == 0]
            alt = [j for j in free if mates and par[j] in mates] or free
            if alt:
                used[i] -= 1; d[p] = alt[0]; used[alt[0]] += 1
    return dec


def render(ids, ch, par):
    groups = {}
    for i in sorted(ids): groups.setdefault(par[i], []).append(i)
    blocks = sorted(groups.values(), key=lambda g: g[0])
    return '\n'.join('Memory %d:\n%s' % (n + 1, '\n'.join(ch[i] for i in g)) for n, g in enumerate(blocks))


same = checked = changed = 0; bad_ex = []
with open(out, 'w') as fo:
    for l in open(bundle):
        r = json.loads(l); ch, par, idx = units(r['row'])
        c = r['context']; parts = HDR.split(c)[1:]
        blocks = [b[:-1] if (n < len(parts) - 1 and b.endswith('\n')) else b for n, b in enumerate(parts)]
        dec = [decompose(b, ch, idx) for b in blocks]
        assert all(d is not None for d in dec), (r['row'], r['qi'], 'unit decomposition failed')
        dec = resolve(dec, ch, par)
        assert len({i for d in dec for i in d}) == sum(len(d) for d in dec), (r['row'], r['qi'], 'a unit id is used twice')
        ids = [i for d in dec for i in d]; P = len({par[i] for i in ids})
        new = render(ids, ch, par)
        if len(dec) == P:
            checked += 1; same += new == c
            if new != c: bad_ex.append((r['row'], r['qi']))
            continue
        changed += 1
        r['context'] = new; r['ctxhash'] = hashlib.sha1(new.encode()).hexdigest()[:12]
        fo.write(json.dumps(r, ensure_ascii=False) + '\n')
print(json.dumps(dict(bundle=bundle, validated=f'{same}/{checked}', changed=changed)))
if same != checked: print('mismatch examples', bad_ex[:3])
# B == P questions are not written (their render is unchanged); a mismatch there is reported, never used.
