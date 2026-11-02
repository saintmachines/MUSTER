"""Score one leg under one judge suffix family: SUF1..SUFk files; refuses unless every run has rows == unique == 1540."""
import sys, json, statistics as st, collections
T, pre, k = sys.argv[1], sys.argv[2], int(sys.argv[3]); ov = []; cat = collections.defaultdict(list)
for i in range(1, k + 1):
    J = [json.loads(l) for l in open(f'results/{T}_{pre}{i}.jsonl')]; keys = {(j['row'], j['qi']) for j in J}
    assert len(J) == len(keys) == 1540, f'{T}_{pre}{i}: rows {len(J)} unique {len(keys)}'
    ov.append(100 * sum(j['score'] for j in J) / 1540); c = collections.defaultdict(list)
    for j in J: c[j['category']].append(j['score'])
    for a, v in c.items(): cat[a].append(100 * sum(v) / len(v))
print(f'SCORE {T} judge={pre} {k} run(s) x 1540 rows (unique) | J {st.mean(ov):.2f}' + (f' sd {st.stdev(ov):.2f}' if k > 1 else '') + ' [' + ' '.join('%.2f' % x for x in ov) + '] | ' + ' '.join('%s %.2f' % (a, st.mean(v)) for a, v in sorted(cat.items())))
