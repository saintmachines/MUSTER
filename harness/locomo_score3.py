"""Mean and sd over the three judge runs of one LoCoMo leg; refuses unless every run has rows == unique == 1540."""
import sys, json, statistics as st, collections
T = sys.argv[1]; runs = []
for k in (1, 2, 3):
    J = [json.loads(l) for l in open(f'results/{T}_J{k}.jsonl')]; keys = {(j['row'], j['qi']) for j in J}
    assert len(J) == len(keys) == 1540, f'{T} J{k}: rows {len(J)} unique {len(keys)}'
    runs.append(J)
ov = [100 * sum(j['score'] for j in J) / 1540 for J in runs]
cat = collections.defaultdict(list)
for J in runs:
    c = collections.defaultdict(list)
    for j in J: c[j['category']].append(j['score'])
    for k, v in c.items(): cat[k].append((100 * sum(v) / len(v), len(v)))
print(f'SCORE {T} 3 runs x 1540 rows (unique) | J {st.mean(ov):.2f} sd {st.stdev(ov):.2f} [' + ' '.join('%.2f' % x for x in ov) + '] | '
      + ' '.join('%s %.2f (n=%d)' % (k, st.mean(x for x, _ in v), v[0][1]) for k, v in sorted(cat.items())))
