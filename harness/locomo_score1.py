"""One judge run of one LoCoMo leg (second judge): refuses unless rows == unique == 1540."""
import sys, json, collections
T, suf = sys.argv[1], sys.argv[2]
J = [json.loads(l) for l in open(f'results/{T}_{suf}.jsonl')]; keys = {(j['row'], j['qi']) for j in J}
assert len(J) == len(keys) == 1540, f'{T}_{suf}: rows {len(J)} unique {len(keys)}'
c = collections.defaultdict(list)
for j in J: c[j['category']].append(j['score'])
print(f'SCORE {T} {suf} 1 run x 1540 rows (unique) | J {100*sum(j["score"] for j in J)/1540:.2f} | ' + ' '.join('%s %.2f (n=%d)' % (k, 100*sum(v)/len(v), len(v)) for k, v in sorted(c.items())))
