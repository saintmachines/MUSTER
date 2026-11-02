"""Merge the four shards of one BEAM leg. Refuses unless exactly 400 distinct (row, qi), one each."""
import os, json, glob, collections
os.chdir(os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__))))
P = os.environ['MERGE_P']
keys = collections.Counter(); rows = []
for f in sorted(glob.glob('results/%s_s?.jsonl' % P)):
    for l in open(f):
        r = json.loads(l); keys[(r['row'], r['qi'])] += 1; rows.append(l)
mx = max(keys.values()) if keys else 0
ok = len(keys) == 400 and mx == 1
print('[merge %s] shards=%d distinct keys %d, max per key %d -> %s' % (P, len(glob.glob('results/%s_s?.jsonl' % P)), len(keys), mx, 'OK' if ok else 'REFUSED'))
if ok:
    open('results/%s.jsonl' % P, 'w').writelines(rows); print('[merge %s] wrote results/%s.jsonl' % (P, P))
