"""Aggregate a judge_beam.py output exactly as BEAM's report_results.py does. Refuses a partial file."""
import json, sys, collections
CATS = ['abstention','contradiction_resolution','event_ordering','information_extraction','instruction_following',
        'knowledge_update','multi_session_reasoning','preference_following','summarization','temporal_reasoning']
def score(path, nq=400, nconv=20):
    by = collections.defaultdict(lambda: collections.defaultdict(list)); keys = set()
    for l in open(path):
        d = json.loads(l); keys.add((d['row'], d['qi'])); by[d['row']][d['category']].append(d['score'])
    if len(keys) != nq: return None, len(keys)
    percat = {}
    for c in CATS:
        convs = [sum(v[c]) / len(v[c]) for v in by.values() if v.get(c)]
        assert len(convs) == nconv, (c, len(convs))
        percat[c] = sum(convs) / nconv
    return percat, len(keys)
if __name__ == '__main__':
    for p in sys.argv[1:]:
        pc, n = score(p)
        print('== %s ==' % p)
        if pc is None: print('   INCOMPLETE %d/400 -- not scored' % n); continue
        for c in CATS: print('   %-26s %6.1f' % (c, 100 * pc[c]))
        print('   %-26s %6.1f' % ('AVERAGE (10-cat macro)', 100 * sum(pc.values()) / 10))
