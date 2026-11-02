#!/usr/bin/env python3
"""Compare a gate replay (harness OUT with DUMP_CTX) against the published R1 bundle, question by question.

  python gate_compare.py <gate OUT jsonl> <oracle: R1 bundle or harness dump jsonl> [--arm flr_4_3_3|prune_annot]

Prints: joined questions, byte-identical contexts, and for each mismatch a three-tier classification:
  TEXT_SAME       context text identical (hash recomputed)             -> pass
  SAME_UNITS      same set of "Memory k:" blocks, different order       -> ordering/tie difference
  DIFF_UNITS      different block set (n added / n removed)             -> retrieval difference
Exit code 0 only when every joined question is TEXT_SAME and the join is complete.
"""
import sys, json, hashlib, re
HDR = re.compile(r'(?m)^Memory \d+:\n')

def blocks(ctx):
    parts = HDR.split(ctx)[1:]
    return [p.rstrip('\n') for p in parts]

def main():
    out, oracle = sys.argv[1], sys.argv[2]
    arm = sys.argv[sys.argv.index('--arm') + 1] if '--arm' in sys.argv else 'flr_4_3_3|prune_annot'
    gate = {}
    for ln in open(out, encoding='utf-8'):
        r = json.loads(ln); c = r['ctx|' + arm]
        assert hashlib.sha1(c.encode()).hexdigest()[:12] == r['ctxhash|' + arm], ('self-check failed', r['row'], r['qi'])
        gate[(r['row'], r['qi'])] = c
    ora = {}
    for ln in open(oracle, encoding='utf-8'):
        r = json.loads(ln)
        if 'context' in r:                      # a bundle (make_bundle.py): context + ctxhash
            c, h = r['context'], r['ctxhash']
        else:                                   # a harness dump (DUMP_CTX=1): ctx|<arm> + ctxhash|<arm>
            c, h = r['ctx|' + arm], r['ctxhash|' + arm]
        assert hashlib.sha1(c.encode()).hexdigest()[:12] == h, ('oracle self-check failed', r['row'], r['qi'])
        ora[(r['row'], r['qi'])] = c
    keys = sorted(set(gate) & set(ora))
    same = 0; diffs = []
    for k in keys:
        g, o = gate[k], ora[k]
        if g == o:
            same += 1; continue
        bg, bo = blocks(g), blocks(o)
        if sorted(bg) == sorted(bo):
            diffs.append((k, 'SAME_UNITS', len(bg), 0, 0))
        else:
            sg, so = set(bg), set(bo)
            diffs.append((k, 'DIFF_UNITS', len(bg), len(sg - so), len(so - sg)))
    print('gate questions %d | oracle questions %d | joined %d | missing in gate %d | extra in gate %d'
          % (len(gate), len(ora), len(keys), len(set(ora) - set(gate)), len(set(gate) - set(ora))))
    print('byte-identical %d / %d' % (same, len(keys)))
    for k, kind, n, add, rem in diffs:
        print('  row %d qi %d  %s  blocks=%d  gate-only=%d  oracle-only=%d' % (k[0], k[1], kind, n, add, rem))
    ok = (same == len(keys)) and (len(set(ora) - set(gate)) == 0)
    print('GATE', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
