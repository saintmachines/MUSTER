"""Gate stage 0d, comparison. The bundles run_mab_ctx.sh rebuilt with the released code, against (1) the bundle files the paper's
reads consumed (whole-file sha256) and (2) the context hash each paid read recorded for every question, over the 48 legs of the
MemoryAgentBench board (14 cells x 3 readers, DetectiveQA read three times per reader), and LoCoMo / LoCoMo-Refined at U=10.
Reads EXPERIMENT_DIR (read-only) and GATE_DIR/mab_ctx; writes nothing. CPU only.
  GATE_DIR=... EXPERIMENT_DIR=... python compare_mab_ctx.py"""
import os, json, hashlib
G, M = os.environ['GATE_DIR'], os.environ['EXPERIMENT_DIR']
W = os.path.join(G, 'mab_ctx', 'bundle_out')
CELLS = ['shqa', 'mhqa', 'lme', 'evq', 'b77', 'clin', 'nlu', 'trcC', 'trcF', 'rec', 'summ', 'det', 'fcsh', 'fcmh']
ICL = ('b77', 'clin', 'nlu', 'trcC', 'trcF')
READERS = (('4om', '4omini'), ('41m', '41mini'), ('5m', '5mini'))   # GPT-4o-mini, GPT-4.1-mini, GPT-5-mini


def arm(c):
    return 'flr_4_3_3' if c in ('evq', 'summ', 'det') else 'attprio_sess' if c.startswith('fc') else 'flrs_4_3_3'


def legs(c):
    """The board's paid read tags for one cell, all three readers (the legs behind 57.52 / 60.92 / 64.71)."""
    out = []
    for m, fc in READERS:
        if c in ('shqa', 'mhqa', 'lme', 'rec'): out.append('gOA_l2k_%s_%s' % (m, c))
        elif c in ('evq', 'summ'): out.append('gOA_r4_%s_%s' % (m, c))
        elif c == 'det': out += ['gOA_r4_%s_det' % m, 'gOA_r4rep_%s_det' % m, 'gOA_r4rep2_%s_det' % m]
        elif c in ICL: out.append('nx_ra_%s' % c if m == '4om' else 'gOA_nx_%s_%s' % (m, c))
        else: out.append('gOA_%s_%s_struct_notag' % (fc, c))
    return out


def final(c):
    """The rebuilt bundle the board's contexts must equal: R4 for EventQA / DetectiveQA / InfBench-Sum, NOTAG for FC."""
    return os.path.join(W, 'r4_lz_%s.jsonl' % c if c in ('evq', 'summ', 'det') else
                        'mg_%s_struct_notag.jsonl' % c if c.startswith('fc') else 'full_lz_%s.jsonl' % c)


ORACLE = {c: [('full_lz_%s.jsonl' % c, 'bundle_out/len0928/full_lz_%s.jsonl' % c)] for c in ('shqa', 'mhqa', 'lme', 'rec', 'evq', 'det', 'summ')}
for c in ('evq', 'det', 'summ'): ORACLE[c].append(('r4_lz_%s.jsonl' % c, 'bundle_out/rules/r4_mab_%s.jsonl' % c))
for c in ('fcsh', 'fcmh'):
    ORACLE[c] = [('mg_%s_struct.jsonl' % c, 'bundle_out/mab_glm/mg_%s_struct.jsonl' % c),
                 ('mg_%s_struct_notag.jsonl' % c, 'bundle_out/mab_glm/mg_%s_struct_notag.jsonl' % c)]


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


tot_q = tot_ok = tot_legs = legs_ok = files = files_ok = 0
print('%-5s %5s %8s  %-40s  %s' % ('cell', 'q', 'selfhash', 'whole file, rebuilt vs oracle', 'board legs: questions with equal ctxhash'))
for c in CELLS:
    g = rows(final(c)); gh = {(r['row'], r['qi']): r['ctxhash'] for r in g}
    selfok = sum(hashlib.sha1(r['context'].encode()).hexdigest()[:12] == r['ctxhash'] for r in g)
    fs = []
    for mine, orc in ORACLE.get(c, []):
        a, b = sha(os.path.join(W, mine)), sha(os.path.join(M, orc)); files += 1; files_ok += a == b
        fs.append('%s %s' % (mine.replace('.jsonl', ''), 'MATCH' if a == b else 'DIFF %s/%s' % (a[:8], b[:8])))
    ls = []
    for t in legs(c):
        rs = rows(os.path.join(M, 'results', t + '.jsonl')); k = 'ctxhash|%s|prune_annot' % arm(c)
        n_eq = sum(gh.get((r['row'], r['qi'])) == r[k] for r in rs)
        tot_legs += 1; legs_ok += n_eq == len(rs) == len(g); tot_q += len(rs); tot_ok += n_eq
        ls.append('%d/%d' % (n_eq, len(rs)))
    print('%-5s %5d %4d/%-3d  %-40s  %s' % (c, len(g), selfok, len(g), '; '.join(fs) or '(per question only)', ' '.join(ls)))
print('\nMemoryAgentBench: %d/%d legs with every question equal; %d/%d questions; %d/%d oracle files byte-identical'
      % (legs_ok, tot_legs, tot_ok, tot_q, files_ok, files))
lf = lo = 0
for t in ('lcm_u10', 'lcmrefc_u10'):
    for r in ('r0', 'r4'):
        p = os.path.join(W, '%s_%s.jsonl' % (t, r))
        a, b = sha(p), sha(os.path.join(M, 'bundle_out/locomo/%s_%s.jsonl' % (t, r)))
        lf += 1; lo += a == b
        print('%-12s %s %5d q  rebuilt %s  oracle %s  %s' % (t, r, len(rows(p)), a[:12], b[:12], 'MATCH' if a == b else 'DIFF'))
print('LoCoMo / LoCoMo-Refined U=10: %d/%d files byte-identical' % (lo, lf))
