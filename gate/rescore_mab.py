"""Gate stage 0b, MemoryAgentBench: the paper's board re-scored offline from its 48 per-cell result tags.

The board is the paper's final configuration: Overall 57.52 / 60.92 / 64.71 with the GPT-4o-mini / GPT-4.1-mini / GPT-5-mini
readers (Tables 1-2; every cell in the appendix tables). Per cell, the leg behind it and the context the reader saw:

  SH-QA, MH-QA, LongMemEval, Rec   leg lz_<cell> (harness/len_legs.json), U=10, flrs_4_3_3; the bundle make_mab_bundle.py
                                   builds from the leg's context dump. Reads gOA_l2k_<r>_<cell>.
  Banking77, CLINC150, NLU,        leg lz_<cell>, U=10, flrs_4_3_3; same bundle recipe. Reads nx_ra_<cell> (GPT-4o-mini),
  TREC-C, TREC-F                   gOA_nx_<r>_<cell> (the other two readers).
  EventQA, InfBench-Sum,           leg lz_<cell>, U=40, flr_4_3_3; the bundle re-rendered with render_rule.py R4 (one block per
  DetectiveQA                      parent, stored order; ATOM=sentence CHUNK=1024 MSCALE=8192, the leg's own). Reads
                                   gOA_r4_<r>_<cell>; DetectiveQA is the mean of three reads (gOA_r4, gOA_r4rep, gOA_r4rep2).
  FC-SH, FC-MH                     leg mg_<cell>_struct (AUTOSTRUCT, attprio_sess); the bundle with "[CURRENT] " removed.
                                   Reads gOA_<4omini|41mini|5mini>_<cell>_struct_notag.

(r = 4om | 41m | 5m.) Gate stage 0d (run_mab_ctx.sh, compare_mab_ctx.py) rebuilds every one of these contexts with the
released code and finds each read's recorded context hash on every question.

Metric per cell as the board defines it (harness/noxc_board.py): LongMemEval the GPT-4o judged file, InfBench-Sum the judge
F1, Rec recall@5 rescored from the stored predictions by the benchmark's own scorer, every other cell substring match;
GPT-5-mini is scored on the uncut answer. Each per-question mean is checked against harness/cellscore.py, the one owner of
every cell's metric, wherever noxc_board.py checks it. AR = mean(SH-QA, MH-QA, LongMemEval, EventQA); TTL = mean(mean of the
five ICL cells, Rec); LRU = mean(InfBench-Sum, DetectiveQA); SF = mean(FC-SH, FC-MH); Overall = mean of the four.
Asserts every cell and Overall against the paper. Reads files only.
  MUSTER_HOME=<harness directory with results/, mab_repo/, processed_data/ beside it> python rescore_mab.py
"""
import os, sys, json, importlib.util
HOME = os.environ.get('MUSTER_HOME', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'harness'))
os.chdir(HOME); sys.path.insert(0, HOME)
import cellscore as CS
_sp = importlib.util.spec_from_file_location('ms', os.path.join(HOME, 'mab_scorers.py'))
MS = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(MS)

CELLS = ['shqa', 'mhqa', 'lme', 'evq', 'b77', 'clin', 'nlu', 'trcC', 'trcF', 'rec', 'summ', 'det', 'fcsh', 'fcmh']
ICL = ('b77', 'clin', 'nlu', 'trcC', 'trcF')
NQ = {'lme': 300, 'rec': 200, 'det': 71}
READERS = (('gpt-4o-mini', '4om', '4omini'), ('gpt-4.1-mini', '41m', '41mini'), ('gpt-5-mini', '5m', '5mini'))
PAPER = {  # appendix tables, 0-100 scale
    '4om': dict(shqa=92.00, mhqa=65.00, lme=56.33, evq=82.00, b77=96.00, clin=90.00, nlu=88.00, trcC=70.00, trcF=65.00,
                rec=13.25, summ=22.24, det=66.20, fcsh=92.00, fcmh=37.00, Overall=57.52),
    '41m': dict(shqa=90.00, mhqa=76.00, lme=61.67, evq=82.00, b77=97.00, clin=89.00, nlu=85.00, trcC=73.00, trcF=49.00,
                rec=15.67, summ=32.33, det=70.89, fcsh=93.00, fcmh=42.00, Overall=60.92),
    '5m': dict(shqa=93.00, mhqa=78.00, lme=71.00, evq=82.00, b77=95.00, clin=94.00, nlu=89.00, trcC=84.00, trcF=59.00,
               rec=11.67, summ=44.47, det=79.34, fcsh=94.00, fcmh=42.00, Overall=64.71)}


def arm(c):
    return 'flr_4_3_3' if c in ('evq', 'summ', 'det') else 'attprio_sess' if c.startswith('fc') else 'flrs_4_3_3'


def tags(m, fc, c):
    if c in ('shqa', 'mhqa', 'lme', 'rec'): return ['gOA_l2k_%s_%s' % (m, c)]
    if c in ('evq', 'summ'): return ['gOA_r4_%s_%s' % (m, c)]
    if c == 'det': return ['gOA_r4_%s_det' % m, 'gOA_r4rep_%s_det' % m, 'gOA_r4rep2_%s_det' % m]
    if c in ICL: return ['nx_ra_%s' % c] if m == '4om' else ['gOA_nx_%s_%s' % (m, c)]
    return ['gOA_%s_%s_struct_notag' % (fc, c)]


def recs(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else None


def vec(c, tag, uncut):
    """{(row, qi): score} on the board metric (harness/noxc_board.py vec)."""
    a = arm(c); K = a + '|prune_annot'
    if c in ('lme', 'summ'):
        rs = recs('results/%s_judged.jsonl' % tag) or []
        d = {(r['row'], r['qi']): (r['f1'] if c == 'summ' else r[K]) for r in rs}
    elif c == 'rec':
        rs = recs('results/%s.jsonl' % tag) or []
        met = MS.COLUMN.get('recsys_redial_full', ('', 'substring_exact_match'))[1]; d = {}
        for r in rs:
            pred = r.get(K + '_pred', r.get(a + '_pred'))
            if pred is not None: d[(r['row'], r['qi'])] = float(MS.score_one('recsys_redial_full', pred, r['gold'], None)[met])
    else:
        rs = recs('results/%s.jsonl' % tag) or []
        k = (a + '|uncut') if (uncut and rs and (a + '|uncut') in rs[0]) else K
        d = {(r['row'], r['qi']): float(r[k]) for r in rs if isinstance(r.get(k), (int, float))}
    if len(d) < NQ.get(c, 100):
        raise SystemExit('incomplete leg %s for %s: %d of %d questions' % (tag, c, len(d), NQ.get(c, 100)))
    return d


def check(c, tag, d, m):
    """The mean must equal cellscore.score where the board checks it (not GPT-5-mini's uncut legs, not the FC NOTAG legs)."""
    if c.startswith('fc') or (m == '5m' and c != 'rec'): return 'n/a'
    s, n = CS.score(c, tag, arm(c), 'prune_annot')
    assert s is not None and abs(s - sum(d.values()) / len(d)) < 1e-9, (c, tag, s)
    return 'ok'


def comp(s):
    mcc = sum(s[c] for c in ICL) / 5; AR = (s['shqa'] + s['mhqa'] + s['lme'] + s['evq']) / 4
    TTL = (mcc + s['rec']) / 2; LRU = (s['summ'] + s['det']) / 2; SF = (s['fcsh'] + s['fcmh']) / 2
    return dict(AR=AR, TTL=TTL, LRU=LRU, SF=SF, Overall=(AR + TTL + LRU + SF) / 4)


out, checks, bad = {}, 0, []
for name, m, fc in READERS:
    s = {}
    for c in CELLS:
        vs = []
        for t in tags(m, fc, c):
            d = vec(c, t, uncut=(m == '5m')); checks += check(c, t, d, m) == 'ok'; vs.append(d)
        keys = set(vs[0])
        assert all(set(v) == keys for v in vs), (c, 'reads cover different questions')
        s[c] = 100 * sum(sum(v[k] for v in vs) / len(vs) for k in keys) / len(keys)   # DetectiveQA: per-question mean of three reads
    out[m] = dict(s, **comp(s))
    bad += ['%s %s %.2f (paper %.2f)' % (m, k, out[m][k], PAPER[m][k]) for k in PAPER[m] if abs(out[m][k] - PAPER[m][k]) > 0.005]
print('%-8s' % 'cell' + ''.join('%14s' % n for n, _, _ in READERS))
for k in CELLS + ['AR', 'TTL', 'LRU', 'SF', 'Overall']:
    print('%-8s' % k + ''.join('%14.2f' % out[m][k] for _, m, _ in READERS))
print('\n48 result tags; %d per-question means equal to cellscore.score; paper values: %s'
      % (checks, 'all 45 equal (14 cells and Overall, three readers)' if not bad else 'DIFF ' + '; '.join(bad)))
sys.exit(1 if bad else 0)
