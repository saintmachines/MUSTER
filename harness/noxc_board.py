"""The MAB board, published (ON: read-time extraction + LLM [CURRENT]) vs OFF (NO_EXTRACT), three readers. $0: reads files only.

Metric per cell = cellscore.py (the board's single owner): substring = recorded arm|prune_annot; lme = GPT-4o judged file;
summ = judge_infbench f1; rec = mab_scorers recall@5 over the stored predictions. Every mean below is ALSO taken from
cellscore.score() and must agree with the keyed per-question vector used for the paired test (asserted).
Overall = mean of AR (SH, MH, LME, EvQA), TTL (mean(MCC = mean of 5 ICL), Rec), LRU (Summ, Det), SF (FC-SH, FC-MH).
FC OFF: (a) = the published structural prune_annot unchanged (AUTOSTRUCT never calls an LLM); (b) = NOTAG twin (same bytes
minus "[CURRENT] "). gpt-5-mini is scored on the UNCUT answer everywhere (harness-equivalent: no cap was sent).
Paired: W/L/T over the same (row, qi), exact two-sided sign test. Missing legs print as '--'.
"""
import os, sys, json, math, importlib.util
MUSTER_HOME = os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__))); os.chdir(MUSTER_HOME); sys.path.insert(0, MUSTER_HOME)
import cellscore as CS
_sp = importlib.util.spec_from_file_location('ms', 'mab_scorers.py'); MS = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(MS)

ARM = {c: 'flr_4_3_3' for c in ('evq', 'summ', 'det')}; ARM.update({c: 'attprio_sess' for c in ('fcsh', 'fcmh')})
CELLS = ['shqa', 'mhqa', 'lme', 'evq', 'b77', 'clin', 'nlu', 'trcC', 'trcF', 'rec', 'summ', 'det', 'fcsh', 'fcmh']
ON = {'4o': dict(shqa='at_shqa', mhqa='at_mhqa', lme='p24_lme_m', evq='qe40_evq_4o', b77='ra_b77', clin='ra_clin', nlu='ra_nlu',
                 trcC='ra_trcC', trcF='ra_trcF', rec='ra_rec', summ='b4_summ_4o', det='qd_det_4o', fcsh='as6_fcsh', fcmh='as6_fcmh'),
      '41m': dict(shqa='s6a_shqa', mhqa='s6a_mhqa', lme='lg_41m_k200', evq='qe40_evq_41m', b77='m41_b77', clin='m41_clin',
                  nlu='m41_nlu', trcC='m41_trcC', trcF='m41_trcF', rec='m41_rec', summ='b4_summ_41m', det='qd_det_41m',
                  fcsh='m41_fcsh', fcmh='m41_fcmh'),
      '5m': dict(shqa='s6b_shqa', mhqa='s6b_mhqa', lme='lg_5m_k200', evq='qe40_evq_5m', b77='m5m_b77', clin='m5m_clin',
                 nlu='m5m_nlu', trcC='m5m_trcC', trcF='m5m_trcF', rec='m5m_rec', summ='b4_summ_5m', det='qd_det_5m',
                 fcsh='m5m_fcsh', fcmh='m5m_fcmh')}
NX4 = dict(shqa='nx_at_shqa', mhqa='nx_at_mhqa', lme='nx_p24_lme_m', evq='nx_qe40_evq_4o', b77='nx_ra_b77', clin='nx_ra_clin',
           nlu='nx_ra_nlu', trcC='nx_ra_trcC', trcF='nx_ra_trcF', summ='nx_b4_summ_4o', det='nx_qd_det_4o', rec='gOA_nx_4om_rec')
OFF = {'4o': dict(NX4), '41m': {c: 'gOA_nx_41m_' + c for c in CELLS[:12]}, '5m': {c: 'gOA_nx_5m_' + c for c in CELLS[:12]}}
NOTAG = {'4o': ('gOA_4omini_fcsh_struct_notag', 'gOA_4omini_fcmh_struct_notag'),
         '41m': ('gOA_41mini_fcsh_struct_notag', 'gOA_41mini_fcmh_struct_notag'),
         '5m': ('gOA_5mini_fcsh_struct_notag', 'gOA_5mini_fcmh_struct_notag')}
TAGREREAD = {'4o': ('gOA_4omini_fcsh_struct', 'gOA_4omini_fcmh_struct'), '41m': ('gOA_41mini_fcsh_struct', 'gOA_41mini_fcmh_struct'),
             '5m': ('gOA_5mini_fcsh_struct', 'gOA_5mini_fcmh_struct')}
NQ = {'lme': 300, 'rec': 200, 'det': 71}


def arm(c): return ARM.get(c, 'flrs_4_3_3')


def recs(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else None


def vec(cell, tag, uncut=False):
    """{(row, qi): score} on the board metric, or None if the leg is missing/incomplete."""
    a = arm(cell); K = a + '|prune_annot'
    if cell in ('lme', 'summ'):
        rs = recs(f'results/{tag}_judged.jsonl')
        if not rs: return None
        d = {(r['row'], r['qi']): (r['f1'] if cell == 'summ' else r[K]) for r in rs}   # raw-key-ok: judged file, board metric
    elif cell == 'rec':
        rs = recs(f'results/{tag}.jsonl')
        if not rs: return None
        met = MS.COLUMN.get('recsys_redial_full', ('', 'substring_exact_match'))[1]
        d = {}
        for r in rs:
            pred = r.get(K + '_pred', r.get(a + '_pred'))
            if pred is not None: d[(r['row'], r['qi'])] = float(MS.score_one('recsys_redial_full', pred, r['gold'], None)[met])
    else:
        rs = recs(f'results/{tag}.jsonl')
        if not rs: return None
        k = (a + '|uncut') if (uncut and (a + '|uncut') in rs[0]) else K
        d = {(r['row'], r['qi']): float(r[k]) for r in rs if isinstance(r.get(k), (int, float))}   # raw-key-ok: per-question SubEM of this leg
    if len(d) < NQ.get(cell, 100): return None
    return d


def check_mean(cell, tag, d):
    """cellscore.score must give the same mean (the one owner of the metric)."""
    if tag.startswith('gOA_') and cell in ('fcsh', 'fcmh'): return
    try:
        m, n = CS.score(cell, tag, arm(cell), 'prune_annot')
    except CS.Refused as e:
        raise SystemExit(f'cellscore REFUSED {cell} {tag}: {e}')
    mine = sum(d.values()) / len(d)
    if cell == 'rec' or not tag.startswith('gOA_5m'):
        assert m is not None and abs(m - mine) < 1e-9, (cell, tag, m, mine)


def paired(a, b):
    ks = sorted(set(a) & set(b)); w = sum(a[k] > b[k] for k in ks); l = sum(a[k] < b[k] for k in ks); n = w + l
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(w, l) + 1)) / 2 ** n) if n else 1.0
    return w, l, len(ks) - n, p


def comp(s):
    ok = all(s.get(c) is not None for c in CELLS)
    if not ok: return None
    mcc = sum(s[c] for c in ('b77', 'clin', 'nlu', 'trcC', 'trcF')) / 5
    AR = (s['shqa'] + s['mhqa'] + s['lme'] + s['evq']) / 4; TTL = (mcc + s['rec']) / 2
    LRU = (s['summ'] + s['det']) / 2; SF = (s['fcsh'] + s['fcmh']) / 2
    return dict(MCC=mcc, AR=AR, TTL=TTL, LRU=LRU, SF=SF, Overall=(AR + TTL + LRU + SF) / 4)


f = lambda x: '   --' if x is None else f'{100 * x:5.1f}'
summary = {}
for rd in ('4o', '41m', '5m'):
    print(f'\n=== reader {rd}  (ON = published leg; OFF = NO_EXTRACT; 5-mini scored uncut) ===')
    print(f'{"cell":5s} {"ON tag":14s} {"ON":>5s} {"OFF tag":30s} {"OFF":>5s} {"OFF-ON":>6s}  W/L/T (OFF wins/ON wins/ties)  sign p')
    son, soff_a, soff_b = {}, {}, {}
    for c in CELLS:
        on_t = ON[rd][c]; von = vec(c, on_t)
        if von: check_mean(c, on_t, von); son[c] = sum(von.values()) / len(von)
        if c in ('fcsh', 'fcmh'):
            soff_a[c] = son.get(c)
            i = 0 if c == 'fcsh' else 1; nt = NOTAG[rd][i]; vnt = vec(c, nt, uncut=(rd == '5m'))
            soff_b[c] = (sum(vnt.values()) / len(vnt)) if vnt else None
            s = f'{c:5s} {on_t:14s} {f(son.get(c))} {nt + " (b)":30s} {f(soff_b[c])} '
            if von and vnt:
                w, l, t, p = paired(vnt, von); s += f'{100 * (soff_b[c] - son[c]):+6.1f}  {w}/{l}/{t}  p={p:.3f}'
                if rd == '4o': s += '  [as6 bytes != struct bundle: not same-bytes]'
            vtr = vec(c, TAGREREAD[rd][i], uncut=(rd == '5m'))
            if vtr and vnt:
                w, l, t, p = paired(vnt, vtr)
                s += f'  | same bytes: notag vs tag re-read {f(sum(vtr.values()) / len(vtr))}: {w}/{l}/{t} p={p:.3f}'
            print(s); continue
        off_t = OFF[rd][c]; voff = vec(c, off_t, uncut=(rd == '5m'))
        if voff and c != 'rec' and not off_t.startswith('gOA_5m'): check_mean(c, off_t, voff)
        soff_a[c] = soff_b[c] = (sum(voff.values()) / len(voff)) if voff else None
        s = f'{c:5s} {on_t:14s} {f(son.get(c))} {off_t:30s} {f(soff_a[c])} '
        if von and voff:
            w, l, t, p = paired(voff, von); s += f'{100 * (soff_a[c] - son[c]):+6.1f}  {w}/{l}/{t}  p={p:.3f}'
        print(s)
    for lab, s in (('ON (published)', son), ('OFF (a) FC unchanged', soff_a), ('OFF (b) FC NOTAG', soff_b)):
        cp = comp(s); summary[(rd, lab)] = cp
        print(f'  {lab:22s} ' + ('incomplete' if cp is None else '  '.join(f'{k} {100 * v:5.1f}' for k, v in cp.items())))
print('\n=== Overall (MAB scale) ===')
print(f'{"reader":6s} {"ON":>6s} {"OFF(a)":>7s} {"OFF(b)":>7s}')
for rd in ('4o', '41m', '5m'):
    g = lambda lab: summary.get((rd, lab)); h = lambda x: '    --' if not x else f'{100 * x["Overall"]:6.2f}'
    print(f'{rd:6s} {h(g("ON (published)"))} {h(g("OFF (a) FC unchanged")):>7s} {h(g("OFF (b) FC NOTAG")):>7s}')
