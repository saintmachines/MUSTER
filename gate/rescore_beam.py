"""Gate stage 0b, BEAM: every score of the paper's main BEAM table (Table 5) re-scored offline from its judge files.

Table 5 reads every store with GLM-5.3-Flash under our prompt and judges with BEAM's official rubric judge (GPT-4.1-mini).
Per row, the judge files behind it, results/bOR_<tag>_bj_gpt-4.1-mini.jsonl, with <tag> ending in _100k, _1m or _10m:

  Mem0 2.1.0, full-ver.          mean of its two searches, M0 and M0R (tags in ROWS below)
  Mem0 2.1.0, full-ver., 19.5K   M0B_b19500
  Honcho                         HC_glmmax, HC1m_glmmax, HC10m_glmmax
  Hindsight 0.10.3               pC_glm_hs
  EverMemOS, EverMemOS 19.5K     pC_glm_emososs, pC_glm_emosd19500
  HippoRAG-v2, 9.7K and 19.5K    pC_glm_hr2r40b9700, pC_glm_hr2r40b19500
  True Memory, 9.7K and 19.5K    mean of two reads, pC_glm_tmb9700 and pC_glm_tmb9700rep (19500 likewise)
  MUSTER, U=20 and U=40          mean of two reads, r4_u20 and r4rep_u20 (u40 likewise)

Score per file: harness/beam_score.py, BEAM's own aggregation (rubric-item mean per question, mean over a conversation's
questions per category, mean over conversations), then the ten-category macro on a 0-100 scale. Mean = mean of the three scales. Asserts all 48
printed values (12 rows, three scales and the mean). Reads files only, no model and no API call.
  MUSTER_HOME=<directory with results/> python gate/rescore_beam.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'harness'))
import beam_score

HOME = os.environ.get('MUSTER_HOME', os.path.join(HERE, '..', 'harness'))
SCALES = (('100k', 400, 20), ('1m', 700, 35), ('10m', 200, 10))   # questions and conversations per scale


def tags(*bases):
    return {s: ['%s_%s' % (b, s) for b in bases] for s, _, _ in SCALES}


ROWS = [  # row as Table 5 prints it, the tags averaged per scale, printed 100K / 1M / 10M / mean
    ('Mem0 2.1.0, full-ver.', {'100k': ['M0_top200_glmmax', 'M0R_top200_100k'], '1m': ['M0_1m_glmmax', 'M0R_top200_1m'],
                               '10m': ['M0_10m_glmmax', 'M0R_top200_10m']}, (75.5, 72.3, 62.3, 70.0)),
    ('Mem0 2.1.0, full-ver., 19.5K', tags('M0B_b19500'), (75.5, 73.1, 63.9, 70.8)),
    ('Honcho', {'100k': ['HC_glmmax'], '1m': ['HC1m_glmmax'], '10m': ['HC10m_glmmax']}, (67.1, 60.7, 48.9, 58.9)),
    ('Hindsight 0.10.3', tags('pC_glm_hs'), (66.4, 67.2, 52.2, 61.9)),
    ('EverMemOS', tags('pC_glm_emososs'), (72.0, 69.4, 60.6, 67.3)),
    ('EverMemOS, 19.5K', tags('pC_glm_emosd19500'), (76.3, 73.1, 62.7, 70.7)),
    ('HippoRAG-v2, 9.7K', tags('pC_glm_hr2r40b9700'), (72.1, 68.6, 58.7, 66.5)),
    ('HippoRAG-v2, 19.5K', tags('pC_glm_hr2r40b19500'), (75.7, 71.3, 60.0, 69.0)),
    ('True Memory, 9.7K', tags('pC_glm_tmb9700', 'pC_glm_tmb9700rep'), (75.4, 72.1, 61.4, 69.6)),
    ('True Memory, 19.5K', tags('pC_glm_tmb19500', 'pC_glm_tmb19500rep'), (78.1, 74.6, 62.4, 71.7)),
    ('MUSTER, U=20', tags('r4_u20', 'r4rep_u20'), (75.0, 72.4, 62.7, 70.0)),
    ('MUSTER, U=40', tags('r4_u40', 'r4rep_u40'), (76.1, 73.4, 64.3, 71.3)),
]


def macro(tag, nq, nconv):
    percat, n = beam_score.score(os.path.join(HOME, 'results', 'bOR_%s_bj_gpt-4.1-mini.jsonl' % tag), nq=nq, nconv=nconv)
    assert percat is not None, (tag, 'covers %d of %d questions' % (n, nq))
    return 100.0 * sum(percat.values()) / len(percat)


bad, n = [], 0
print('%-30s' % 'row (re-scored / printed)' + ''.join('%17s' % h for h in ('100K', '1M', '10M', 'mean')))
for name, tg, printed in ROWS:
    v = [sum(macro(t, nq, nc) for t in tg[s]) / len(tg[s]) for s, nq, nc in SCALES]
    v.append(sum(v) / len(v))
    cells = []
    for x, p in zip(v, printed):
        n += 1
        if '%.1f' % x != '%.1f' % p:
            bad.append('%s %.3f (paper %.1f)' % (name, x, p))
        cells.append('%.3f / %.1f' % (x, p))
    print('%-30s' % name + ''.join('%17s' % c for c in cells))
files = {t for _, tg, _ in ROWS for s in tg for t in tg[s]}
print('\n%d judge files; paper values: %s' % (len(files), 'all %d equal (12 rows, three scales and the mean)' % n
                                                 if not bad else 'DIFF ' + '; '.join(bad)))
sys.exit(1 if bad else 0)
