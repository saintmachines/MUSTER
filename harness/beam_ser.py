"""Three serializations of the BEAM 100K store, and what derive_config's two statistics read on each.
S0 mirrors BEAM's own RAG baseline (long_term_memory_methods.py:281-330): USER/ASSISTANT pairs, metadata dropped.
S1 is a faithful dump: every field the parquet carries, one line per message, the writer's own id first.
S2 is the same fields with the id NOT at line start, so the serial regex cannot see it.
Nothing is added that the parquet does not hold. The statistics are derive_config.stats() verbatim."""
import os, sys, re, json, collections
MUSTER_HOME = os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, MUSTER_HOME); os.chdir(MUSTER_HOME)
import pandas as pd, tiktoken
ENC = tiktoken.get_encoding('cl100k_base')
NUM = re.compile(r'^\s*\d+\.\s+\S')           # derive_config.py:33, verbatim
PROT = 4096                                    # BEAM RAG baseline context is 32K; MAB protocol chunk is 4096. Both reported.

def s0(chat):
    out = []
    for sess in chat:
        msgs = list(sess)
        for i in range(0, len(msgs), 2):
            u = msgs[i]; a = msgs[i+1] if i+1 < len(msgs) else None
            out.append('USER: %s \n\nASSISTANT: %s' % (u['content'], a['content'] if a else 'N/A'))
    return '\n\n'.join(out)

def s1(chat):
    out = []
    for sess in chat:
        for m in sess:
            meta = []
            if m.get('time_anchor'): meta.append(m['time_anchor'])
            if m.get('index'): meta.append('session,turn %s' % m['index'])
            tag = (' [' + ' | '.join(meta) + ']') if meta else ''
            out.append('%s.%s %s: %s' % (m['id'], tag, m['role'].upper(), m['content']))
    return '\n\n'.join(out)

def s2(chat):
    out = []
    for sess in chat:
        for m in sess:
            meta = ['id %s' % m['id']]
            if m.get('time_anchor'): meta.append(m['time_anchor'])
            if m.get('index'): meta.append('session,turn %s' % m['index'])
            out.append('[%s] %s: %s' % (' | '.join(meta), m['role'].upper(), m['content']))
    return '\n\n'.join(out)

def stats(txt, prot):
    tot = fit = 0
    for p in txt.split('\n\n'):
        if not p.strip(): continue
        t = len(ENC.encode(p)); tot += t
        if t <= prot: fit += t
    lines = [l for l in txt.split('\n') if l.strip()][:20000]
    nf = sum(1 for l in lines if NUM.match(l)) / float(len(lines)) if lines else 0.0
    return fit/float(tot) if tot else 0.0, nf, len(ENC.encode(txt))

df = pd.read_parquet('beam/100K.parquet')
res = {}
for name, fn in (('S0 BEAM-mirror', s0), ('S1 faithful, id first', s1), ('S2 faithful, id inside', s2)):
    covs, nums, toks = [], [], []
    for _, r in df.iterrows():
        c, n, t = stats(fn(list(r['chat'])), PROT); covs.append(c); nums.append(n); toks.append(t)
    res[name] = (covs, nums, toks)
print('%-24s %9s %9s %9s %9s %10s   %s' % ('serialization', 'atom_cov', 'min', 'numbered', 'min', 'tok/conv', 'rule fires'))
print('-' * 96)
for name, (c, n, t) in res.items():
    ac, an = sum(c)/len(c), sum(n)/len(n)
    fire = 'AUTOSTRUCT (serial)' if min(n) >= 0.80 else ('blank atom, chunk=%d' % (PROT//8) if min(c) >= 0.90 else 'sentence, chunk=%d' % PROT)
    print('%-24s %9.4f %9.4f %9.4f %9.4f %10.0f   %s' % (name, ac, min(c), an, min(n), sum(t)/len(t), fire))
print()
print('MAB 14 stores, for reference: atom stores min atom_cov 0.9751; non-atom max 0.5504; numbered fires only on FC (1.0), else ~0.')
print('thresholds: ATOM_TAU 0.90, NUMBERED_TAU 0.80.  Empty bands measured on MAB: 0.998 (serial) and 0.425 (atom_cov).')
print()
print('=== 20 conversations, per-store spread (S1) ===')
c, n, t = res['S1 faithful, id first']
print('atom_cov:', ' '.join('%.3f' % x for x in c))
print('numbered:', ' '.join('%.3f' % x for x in n))

print()
print('=== record-level reading: fraction of \\n\\n-records whose FIRST line opens with N. ===')
def numbered_records(txt):
    recs = [p for p in txt.split('\n\n') if p.strip()]
    return sum(1 for p in recs if NUM.match(p.lstrip().split('\n')[0])) / float(len(recs))
for name, fn in (('S0 BEAM-mirror', s0), ('S1 faithful, id first', s1), ('S2 faithful, id inside', s2)):
    v = [numbered_records(fn(list(r['chat']))) for _, r in df.iterrows()]
    print('%-24s mean %.4f  min %.4f  max %.4f' % (name, sum(v)/len(v), min(v), max(v)))
print()
print('=== line-level, MAB definition, for the same three (from above) ===')
for name, (c, n, t) in res.items(): print('%-24s mean %.4f  min %.4f' % (name, sum(n)/len(n), min(n)))
