"""MAB x GLM bundle: one JSON line per question with the byte-identical context the harness rendered for the scored arm.

  python make_mab_bundle.py <tag> <arm> <parquet> <rows> <nq> <ref_tag|-> <out.jsonl>

Reads results/<tag>.jsonl (lme_pf29.py DRY=1 DUMP_CTX=1), checks sha1(ctx)[:12] == the record's ctxhash, compares the
scored arm's ctxhash|<arm>|prune_annot with results/<ref_tag>.jsonl (the paid leg it replays), joins the parquet (question,
gold, question_id) and writes: row, qi, qt, dataset, question, gold, question_id, arm, ctxhash, context, tok, n_lines,
n_current, n_units. Refuses a partial or duplicated export. Prints render stats and the bundle sha256 (JSON on the last line).
"""
import sys, json, hashlib, os, collections
import pandas as pd
os.chdir(os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__))))
tag, arm, parq, rows, nq, ref, out = sys.argv[1:8]
rows = [int(x) for x in rows.split(',')]; nq = int(nq)
V = arm + '|' + os.environ.get('VIEW', 'prune_annot')   # VIEW=composite (2026-09-24): keep-all, newest tagged
assert V.endswith(('|prune_annot', '|composite')), V
recs = {}; dup = 0
for l in open(f'results/{tag}.jsonl'):
    if not l.strip(): continue
    r = json.loads(l); k = (r['row'], r['qi'])
    dup += k in recs; recs[k] = r
assert dup == 0, f'{tag}: {dup} duplicated (row,qi)'
assert len(recs) == nq, f'{tag}: partial export {len(recs)}/{nq}'
df = pd.read_parquet(parq)
refd = {}
if ref != '-':
    for l in open(f'results/{ref}.jsonl'):
        if l.strip(): r = json.loads(l); refd[(r['row'], r['qi'])] = r
n = 0; same = 0; stats = collections.Counter(); ncur = []; toks = []; nl = []
with open(out, 'w') as fo:
    for k in sorted(recs):
        r = recs[k]; row, qi = k
        ctx = r['ctx|' + V]   # raw-key-ok: context text, not a score
        h = hashlib.sha1(ctx.encode()).hexdigest()[:12]
        assert h == r['ctxhash|' + V], (k, 'ctx text does not hash to the record ctxhash')
        if refd:
            same += refd.get(k, {}).get('ctxhash|' + V) == h
        m = df['metadata'].iloc[row]; qs = list(df['questions'].iloc[row]); ans = df['answers'].iloc[row]
        a = ans[qi]; gold = [str(x) for x in a] if hasattr(a, '__len__') and not isinstance(a, str) else [str(a)]
        ids = list(m['question_ids']) if m.get('question_ids') is not None else []
        lines = [x for x in ctx.split('\n') if x.strip()]
        c = ctx.count('[CURRENT]'); ncur.append(c); toks.append((r.get('tok_by_arm') or {}).get(V) or 0); nl.append(len(lines))
        fo.write(json.dumps(dict(row=row, qi=qi, qt=r['qt'], dataset=os.environ.get('DSNAME') or (r['qt'] if r['qt'] == 'factconsolidation' else 'longmemeval'),
                                 question=str(qs[qi]), gold=gold, gold_str=r['gold'], question_id=str(ids[qi]) if qi < len(ids) else '',
                                 arm=V, src_tag=tag, ctxhash=h, tok=(r.get('tok_by_arm') or {}).get(V), n_lines=len(lines),
                                 n_current=c, n_units=r.get('n_units'), context=ctx), ensure_ascii=False) + '\n')
        n += 1
sha = hashlib.sha256(open(out, 'rb').read()).hexdigest()
mean = lambda v: sum(v) / max(1, len(v))
res = dict(tag=tag, arm=V, bundle=os.path.abspath(out), n=n, sha256=sha, ref=ref, ref_same=(same if refd else None),
           ref_n=(len(refd) if refd else None), tok_mean=round(mean(toks), 1), tok_max=max(toks), lines_mean=round(mean(nl), 2),
           current_tags_mean=round(mean(ncur), 2), q_with_current=sum(1 for c in ncur if c), current_tags_total=sum(ncur))
print(f"{out}: {n} q  sha256={sha}  ref {ref}: ctx_same {same}/{len(refd) if refd else '-'}  tok mean {res['tok_mean']} "
      f"max {res['tok_max']}  [CURRENT] tags {res['current_tags_total']} (q with any {res['q_with_current']}/{n})")
print('BUNDLE_JSON ' + json.dumps(res))
