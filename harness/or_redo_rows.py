"""Remove selected rows from an OpenRouter leg (answers + every judged file), keeping .bak_<kind> copies, so that a resume re-asks and
re-judges exactly those rows.   python or_redo_rows.py <TAG> <kind>
  kind=error        or_finish == 'error' (provider error delivered as 200, 0 tokens, $0)
  kind=emptylength  empty answer AND or_finish == 'length' (reasoning used the whole think allowance; author, 2026-09-22:
                    re-run every empty answer before scoring -> re-ask with a larger allowance, THINK_ALLOW, and re-score)
Rows that finished with 'stop', and non-empty 'length' rows (answer cut at the 1,200-token answer cap), are never touched."""
import sys, json, os, shutil, glob
os.chdir(os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__))))
tag, kind = sys.argv[1], sys.argv[2]
PRED = 'flr_4_3_3|prune_annot_pred'   # raw-key-ok: prediction text, not a score
src = f'results/{tag}.jsonl'
rows = [json.loads(l) for l in open(src) if l.strip()]
sel = {'error': lambda r: r.get('or_finish') == 'error',
       'emptylength': lambda r: not (r.get(PRED) or '').strip() and r.get('or_finish') == 'length'}[kind]
bad = {(int(r['row']), int(r['qi'])) for r in rows if sel(r)}
print(f'{tag}: {len(bad)} {kind} rows of {len(rows)}: {sorted(bad)}')
if not bad: sys.exit(0)
for p in [src] + sorted(glob.glob(f'results/{tag}_bj_*.jsonl')) + sorted(glob.glob(f'results/{tag}_cj_*.jsonl')):
    shutil.copy(p, f'{p}.bak_{kind}')
    keep = [l for l in open(p) if l.strip() and (int(json.loads(l)['row']), int(json.loads(l)['qi'])) not in bad]
    n0 = sum(1 for l in open(p) if l.strip())
    open(p, 'w').write(''.join(keep)); print(f'  {p}: {n0} -> {len(keep)} rows (backup {p}.bak_{kind})')
