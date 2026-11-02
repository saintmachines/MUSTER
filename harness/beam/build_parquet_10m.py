"""BEAM 10M -> a MemoryAgentBench-shaped parquet the harness can read (same serialization S1 as beam/build_parquet.py).

Source: the HF dataset Mohammadta/BEAM, file data/1M-00000-of-00001.parquet (columns as the 100K split: conversation_id,
chat, probing_questions, ...). Output: beam/beam1m.parquet, source="beam_1m". Nothing added, nothing dropped: one line per
message "<id>. [<time_anchor> | session,turn <index>] <ROLE>: <content>"; questions in probing_questions' own (category,
index) order; answers=[[ideal]]; metadata question_types / cat_index / rubric / conv_id.

  python build_parquet_1m.py       (needs HF access; ~500 MB download)
"""
import os, ast
os.chdir(os.environ.get('MUSTER_HOME', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# 10M differs from 1M only in the chat container: chat[i] = {'plan-1'..'plan-10': None except 'plan-(i+1)'}, each plan a list
# of batches {batch_number, time_anchor, turns=[[message, ...], ...]}. Plans are read in numeric order (plan-2 before plan-10),
# batches and turns in stored order; message ids then run 0..N-1 without gaps (asserted), i.e. the stored chronology.
SRC = 'beam/10M.parquet'
if not os.path.exists(SRC):
    fs = [hf_hub_download('Mohammadta/BEAM-10M', 'data/10M-0000%d-of-00002.parquet' % i, repo_type='dataset') for i in (0, 1)]
    pd.concat([pd.read_parquet(f) for f in fs]).to_parquet(SRC)
src = pd.read_parquet(SRC)
print('10M split rows:', len(src), 'columns:', src.columns.tolist())

def flat10m(chat):
    plans = []
    for d in chat:
        ks = [k for k in d if d[k] is not None]
        assert len(ks) == 1, ks
        plans.append((int(ks[0].split('-')[1]), d[ks[0]]))
    assert [n for n, _ in plans] == list(range(1, len(plans) + 1)), [n for n, _ in plans]
    msgs = [m for _, pl in plans for b in pl for t in b['turns'] for m in t]
    assert [int(m['id']) for m in msgs] == list(range(len(msgs))), 'message ids not 0..N-1 in stored order'
    return msgs

def s1(msgs):
    out = []
    for m in msgs:
        if True:
            meta = []
            if m.get('time_anchor'): meta.append(m['time_anchor'])
            if m.get('index'): meta.append('session,turn %s' % m['index'])
            tag = (' [' + ' | '.join(meta) + ']') if meta else ''
            out.append('%s.%s %s: %s' % (m['id'], tag, m['role'].upper(), m['content']))
    return '\n\n'.join(out)

CATS = ['abstention', 'contradiction_resolution', 'event_ordering', 'information_extraction', 'instruction_following',
        'knowledge_update', 'multi_session_reasoning', 'preference_following', 'summarization', 'temporal_reasoning']
rows = []
for _, r in src.iterrows():
    pqs = r['probing_questions']
    pqs = ast.literal_eval(pqs) if isinstance(pqs, str) else pqs
    assert sorted(pqs) == sorted(CATS), sorted(pqs)
    qs, ans, cats, idx, rub = [], [], [], [], []
    for cat in CATS:
        for i, q in enumerate(pqs[cat]):
            ideal = None
            for k in ('ideal_response', 'ideal_answer', 'answer', 'expected_compliance', 'ideal_summary'):
                if k in q: ideal = q[k]; break
            assert ideal is not None, (cat, i, list(q))
            qs.append(q['question']); ans.append([str(ideal)]); cats.append(cat); idx.append(i)
            rub.append([str(x) for x in q.get('rubric', [])])
    assert len(qs) == 20, len(qs)
    rows.append(dict(context=s1(flat10m(list(r['chat']))), questions=qs, answers=ans,
                     metadata=dict(source='beam_10m', conv_id=str(r['conversation_id']), question_types=cats, cat_index=idx, rubric=rub)))
df = pd.DataFrame(rows)
pq.write_table(pa.Table.from_pandas(df, preserve_index=False), 'beam/beam10m.parquet')
chk = pd.read_parquet('beam/beam10m.parquet')
print('wrote beam/beam10m.parquet rows=%d questions=%d context chars mean=%.0f (100K was 527K chars/row)'
      % (len(chk), sum(len(q) for q in chk['questions']), chk['context'].str.len().mean()))
