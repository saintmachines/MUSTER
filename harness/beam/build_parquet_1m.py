"""BEAM 1M -> a MemoryAgentBench-shaped parquet the harness can read (same serialization S1 as beam/build_parquet.py).

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

SRC = 'beam/1M.parquet'
if not os.path.exists(SRC):
    p = hf_hub_download('Mohammadta/BEAM', 'data/1M-00000-of-00001.parquet', repo_type='dataset')
    import shutil; shutil.copy(p, SRC)
src = pd.read_parquet(SRC)
print('1M split rows:', len(src), 'columns:', src.columns.tolist())

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
    rows.append(dict(context=s1(list(r['chat'])), questions=qs, answers=ans,
                     metadata=dict(source='beam_1m', conv_id=str(r['conversation_id']), question_types=cats, cat_index=idx, rubric=rub)))
df = pd.DataFrame(rows)
pq.write_table(pa.Table.from_pandas(df, preserve_index=False), 'beam/beam1m.parquet')
chk = pd.read_parquet('beam/beam1m.parquet')
print('wrote beam/beam1m.parquet rows=%d questions=%d context chars mean=%.0f (100K was 527K chars/row)'
      % (len(chk), sum(len(q) for q in chk['questions']), chk['context'].str.len().mean()))
