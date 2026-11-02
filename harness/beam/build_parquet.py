"""BEAM 100K -> a MemoryAgentBench-shaped parquet the harness can read.

context  : serialization S1. Every field the parquet carries, one line per message, nothing added,
           nothing dropped: "<id>. [<time_anchor> | session,turn <index>] <ROLE>: <content>".
questions: the 400 probing questions, in probing_questions' own (category, index) order.
answers  : [[ideal]] per question; abstention uses ideal_response, the rest ideal_answer.
metadata : source="beam_100k", question_types=[category...], cat_index=[...], rubric=[[...]...],
           conv_id. cat_index lets the answers file be rebuilt in the exact order the BEAM judge expects.
"""
import os, sys, ast, json
os.chdir(os.environ.get('MUSTER_HOME', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq

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

src = pd.read_parquet('beam/100K.parquet')
rows = []
CATS = ['abstention','contradiction_resolution','event_ordering','information_extraction',
        'instruction_following','knowledge_update','multi_session_reasoning','preference_following',
        'summarization','temporal_reasoning']
for _, r in src.iterrows():
    pqs = ast.literal_eval(r['probing_questions'])
    assert sorted(pqs) == sorted(CATS), sorted(pqs)
    qs, ans, cats, idx, rub = [], [], [], [], []
    for cat in CATS:                       # fixed category order, index order inside
        for i, q in enumerate(pqs[cat]):
            ideal = None
            for k in ('ideal_response', 'ideal_answer', 'answer', 'expected_compliance', 'ideal_summary'):
                if k in q: ideal = q[k]; break
            assert ideal is not None, (cat, i, list(q))
            qs.append(q['question']); ans.append([str(ideal)]); cats.append(cat); idx.append(i)
            rub.append([str(x) for x in q.get('rubric', [])])
    assert len(qs) == 20, len(qs)
    rows.append(dict(context=s1(list(r['chat'])), questions=qs, answers=ans,
                     metadata=dict(source='beam_100k', conv_id=str(r['conversation_id']),
                                   question_types=cats, cat_index=idx, rubric=rub)))
df = pd.DataFrame(rows)
tbl = pa.Table.from_pandas(df, preserve_index=False)
pq.write_table(tbl, 'beam/beam100k.parquet')
chk = pd.read_parquet('beam/beam100k.parquet')
print('wrote beam/beam100k.parquet  rows=%d  questions=%d  context chars mean=%.0f'
      % (len(chk), sum(len(q) for q in chk['questions']), chk['context'].str.len().mean()))
m = dict(chk.iloc[0]['metadata']); print('metadata keys:', list(m)); print('source:', m['source'], '| qt[0..3]:', list(m['question_types'][:3]))
print('context head:', chk.iloc[0]['context'][:160].replace('\n', '\\n'))
