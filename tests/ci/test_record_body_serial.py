"""A: anchored record body, preserved metadata, actual native selection/assembly."""
import dataclasses
import json
import pytest
import muster
from muster.index import btok,slotkey
from muster.records import as_record,serialize

STYLES=['Step %d: %s','#%d %s','(%d) %s','[%d] %s','%d. %s','[step %d] %s']

def body(i):
    # Every eighth record revises the one before it: the update evidence a numbered plan needs.
    j=i-1 if i%8==7 else i
    name='Entity'+chr(65+j//26)+chr(65+j%26)
    return name+(' is registered in Porto.' if i%8==7 else ' is registered in Lisbon.')

def slot(i,fmt):
    # Spelled here rather than computed by the code under test. A dot serial takes the native subject-relation slot
    # (exact.md N8), subject and relation without the value; a log's own spelling keeps the paper's slot, names masked.
    j=i-1 if i%8==7 else i
    e='entity'+chr(97+j//26)+chr(97+j%26)
    return 's:v:'+e+'||is registered in' if fmt=='%d. %s' else 's:'+e+'|_ is registered in _'

def kept(n):
    # The rows a numbered read shows: a revised record gives way to its revision.
    return [i for i in range(n) if not (i%8==6 and i+1<n)]

def records(fmt='Step %d: %s',n=32):
    return [dict(id=str(1000+i),time='2026-02-%02d'%(1+i%28),speaker='UniqueSpeaker%d'%i,
                 text=fmt%(i,body(i)),session='session',slot='metadata-not-a-slot',order=999999-i,
                 meta={'opaque':'value%d'%i}) for i in range(n)]

@pytest.mark.parametrize('fmt',STYLES)
def test_metadata_numbered_bodies_select_assemble_and_map_original_records(fmt):
    original=records(fmt);mem=muster.build(original,models=muster.Models.fake(32))
    assert mem.plan.branch=='numbered' and mem.plan.policy=='attprio_sess'
    assert mem.cut.order==tuple(range(32)) and mem.parent==tuple(slot(i,fmt) for i in range(32))
    assert mem.index.r_slot==[slotkey(body(i)) for i in range(32)]
    normalized=[as_record(r) for r in original];text,spans=serialize(normalized)
    assert mem.text==text and mem.cut.spans==tuple(spans)
    assert [dataclasses.asdict(r) for r in mem.records]==[dataclasses.asdict(r) for r in normalized]
    for i in range(32):
        assert mem.units[i]==text[spans[i][0]:spans[i][1]]
        assert mem.cut.keys[i]==mem.units[i] and mem.unit_records(i)==[mem.records[i]]
    scores=mem.index.bm25.get_scores(btok('UniqueSpeaker7'))
    assert scores[7]>max(v for i,v in enumerate(scores) if i!=7)
    assert '2026-02-08' in mem.cut.keys[7] and 'UniqueSpeaker7' in mem.cut.keys[7]
    selection=mem.search('What did UniqueSpeaker7 record on 2026-02-08?',U=32);ctx=mem.render(selection)
    assert set(ctx.ids)==set(kept(32)) and 'UniqueSpeaker7' in ctx.text and '2026-02-08' in ctx.text
    assert ctx.text=='\n'.join(mem.units[i] for i in ctx.ids)
    assert muster.Plan.from_dict(json.loads(mem.plan.to_json())).plan_id==mem.plan.plan_id

@pytest.mark.parametrize('field,style',[('id','%d'),('id','%d.'),('time','%d'),('time','%d.'),('speaker','Step %d')])
def test_metadata_serials_do_not_turn_plain_bodies_into_numbered(field,style):
    raw=[dict(text='An ordinary short message.',**{field:style%i}) for i in range(1,61)]
    plan=muster.derive(raw);mem=muster.build(raw,embed=False)
    assert plan.branch!='numbered' and mem.plan.branch!='numbered' and not mem.index.structural
    assert mem.records[0].text=='An ordinary short message.'

def test_literal_body_role_header_is_not_removed_or_guessed():
    raw=[dict(id=str(i),speaker='actual-speaker',time=str(i),text='user: Step %d: '%i+body(i)) for i in range(40)]
    mem=muster.build(raw,embed=False)
    assert mem.plan.branch!='numbered'
    assert all('user: Step %d: '%i+body(i) in mem.text for i in range(40))
    assert [r.text for r in mem.records]==[r['text'] for r in raw]

def test_integer_ids_are_metadata_and_ambiguous_numeric_time_remains_refused():
    raw=[dict(id=i,text='An ordinary short message.') for i in range(60)]
    assert muster.derive(raw).branch!='numbered'
    mem=muster.build(raw,embed=False);assert [r.id for r in mem.records]==[str(i) for i in range(60)]
    with pytest.raises(muster.RecordError,match='ambiguous'):
        muster.build([dict(time=i,text='An ordinary short message.') for i in range(60)],embed=False)

def test_multiline_body_is_one_record_with_exact_metadata_and_spans():
    raw=records(n=24)
    for r in raw:r['text']+='\r\nNote: 두 번째 줄.'
    mem=muster.build(raw,models=muster.Models.fake(16))
    assert mem.plan.branch=='numbered' and len(mem)==24
    for i,r in enumerate(raw):
        assert mem.unit_records(i)[0].text==r['text']
        assert '\nNote: 두 번째 줄.' in mem.units[i] and '\r' not in mem.units[i]
    kept=muster.build(raw,models=muster.Models.fake(16),normalize_newlines=False)
    assert all('\r\nNote:' in u for u in kept.units)

@pytest.mark.parametrize('fmt',STYLES)
def test_metadata_free_numbered_record_spelling_matches_plain_numbered(fmt):
    raw=[fmt%(i,body(i)) for i in range(24)]
    rec=muster.build(raw,embed=False);plain=muster.build('\n'.join(raw),embed=False)
    assert rec.plan.branch==plain.plan.branch=='numbered'
    assert rec.cut.units==plain.cut.units and rec.parent==plain.parent and rec.cut.order==plain.cut.order
    assert rec.render([3,1]).text==plain.render([3,1]).text

def test_one_unmarked_empty_of_120_never_becomes_a_revision_clock():
    raw=records(n=120);raw[60]['text']=''
    mem=muster.build(raw,models=muster.Models.fake(16))
    assert mem.plan.branch=='numbered' and len(mem)==120
    assert mem.cut.order[60]==-1 and mem.parent[60]=='u:record60'
    assert len(set(mem.parent))<120 and len(set(mem.cut.order))>=.99*120
    assert not mem.index.structural and not mem.index.newest_row
    assert mem.unit_records(60)==[mem.records[60]] and mem.records[60].id=='1060'
    ctx=mem.render(list(reversed(range(120))))
    assert ctx.ids==tuple(range(120)) and ctx.text.index('1059')<ctx.text.index('1060')<ctx.text.index('1061')
    # A span with no bytes still refers to its original record, never its neighbour.
    raw[60]={'text':''};empty=muster.build(raw,embed=False)
    assert empty.cut.spans[60][0]==empty.cut.spans[60][1]
    assert empty.unit_records(60)==[empty.records[60]] and empty.records[60].text==''

@pytest.mark.parametrize('missing',[0,60,119])
def test_unmarked_literal_record_is_kept_as_written(missing):
    raw=records(n=120);raw[missing]['text']='This unmarked note is a literal original record.'
    mem=muster.build(raw,models=muster.Models.fake(16))
    assert mem.plan.branch=='numbered' and mem.cut.order[missing]==-1 and not mem.index.structural
    assert raw[missing]['text'] in mem.units[missing] and mem.unit_records(missing)[0].text==raw[missing]['text']
    ctx=mem.context('literal original record',U=120)
    assert ctx.ids==tuple(range(120)) and raw[missing]['text'] in ctx.text
    assert ctx.text=='\n'.join(mem.units[i] for i in range(120))
    assert ''.join(b.text for b in ctx.blocks).count('Step ')==119

def test_all_marked_dot_records_keep_existing_numeric_clock_render_order():
    raw=list(reversed(records('%d. %s',n=24)));mem=muster.build(raw,models=muster.Models.fake(16))
    assert mem.cut.order==tuple(reversed(range(24)))
    ctx=mem.context('registered Lisbon',U=24)
    assert ctx.ids==tuple(23-i for i in kept(24))
    assert '1000 [2026-02-01] UniqueSpeaker0: 0.' in ctx.blocks[0].text

@pytest.mark.parametrize('marked,branch',[(85,'numbered'),(80,'numbered'),(79,'record')])
def test_dot_decision_cut_boundary_preserves_all_original_records(marked,branch):
    from muster.cut import AS_MINREC
    from muster.derive import NUMBERED_TAU
    assert AS_MINREC==NUMBERED_TAU==.8
    raw=records('%d. %s',n=100)
    for i in range(marked,100):raw[i]['text']='An unmarked original note at position %d.'%i
    mem=muster.build(raw,models=muster.Models.fake(16))
    assert mem.plan.branch==branch and len(mem.records)==100
    if branch=='numbered':
        assert len(mem)==100 and mem.cut.order[marked:]==(-1,)*(100-marked)
        assert len(set(mem.parent[marked:]))==100-marked and not mem.index.structural
    context=mem.context('original note registered',U=100)
    mapped=[r.id for block in context.blocks for i in block.ids for r in mem.unit_records(i)]
    assert mapped==[str(1000+i) for i in range(100)]
    assert all(r['text'] in context.text for r in raw)

def test_empty_record_array_and_empty_metadata_bodies_are_safe():
    assert muster.build([],embed=False).plan.branch=='empty'
    raw=[dict(id=str(i),speaker='user',time=str(i),text='') for i in range(40)]
    mem=muster.build(raw,embed=False)
    assert mem.plan.branch!='numbered' and [r.id for r in mem.records]==[str(i) for i in range(40)]

def test_store_add_fresh_rebuild_reopen_and_retry_are_identical(tmp_path):
    raw=records(n=36);path=str(tmp_path/'body.muster');model=muster.Models.fake(32)
    with muster.Store.open(path,models=model) as store:
        for i in range(0,36,6):store.add(raw[i:i+6],request_id='batch%d'%i)
        before=store.version;assert store.add(raw[-6:],request_id='batch30').duplicate and store.version==before
        head=store.head();fresh=muster.build(store.records(),models=model)
        assert head.plan==fresh.plan and head.units==fresh.units and head.parent==fresh.parent
        assert head.cut.order==fresh.cut.order and head.index.r_slot==fresh.index.r_slot
        expected=store.context('UniqueSpeaker8',U=36).text
        assert expected==fresh.context('UniqueSpeaker8',U=36).text
        ids=[r.id for r in store.records()]
    with muster.Store.open(path,models=model) as store:
        assert store.head().plan.branch=='numbered' and [r.id for r in store.records()]==ids
        assert store.context('UniqueSpeaker8',U=36).text==expected

def test_custom_template_display_is_honored_without_becoming_a_clock():
    raw=records(n=24);template='{speaker} [{time}] {id}: {text}'
    mem=muster.build(raw,models=muster.Models.fake(16),template=template)
    assert mem.plan.branch=='numbered' and mem.cut.order==tuple(range(24))
    assert mem.units[0].startswith('UniqueSpeaker0 [2026-02-01] 1000: Step 0:')
    assert mem.index.r_slot[0]==slotkey(body(0)) and mem.unit_records(0)[0].id=='1000'
