"""Explicit serial codecs: actual SDK/CLI/HTTP, with CPU fake models only."""
import dataclasses,hashlib,json,threading,urllib.error,urllib.request
import pytest
import muster
from muster.serve import Service,http_server

STYLES=['Step %d: %s','#%d %s','(%d) %s','[%d] %s','%d. %s','[step %d] %s']
BODIES=['Ann moved to Paris.','Ann moved to Seoul.','Ben likes green tea.']

def declaration():
    return muster.SourceOrder(hashlib.sha256(b'serial codec CPU fixture').hexdigest(),'fixture.json#/records',
        'structured_array','The producer declares this array in append order.','neutral-codec-fixture')

def envelope(records):return dict(records=records,order=dataclasses.asdict(declaration()),format='serial')

def build(records,**kw):
    return muster.build(records,declared_order=declaration(),record_format='serial',models=muster.Models.fake(16),**kw)

@pytest.mark.parametrize('fmt',STYLES)
def test_serial_style_reaches_search_render_and_exact_roundtrip(fmt):
    original=[fmt%(i,b) for i,b in enumerate(BODIES)]
    mem=build(original);ref=muster.build(BODIES,declared_order=declaration(),models=muster.Models.fake(16))
    assert mem.units==tuple(BODIES) and mem.plan.source=='declared-order:record-boundaries-v1'
    assert mem.plan.supersede=='off' and not mem.index.structural and not mem.index.newest_row
    assert mem.cut.order==(0,1,2) and len(set(mem.parent))==3
    assert mem.context('Where did Ann move?',U=3).text==ref.context('Where did Ann move?',U=3).text
    assert mem.ordered_records.roundtrip()==original
    assert all(r.prefix+r.body==r.raw_text==raw for r,raw in zip(mem.ordered_records.records,original))

def test_marker_gaps_resets_duplicates_do_not_reorder_or_prune():
    raw=['Step 9: Ann moved to Paris.','#1 Ann moved to Seoul.','(1) Ann moved to Boston.']
    mem=build(raw);diag=mem.ordered_records.diagnostics();text=mem.context('Where did Ann move?',U=3).text
    assert mem.cut.order==(0,1,2) and diag['marker_values']==[9,1,1]
    assert not diag['contiguous_markers'] and not diag['marker_is_revision_clock']
    assert diag['deleted_record_ids']==[] and diag['policy']=='keep_all' and not diag['supersession_eligible']
    assert text.index('Paris')<text.index('Seoul')<text.index('Boston')

def test_codec_acceptance_has_no_95_percent_or_20_token_gate():
    body='A long declared record with repeated context. '*80
    mem=build(['Step 30: '+body,'Step 4: '+body,'#4 '+body])
    assert mem.units==(body,body,body) and len(mem.parent)==3 and not mem.index.structural
    assert not mem.ordered_records.diagnostics()['contiguous_markers']
    assert build(['Step 3: one declared record']).units==('one declared record',)

def test_multiline_unicode_metadata_roundtrip_preserves_received_values():
    body='Ann: 첫 줄\r\n\r\n2. body line\n<|endoftext|> 끝\t  '
    raw=[dict(id='origin',content='\tStep ٢:\t'+body,time='2026-11-04',speaker='Ann',nested={'opaque':[1,'값']})]
    mem=build(raw);rec=mem.ordered_records.records[0]
    assert rec.marker==2 and rec.body==body and rec.prefix=='\tStep ٢:\t'
    assert rec.prefix+rec.body==raw[0]['content'] and mem.ordered_records.roundtrip()==raw
    assert mem.units==(body,) and mem.unit_records(0)[0].text==body

def test_unicode_digit_and_32_codepoint_lexical_boundary():
    assert build(['Step 𝟙: body']).ordered_records.records[0].marker==1
    accepted='X'*30+' 1: body';rejected='X'*31+' 1: body'
    assert build([accepted]).ordered_records.roundtrip()==[accepted]
    with pytest.raises(muster.RecordError,match='prefix'):build([rejected])

@pytest.mark.parametrize('bad',['timestamp 2026-11-04 body','Step 123456789: body','Step -1: body','Étape 1: body','D1:2 body','Step 1:   '])
def test_invalid_serial_is_not_silently_reinterpreted(bad):
    with pytest.raises(muster.RecordError):build([bad])

def test_existing_strict_codecs_and_provenance_refusals_remain():
    with pytest.raises(muster.RecordError,match='prefix'):
        muster.build(['Step 1: body'],declared_order=declaration(),record_format='step',embed=False)
    with pytest.raises(muster.RecordError,match='prefix'):
        muster.build(['#1 body'],declared_order=declaration(),record_format='integer_dot',embed=False)
    with pytest.raises(muster.PlanError):muster.build(['#1 body'],record_format='serial',embed=False)
    with pytest.raises(muster.RecordError):build('Step 1: body')
    with pytest.raises(muster.PlanError,match='exact'):
        muster.build(['#1 body'],'beam.u20',declared_order=declaration(),record_format='serial',embed=False)
    with pytest.raises(muster.RecordError,match='duplicate'):
        build([dict(id='x',content='#1 one'),dict(id='x',content='#2 two')])

def test_cli_serial_envelope_uses_actual_declared_builder(tmp_path,capsys):
    from muster.cli import main
    path=tmp_path/'serial.json';path.write_text(json.dumps(envelope(['Step 8: Ann keeps a blue key.','#1 Ben keeps a red key.'])))
    main(['query',str(path),'Which keys?','--declared-records','--units','2','--fake-models','--explain'])
    captured=capsys.readouterr();assert captured.out.index('blue key')<captured.out.index('red key')
    assert 'Step 8:' not in captured.out and '#1' not in captured.out and 'all versions kept' in captured.err

def test_http_serial_envelope_requires_optin_and_does_not_write_store(tmp_path):
    with muster.Store.open(str(tmp_path/'memory.muster'),models=muster.Models.fake(16)) as store:
        service=Service(store);server=http_server(service,'127.0.0.1',0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url='http://127.0.0.1:%d/v1/declared-context'%server.server_address[1]
        payload=dict(store=envelope([dict(id='first',content='Step 8: Ann in Paris.'),dict(id='last',content='#1 Ann in Seoul.')]),query='Where is Ann?',units=2)
        def post():
            request=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            return json.loads(urllib.request.urlopen(request,timeout=10).read())
        try:
            with pytest.raises(urllib.error.HTTPError) as exc:post()
            assert exc.value.code==400
            service.allow_declared_records=True;out=post()
            assert out['stateless'] and out['record_order']['record_format']=='serial'
            assert out['record_order']['record_ids']==['first','last'] and out['record_order']['marker_values']==[8,1]
            assert out['text'].index('Paris')<out['text'].index('Seoul') and store.version==0
            assert not out['record_order']['supersession_eligible'] and out['record_order']['deleted_record_ids']==[]
        finally:server.shutdown();server.server_close();thread.join()
