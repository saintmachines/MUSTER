"""Declared input contracts through SDK, CLI and an actual HTTP transport."""
import dataclasses
import hashlib
import json
import threading
import urllib.error
import urllib.request

import pytest
import muster
from muster.index import slotkey
from muster.serve import Service, http_server


def declaration(**kw):
    fields = dict(source_sha256=hashlib.sha256(b"fixture source").hexdigest(),
                  source_locator="fixture.json#/records", kind="structured_array",
                  declaration="The producer declares this array in append order.", namespace="fixture")
    fields.update(kw)
    return muster.SourceOrder(**fields)


def envelope(records, fmt="content"):
    return {"records": records, "order": dataclasses.asdict(declaration()), "format": fmt}


def memory(records, **kw):
    return muster.build(records, declared_order=declaration(), models=muster.Models.fake(16), **kw)


def test_equivalent_declared_formats_reach_real_search_and_render():
    bodies = ["Ann moved to Paris.", "Ann moved to Seoul.", "Ben likes green tea."]
    memories = [memory([{"content": b} for b in bodies]),
                memory(["[step %d] %s" % (i, b) for i, b in enumerate(bodies)], record_format="step"),
                memory(["%d. %s" % (i, b) for i, b in enumerate(bodies)], record_format="integer_dot")]
    contexts = []
    for mem in memories:
        assert mem.units == tuple(bodies)
        assert mem.plan.supersede == "off" and not mem.index.structural and not mem.index.newest_row
        assert mem.index.r_slot == [slotkey(b) for b in bodies]
        sel = mem.search("Where did Ann move?", U=3)
        assert set(sel.ids) == {0, 1, 2}
        ctx = mem.render(sel)
        assert "Paris" in ctx.text and "Seoul" in ctx.text
        assert ctx.text.index("Paris") < ctx.text.index("Seoul")
        assert [r.text for r in mem.unit_records(1)] == [bodies[1]]
        contexts.append(ctx.text)
    assert len(set(contexts)) == 1


def test_ambiguous_dates_never_reorder_records():
    records = [{"id": "later", "content": "03/04/25 Ann moved to Paris."},
               {"id": "earlier", "content": "04/03/25 Ann moved to Seoul."}]
    mem = memory(records)
    assert [r.id for r in mem.ordered_records.records] == ["later", "earlier"]
    assert mem.units == tuple(r["content"] for r in records)
    assert mem.ordered_records.roundtrip() == records


def test_serial_reset_is_diagnostic_without_revision_semantics():
    mem = memory(["1. Ann likes tea.", "2. Ann likes coffee.", "1. Ann likes milk."], record_format="integer_dot")
    diag = mem.ordered_records.diagnostics()
    assert not diag["contiguous_markers"] and not diag["marker_is_revision_clock"]
    assert len(mem.units) == 3 and not mem.index.structural


def test_duplicate_record_id_is_refused():
    with pytest.raises(muster.RecordError, match="duplicate"):
        memory([{"id": "a", "content": "one"}, {"id": "a", "content": "two"}])


def test_gaps_do_not_infer_missing_records_or_deletions():
    mem = memory(["[step 1] first", "[step 7] seventh"], record_format="step")
    assert mem.units == ("first", "seventh")
    assert not mem.ordered_records.diagnostics()["contiguous_markers"]
    assert mem.ordered_records.diagnostics()["deleted_record_ids"] == []


def test_multiline_crlf_unicode_and_special_text_roundtrip():
    text = "[step 2] Ann: 첫 줄\r\n\r\n2. body line\n<|endoftext|> 끝"
    mem = memory([text], record_format="step")
    rec = mem.ordered_records.records[0]
    assert rec.body == text[len("[step 2] "):]
    assert rec.prefix + rec.body == rec.raw_text == text
    assert mem.ordered_records.roundtrip() == [text]
    assert mem.units == (rec.body,)


def test_delete_request_and_state_update_are_kept_distinct_and_not_applied():
    records = [{"content": "Ann likes tea."}, {"content": "Ann now likes coffee."},
               {"content": "Please forget that Ann likes tea."}]
    mem = memory(records)
    assert len(mem.units) == 3 and not mem.index.newest_row
    assert all(r["content"] in mem.context("What are Ann's preferences?", U=3).text for r in records)


def test_undeclared_plain_blob_and_bad_prefix_are_refused():
    with pytest.raises(muster.RecordError, match="record array"):
        memory("1. first\n2. second")
    with pytest.raises(muster.RecordError):
        muster.build(["body"], declared_order={"kind": "structured_array"}, embed=False)
    with pytest.raises(muster.RecordError, match="prefix"):
        memory(["1. body"], record_format="step")
    with pytest.raises(muster.PlanError):
        muster.build(["1. body"], record_format="integer_dot", embed=False)


def test_explicit_metadata_is_not_a_slot_or_revision_contract():
    records = [{"content": "Ann lives in Paris.", "slot": "location", "order": 9},
               {"content": "Ann lives in Seoul.", "slot": "location", "order": 1}]
    mem = memory(records)
    assert mem.ordered_records.roundtrip() == records
    assert mem.cut.order == (0, 1) and len(set(mem.parent)) == 2
    assert not mem.index.structural and mem.plan.supersede == "off"


def test_exact_replay_cannot_be_changed_by_new_option():
    with pytest.raises(muster.PlanError, match="exact"):
        muster.build(["body"], "beam.u20", declared_order=declaration(), embed=False)


def test_custom_template_cannot_silently_hide_reader_metadata():
    records = [{"content": "Ann moved.", "time": "2026-01-02", "speaker": "Ann"}]
    with pytest.raises(muster.PlanError, match="custom template"):
        memory(records, template="{time} {speaker}: {text}")
    mem = memory(records)
    assert mem.ordered_records.roundtrip() == records
    assert mem.units == ("Ann moved.",)


def test_cli_declared_records_uses_real_builder(tmp_path, capsys):
    from muster.cli import main
    p = tmp_path / "records.json"
    p.write_text(json.dumps(envelope(["[step 9] Ann keeps a blue key.", "[step 2] Ben keeps a red key."], "step")))
    main(["query", str(p), "Which keys?", "--declared-records", "--units", "2", "--fake-models", "--explain"])
    captured = capsys.readouterr()
    out = captured.out
    assert out.index("blue key") < out.index("red key")
    assert "[step " not in out
    assert "one per declared record" in captured.err and "all versions kept" in captured.err


def test_malformed_utf8_and_conflicting_content_are_refused():
    with pytest.raises(muster.RecordError, match="UTF-8"):
        memory(["invalid surrogate \ud800"])
    with pytest.raises(muster.RecordError, match="disagree"):
        memory([{"text": "a", "content": "b"}])


def test_http_endpoint_requires_optin_and_preserves_store(tmp_path):
    def post(base, obj):
        req = urllib.request.Request(base + "/v1/declared-context", data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=10).read())
    with muster.Store.open(str(tmp_path / "memory.muster"), models=muster.Models.fake(16)) as store:
        service = Service(store)
        server = http_server(service, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = "http://127.0.0.1:%d" % server.server_address[1]
        payload = {"store": envelope([{"id": "first", "content": "Ann in Paris."},
                                      {"id": "last", "content": "Ann in Seoul."}]),
                   "query": "Where is Ann?", "units": 2}
        try:
            with pytest.raises(urllib.error.HTTPError) as exc:
                post(base, payload)
            assert exc.value.code == 400
            service.allow_declared_records = True
            out = post(base, payload)
            assert out["stateless"] and out["record_order"]["record_ids"] == ["first", "last"]
            assert out["text"].index("Paris") < out["text"].index("Seoul")
            assert [r["id"] for b in out["blocks"] for r in b["records"]] == ["first", "last"]
            assert store.version == 0
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
