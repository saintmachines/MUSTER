"""What an application can really send: a store with no words, half an emoji, chat messages with parts, a bad request.
Each works or gets one sentence; none stops the process or leaves a store that cannot be read."""
import http.client
import json
import threading
from datetime import datetime

import pytest

import muster
from muster.cli import main
from muster.records import as_record
from muster.serve import Service, http_server

FAKE = muster.Models.fake(16)
HALF = "half \ud83d emoji"


def stored(*batches):
    store = muster.Store.open(":memory:", models=FAKE)
    for b in batches:
        store.add(b)
    return store


@pytest.mark.parametrize("text", ["\U0001F44D", "?", "... !!!", "  .  ", "\U0001F44D\n\n\U0001F642"])
def test_a_store_without_words_can_be_read(text):
    assert muster.build(text, models=FAKE).context("How did they react?").text.strip()
    with stored([text]) as store:
        assert store.context("How did they react?").text.strip()


def test_a_store_with_a_few_words_reads_as_before():
    mem = muster.build(["\U0001F44D", "Alice lives in Rome.", "\U0001F642"], models=FAKE)
    assert "Alice lives in Rome." in mem.context("Where does Alice live?", U=3).text


@pytest.mark.parametrize("call", [
    lambda s: muster.Record(s),
    lambda s: muster.Record("ok", speaker=s),
    lambda s: as_record({"text": "ok", "time": s}),
    lambda s: muster.derive(s),
    lambda s: muster.build(s, models=FAKE),
    lambda s: muster.build([s], models=FAKE),
])
def test_half_an_emoji_is_refused_where_it_enters(call):
    with pytest.raises(muster.RecordError, match="UTF-8"):
        call(HALF)


def test_half_an_emoji_leaves_a_store_as_it_was():
    with stored(["Alice lives in Rome."]) as store:
        for bad in ([HALF], [{"text": "ok", "topic": HALF}], [{"text": "ok", "meta": {"k": HALF}}]):
            with pytest.raises(muster.RecordError, match="UTF-8"):
                store.add(["kept back"] + bad)
        with pytest.raises(muster.RecordError, match="UTF-8"):
            store.add(["kept back"], request_id=HALF)
        assert store.version == 1
        assert "Alice" in store.context("Where does Alice live?").text


def test_the_refusal_says_where_and_how_to_clean_the_text():
    with pytest.raises(muster.RecordError) as e:
        muster.Record(HALF)
    assert "position 5" in str(e.value) and "s.encode('utf-16', 'surrogatepass').decode('utf-16', 'replace')" in str(e.value)
    s = "half \ud83d and whole \ud83d\ude00"
    cleaned = s.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    assert cleaned == "half \ufffd and whole \U0001F600"
    assert muster.Record(cleaned).text == cleaned


@pytest.mark.parametrize("query", ["", "   ", None, 5, b"x", ["x"], "\ud83d"])
def test_a_query_the_library_cannot_read_is_one_error(query):
    mem = muster.build(["Alice lives in Rome."], models=FAKE)
    with pytest.raises(muster.QueryError) as e:
        mem.search(query)
    assert isinstance(e.value, muster.MusterError) and isinstance(e.value, ValueError)


def test_a_query_of_emoji_or_odd_characters_is_read():
    mem = muster.build(["Alice lives in Rome."], models=FAKE)
    for query in ("\U0001F44D", "?", "\x00", "<|endoftext|>", "{context}"):
        assert mem.context(query).text


@pytest.mark.parametrize("U", [-1, -100, "x", [1], float("inf")])
def test_a_budget_that_is_not_a_positive_number_is_refused(U):
    mem = muster.build(["Alice lives in Rome.", "Bob lives in Oslo."], models=FAKE)
    with pytest.raises(muster.PlanError, match="positive whole number"):
        mem.search("Where?", U=U)


def test_a_budget_reads_as_before():
    mem = muster.build(["%s lives far away. %s" % (n, "They walk every day. " * 60) for n in "Alice Bob Carol Dan Eve Finn".split()], models=FAKE)
    assert len(mem) >= 3
    default = mem.context("Where?").text
    assert mem.context("Where?", U=None).text == default and mem.context("Where?", U=0).text == default
    assert len(mem.search("Where?", U=1).ids) == 1
    assert len(mem.search("Where?", U=2).ids) == 2


def test_chat_parts_keep_their_text():
    rec = as_record({"role": "user", "content": [{"type": "text", "text": "Alice lives in Rome."}, {"type": "text", "text": "Bob too."}]})
    assert rec.text == "Alice lives in Rome.\nBob too." and rec.speaker == "user"
    assert as_record({"content": ["a", {"text": "b"}]}).text == "a\nb"
    with stored([{"role": "user", "content": [{"type": "text", "text": "Alice lives in Rome."}]}]) as store:
        shown = store.context("Where does Alice live?").text
    assert "Alice lives in Rome." in shown and "{'type'" not in shown


@pytest.mark.parametrize("content", [
    [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "data:,"}}],
    [{"type": "tool_use", "id": "t", "name": "f", "input": {}}],
    [{"type": "text", "text": 5}],
    [5],
    b"bytes",
    {"a": 1},
])
def test_other_content_is_refused_not_printed(content):
    with pytest.raises(muster.RecordError):
        as_record({"content": content})


def test_a_number_is_still_read_as_text():
    assert as_record({"text": 7}).text == "7" and as_record({"text": 1.5}).text == "1.5"


@pytest.mark.parametrize("meta", [5, "abc", ["x"]])
def test_meta_must_be_a_mapping(meta):
    with pytest.raises(muster.RecordError, match="meta"):
        as_record({"text": "a", "meta": meta})
    with pytest.raises(muster.RecordError, match="meta"):
        muster.Record("a", meta=meta)


def test_meta_the_store_cannot_save_is_refused_before_anything_is_written():
    with stored() as store:
        with pytest.raises(muster.RecordError, match="stored"):
            store.add(["kept back", {"text": "a", "meta": {"k": datetime(2026, 5, 2)}}])
        assert store.version == 0
        store.add([{"text": "a", "meta": {"k": "v", "n": 5}}])
        assert store.records()[0].meta == {"k": "v", "n": 5}


@pytest.mark.parametrize("one", [{"text": "Alice lives in Rome."}, muster.Record("Alice lives in Rome.")])
def test_one_record_is_a_list_of_one(one):
    assert muster.build(one, models=FAKE).text == muster.build([one], models=FAKE).text == "Alice lives in Rome."
    assert muster.derive(one).plan_id == muster.derive([one]).plan_id
    with stored(one) as store:
        assert store.version == 1


@pytest.mark.parametrize("bad", [7, 1.5, None, {"a", "b"}, frozenset({"a"})])
def test_a_store_that_is_not_an_ordered_list_is_refused(bad):
    with pytest.raises(muster.RecordError, match="records must be a list"):
        muster.derive(bad)
    with pytest.raises(muster.RecordError, match="records must be a list"):
        muster.build(bad, models=FAKE)
    with stored() as store:
        with pytest.raises(muster.RecordError, match="records must be a list"):
            store.add(bad)


def test_an_iterator_of_records_is_read_once():
    assert muster.build((x for x in ["Alice lives in Rome."]), models=FAKE).text == "Alice lives in Rome."
    assert muster.build(("Alice lives in Rome.",), models=FAKE).text == "Alice lives in Rome."


def test_an_unknown_preset_is_a_plan_error():
    with pytest.raises(muster.PlanError, match="no preset or leg named"):
        muster.build("x", "no.such.preset", models=FAKE)


@pytest.fixture
def server():
    with stored(["Alice lives in Rome."]) as store:
        srv = http_server(Service(store), port=0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        yield srv.server_address[1]
        srv.shutdown()
        srv.server_close()


@pytest.mark.parametrize("path,body,code", [
    ("/v1/add", b'{"text": "Bob lives in Oslo."}', 200),
    ("/v1/context", b'{"query": "Where does Alice live?"}', 200),
    ("/v1/search", b'{"query": "Where does Alice live?", "units": 1}', 200),
    ("/v1/add", b"[]", 400),
    ("/v1/add", b'"x"', 400),
    ("/v1/add", b"5", 400),
    ("/v1/add", b"null", 400),
    ("/v1/add", b"{bad", 400),
    ("/v1/add", b"\xff\xfe", 400),
    ("/v1/add", b'{"records": 5}', 400),
    ("/v1/add", b'{"records": ["half \\ud83d"]}', 400),
    ("/v1/context", b'{"query": 5}', 400),
    ("/v1/context", b'{"query": ["x"]}', 400),
    ("/v1/context", b'{"query": null}', 400),
    ("/v1/context", b'{"query": ""}', 400),
    ("/v1/context", b'{"query": "\\ud83d"}', 400),
    ("/v1/context", b'{"query": "x", "units": -1}', 400),
    ("/v1/context", b"{}", 400),
    ("/v1/search", b'{"query": 5}', 400),
    ("/v1/search", b"[]", 400),
    ("/nosuch", b"{}", 404),
])
def test_every_request_is_answered(server, path, body, code):
    c = http.client.HTTPConnection("127.0.0.1", server, timeout=10)
    c.request("POST", path, body, {"Content-Type": "application/json"})
    r = c.getresponse()
    answer = json.loads(r.read())
    c.close()
    assert r.status == code, answer
    if code != 200:
        assert answer["error"]


@pytest.mark.parametrize("length,code", [("-1", 400), ("x", 400), (str(1 << 40), 413)])
def test_a_bad_length_is_answered_at_once(server, length, code):
    c = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
    c.putrequest("POST", "/v1/add")
    c.putheader("Content-Length", length)
    c.endheaders()
    r = c.getresponse()
    assert r.status == code
    c.close()


def test_a_fault_of_the_machine_is_answered_too(server, monkeypatch):
    def broken(self, *a, **k):
        raise muster.AssetMissing("run muster fetch")
    monkeypatch.setattr(Service, "context", broken)
    c = http.client.HTTPConnection("127.0.0.1", server, timeout=10)
    c.request("POST", "/v1/context", b'{"query": "x"}')
    r = c.getresponse()
    assert r.status == 500 and "run muster fetch" in json.loads(r.read())["error"]
    c.close()


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


@pytest.mark.parametrize("name,data,argv,says", [
    ("a.txt", b"plain words\n", ["add", "--store", "{dir}/s.muster", "{file}", "--fake-models"], "not JSON"),
    ("a.jsonl", b'{"text": "a"}\n{bad\n', ["add", "--store", "{dir}/s.muster", "{file}", "--fake-models"], "line 2"),
    ("a.jsonl", b"caf\xe9\n", ["add", "--store", "{dir}/s.muster", "{file}", "--fake-models"], "not UTF-8"),
    ("a.jsonl", b'{"text": "half \\ud83d"}\n', ["add", "--store", "{dir}/s.muster", "{file}", "--fake-models"], "UTF-8"),
    ("a.txt", b"", ["add", "--store", "{dir}/s.muster", "{dir}/none.jsonl", "--fake-models"], "no such file"),
    ("a.txt", b"caf\xe9\n", ["query", "{file}", "q", "--fake-models"], "not UTF-8"),
    ("a.txt", b"Alice lives in Rome.", ["query", "{file}", "", "--fake-models"], "empty query"),
    ("a.txt", b"Alice lives in Rome.", ["query", "{file}", "q", "--units", "-1", "--fake-models"], "positive whole number"),
    ("a.jsonl", b"5\n", ["query", "{file}", "q", "--fake-models"], "cannot read a record"),
    ("a.jsonl", b"{bad\n", ["query", "{file}", "q", "--fake-models"], "not JSON"),
    ("a.txt", b"", ["query", "{dir}/none.txt", "q", "--fake-models"], "no such file"),
    ("a.txt", b"", ["context", "--store", "{dir}/none.muster", "q", "--fake-models"], "none.muster"),
    ("a.muster", b"not a database" * 200, ["context", "--store", "{file}", "q", "--fake-models"], "a.muster"),
])
def test_a_mistake_on_the_command_line_prints_a_sentence(tmp_path, name, data, argv, says):
    f = write(tmp_path, name, data)
    with pytest.raises(SystemExit) as e:
        main([x.replace("{file}", f).replace("{dir}", str(tmp_path)) for x in argv])
    assert isinstance(e.value.code, str) and says in e.value.code


def test_the_command_line_reads_a_store_of_emoji(tmp_path, capsys):
    f = write(tmp_path, "a.txt", "\U0001F44D".encode())
    main(["query", f, "How did they react?", "--fake-models"])
    assert "\U0001F44D" in capsys.readouterr().out
