"""Real SQLite and Store API checks, with fake models and no provider calls."""
import sqlite3

import pytest

import muster
from muster.models import Models


RECORDS = [
    {"id": "a", "speaker": "Ann", "text": "Ann moved to Lisbon."},
    {"id": "b", "speaker": "Ben", "text": "Ben owns a blue bicycle."},
    {"id": "c", "speaker": "Ann", "text": "Ann works on the Atlas project."},
]


def test_readonly_missing_file_is_not_created(tmp_path):
    path = tmp_path / "missing.muster"
    with pytest.raises(sqlite3.OperationalError):
        with muster.Store.open(str(path), readonly=True):
            pass
    assert not path.exists()


def test_readonly_connection_rejects_sql_writes(tmp_path):
    path = tmp_path / "agent.muster"
    with muster.Store.open(str(path)) as writer:
        writer.add(RECORDS)
    with muster.Store.open(str(path), readonly=True) as reader:
        assert reader.version == len(RECORDS)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.db.execute("INSERT INTO meta VALUES ('probe', 'unexpected write')")
        with pytest.raises(PermissionError):
            reader.add("new record")


@pytest.mark.parametrize("warm", [False, True])
def test_readonly_search_preserves_context_and_persistent_schema(tmp_path, monkeypatch, warm):
    # A relative Path with URI metacharacters must designate the literal filename.
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "agent ?#%.muster"
    models = Models.fake(16)
    fresh = muster.build(RECORDS, models=models)
    queries = ["Where did Ann move?", "Who owns a bicycle?"]
    expected = [fresh.context(q) for q in queries]
    with muster.Store.open(str(path), models=models) as writer:
        writer.add(RECORDS)
        if warm:
            assert [writer.context(q).text for q in queries] == [c.text for c in expected]
    with sqlite3.connect(str(path)) as db:
        schema_before = list(db.execute("SELECT name, sql FROM sqlite_master ORDER BY name"))
    before = path.read_bytes()
    with muster.Store.open(path.relative_to(tmp_path), models=models, readonly=True) as reader:
        actual = [reader.context(q) for q in queries]
        assert [c.text for c in actual] == [c.text for c in expected]
        assert [c.ids for c in actual] == [c.ids for c in expected]
        assert reader.version == len(RECORDS)
        schema_after = list(reader.db.execute("SELECT name, sql FROM main.sqlite_master ORDER BY name"))
        assert schema_after == schema_before
    assert path.read_bytes() == before
    assert (tmp_path / (path.name + ".scores")).exists() == warm


def test_pathlike_writable_store_can_search_and_reopen(tmp_path):
    path = tmp_path / "pathlike.muster"
    models = Models.fake(16)
    with muster.Store.open(path, models=models) as writer:
        writer.add(RECORDS, request_id="batch-1")
        expected = writer.context("Where did Ann move?").text
    assert (tmp_path / "pathlike.muster.scores").exists()
    with muster.Store.open(path, models=models) as writer:
        assert writer.add(RECORDS, request_id="batch-1").duplicate
        assert writer.context("Where did Ann move?").text == expected
