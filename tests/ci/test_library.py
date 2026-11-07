"""C11 and the Store, with fake models (no weights): the API an application uses."""
import os
import random

import pytest

import muster
from muster.models import Models

np = pytest.importorskip("numpy")
pytest.importorskip("rank_bm25")


def _chat(n=120, seed=1):
    rnd = random.Random(seed)
    places = ["Lisbon", "Paris", "Seoul", "Boston", "Nairobi"]
    out = []
    for i in range(n):
        p = rnd.choice(places)
        out.append({"id": "D%d:%d" % (i // 10 + 1, i % 10 + 1), "time": "2026-05-%02d" % (1 + i % 28),
                    "speaker": rnd.choice(["Ann", "Ben"]),
                    "text": "%s talked about %s and the %s project number %d." % (rnd.choice(["Ann", "Ben"]), p,
                                                                                rnd.choice(["Atlas", "Nova"]), i)})
    return out


def test_build_search_render_native():
    m = Models.fake(32)
    mem = muster.build(_chat(), models=m)
    assert mem.plan.branch == "record" and mem.plan.mode == "native"
    sel = mem.search("What did Ann say about Lisbon?")
    assert 0 < len(sel.ids) <= mem.plan.U
    ctx = mem.render(sel)
    assert ctx.text.startswith("Memory 1:\n") and ctx.style == "r4"
    for i in sel.ids:
        assert mem.units[i] in ctx.text
    # deterministic: same inputs, same context
    assert muster.build(_chat(), models=m).context("What did Ann say about Lisbon?").text == ctx.text


def test_ties_are_deterministic_in_native_mode():
    m = Models.fake(8, quantize=2)          # coarse vectors and scores: many exact ties
    a = muster.build(_chat(200, 3), models=m).context("project Nova Paris", U=12).text
    b = muster.build(_chat(200, 3), models=m).context("project Nova Paris", U=12).text
    assert a == b


def test_u_beyond_the_store_returns_what_exists():
    m = Models.fake(16)
    mem = muster.build(_chat(6), models=m)
    sel = mem.search("Lisbon", U=50)
    assert len(sel.ids) == len(set(sel.ids)) <= len(mem.units)


def test_empty_query_and_empty_store():
    m = Models.fake(16)
    mem = muster.build(_chat(30), models=m)
    with pytest.raises(ValueError):
        mem.search("   ")
    empty = muster.build([], models=m)
    assert empty.plan.branch == "empty" and empty.search("anything").ids == ()


def test_stale_selection_is_refused():
    m = Models.fake(16)
    a = muster.build(_chat(40), models=m)
    b = muster.build(_chat(41), models=m)
    sel = a.search("Lisbon")
    with pytest.raises(muster.StaleSelection):
        b.render(sel)


def test_floors_follow_the_rule():
    from muster.search import floors_for
    assert floors_for(10) == (4, 3, 3) and floors_for(20) == (8, 6, 6) and floors_for(40) == (16, 12, 12)
    assert floors_for(5) == (3, 1, 1) and floors_for(4) == (2, 1, 1)


def test_numbered_log_newest_slot_wins():
    m = Models.fake(16)
    facts = [f for i in range(10) for f in ("Project%d is led by Maria." % i, "Project%d is led by Jorge." % i)]
    mem = muster.build(muster.numbered(facts), models=m)
    assert mem.plan.branch == "numbered"
    ctx = mem.context("Who leads Project3?")
    assert ctx.style == "numbered" and ctx.text.startswith("[")
    assert "Jorge" in ctx.text and "Maria" not in ctx.text


def test_store_add_search_reopen(tmp_path):
    p = str(tmp_path / "agent.muster")
    m = Models.fake(32)
    with muster.Store.open(p, models=m) as s:
        r1 = s.add(_chat(50), request_id="batch-1")
        r2 = s.add(_chat(50), request_id="batch-1")          # retried request: acknowledged, not appended
        assert not r1.duplicate and r2.duplicate and s.version == 50
        s.add({"speaker": "Ann", "text": "Ann moved to Kyoto in June.", "time": "2026-06-01"})
        c1 = s.context("Where did Ann move?")
        assert "Kyoto" in c1.text
    with muster.Store.open(p, models=m) as s:
        assert s.version == 51
        c2 = s.context("Where did Ann move?")
        assert c2.text == c1.text                                # same version, same context after a restart


def test_store_head_equals_a_fresh_build(tmp_path):
    m = Models.fake(32)
    recs = _chat(90, 5)
    with muster.Store.open(str(tmp_path / "s.muster"), models=m) as s:
        for k in range(0, 90, 7):
            s.add(recs[k:k + 7])
        head = s.head()
        fresh = muster.build(recs, models=m)
        assert head.units == fresh.units and head.parent == fresh.parent and head.plan == fresh.plan
        assert np.array_equal(head.vecs, fresh.vecs)
        q = "What did Ben say about Seoul?"
        assert s.context(q).text == fresh.context(q).text


def test_import_is_light():
    import subprocess, sys
    code = ("import sys, muster; muster.derive('a b c. d e f.'); "
            "print(','.join(m for m in ('numpy','torch','nltk','rank_bm25') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    assert out.returncode == 0 and out.stdout.strip() == "", out.stdout + out.stderr


def test_cli_query_on_a_jsonl_of_records(tmp_path, capsys):
    import json as _json
    from muster.cli import main
    p = tmp_path / "chat.jsonl"
    p.write_text("\n".join(_json.dumps(r) for r in _chat(40)) + "\n" +
                 _json.dumps({"speaker": "Ann", "text": "Ann keeps the spare key in the red box."}) + "\n")
    main(["query", str(p), "Where is the spare key?", "--fake-models"])
    assert "red box" in capsys.readouterr().out


def test_exact_plan_refuses_a_model_of_another_length():
    from muster.plan import ModelSpec
    from muster.models import FakeEncoder, FakeReranker

    class Enc(FakeEncoder):
        spec = ModelSpec("BAAI/bge-m3", None, 448, 32)
    m = Models(encoder=Enc(16), reranker=FakeReranker())
    mem = muster.build("\n\n".join("D%d: some text about Paris number %d" % (i, i) for i in range(300)), "beam.u20",
                       models=m)
    with pytest.raises(muster.EnvironmentRefused):
        mem.search("Paris")


def test_blocks_name_their_source_records():
    m = Models.fake(32)
    recs = _chat(80, 9)
    mem = muster.build(recs, models=m)
    ctx = mem.context("What did Ben say about Nairobi?")
    for b in ctx.blocks:
        srcs = [r for i in b.ids for r in mem.unit_records(i)]
        assert srcs, "every block comes from at least one record"
        for r in srcs:
            assert r.text in b.text
    # every record of the store is covered by exactly the units its text sits in
    covered = [r.id for i in range(len(mem.units)) for r in mem.unit_records(i)]
    assert sorted(set(covered)) == sorted(r["id"] for r in recs)


def test_scratch_store_in_memory():
    with muster.Store.open(":memory:", models=Models.fake(16)) as s:
        s.add([{"speaker": "Ann", "text": "Ann's locker code is 4417."}, {"speaker": "Ben", "text": "Ben likes tea."}])
        assert "4417" in s.context("What is Ann's locker code?").text and s.version == 2
