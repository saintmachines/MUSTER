"""1.0: the record branch's third test and whole-record restore.

A record store larger than its read (T > U x P = 10 x 4,096) whose record share still holds at a quarter of the unit
takes units of 128 tokens and U 40 (the same unit tokens), and a read restores whole records within those tokens.
Every other plan keeps its 0.1.3 JSON and plan_id; the pins below were computed with muster-memory 0.1.3.
"""
import random

import pytest

import muster
from muster.derive import RECORD_TAU, REFINE_TRACE, _parents
from muster.models import FakeEncoder, FakeReranker, Models
from muster.plan import Plan
from muster.stats import store_stats

PIN_013 = {  # muster-memory 0.1.3 on the stores below (make_pins.py)
    'big_long': 'd34a53c35f0d81761138f03ed4b9e962a2ef813c89775896fef6a757dcd9676b',
    'big_short': 'f112e7324a5dd1d2d66360ecea51cbc17c2137cdf6251fe05fda01ec38e324c5',
    'paragraph_records': '8e84bcc28b58e6555f67f8f18bac3e2b55674ab70808ca8a35c12d950b14153a',
    'sha_big_long': '8a20f876ee7dd8fb',
    'sha_big_short': '5ee81cb224983914',
    'sha_small': '2d3cde700f91d837',
    'small': '1a044b507a92bc435ca6d525053e56a78b4a6863174a1e1f63fbf45565eaa318',
    'small_collection': '82c8066226f58695ed0d213f7708324c32d774ec52205947eb2948f8fcd09e20',
}


def _sent(rnd, n):
    words = "alpha beta gamma delta river stone light Paris Maria engine report seven".split()
    return " ".join(rnd.choice(words) for _ in range(n)).capitalize() + "."


def _records_text(n, lo, hi, seed):
    rnd = random.Random(seed)
    return "\n\n".join("D%d:%d [2023-05-0%d] Ann: %s" % (i // 9, i, 1 + i % 9, _sent(rnd, rnd.randint(lo, hi)))
                       for i in range(n))


SMALL = _records_text(400, 8, 16, 7)          # about 10K tokens: T <= U x P, the plan stays
BIG_SHORT = _records_text(3000, 8, 16, 7)     # about 75K tokens of short records: refined
BIG_LONG = _records_text(300, 150, 250, 7)    # about 75K tokens, records over 128 tokens: share at 128 fails, stays


def _paragraph_records():
    """Chat records of several short paragraphs each (about 300 tokens a record): blank-line blocks fit 128 tokens, the
    records do not. The third test reads the records, so the store keeps its 0.1.3 plan (rule v3)."""
    rnd = random.Random(5)
    out = []
    for i in range(400):
        body = "\n\n".join(_sent(rnd, rnd.randint(14, 22)) for _ in range(12))
        out.append(muster.Record(text=body, id="D%d:%d" % (i // 20, i), time="2023-05-08", speaker="Ann"))
    return out


def _mixed_records():
    """Short records with every 200th record long enough to span several 128-token units."""
    rnd = random.Random(3)
    out = []
    for i in range(2600):
        n = rnd.randint(150, 220) if i % 200 == 100 else rnd.randint(8, 16)
        out.append(muster.Record(text=_sent(rnd, n), id="D%d:%d" % (i // 20, i), time="2023-05-08", speaker="Ann"))
    return out


def test_pinned_stores_are_the_stores_pinned():
    import hashlib
    for k, text in (("small", SMALL), ("big_short", BIG_SHORT), ("big_long", BIG_LONG)):
        assert hashlib.sha256(text.encode()).hexdigest()[:16] == PIN_013["sha_" + k]


def test_small_record_store_keeps_its_013_plan():
    plan = muster.derive(SMALL)
    assert plan.branch == "record" and plan.chunk == 512 and plan.U == 10 and plan.restore == "off"
    assert "restore" not in plan.to_dict()
    assert plan.plan_id == PIN_013["small"]


def test_large_store_of_long_records_keeps_its_013_plan():
    plan = muster.derive(BIG_LONG)
    assert plan.stats["tokens"] > plan.U * plan.reference_chunk
    assert store_stats(BIG_LONG, 128).record_share < RECORD_TAU <= store_stats(BIG_LONG, 4096).record_share
    assert plan.chunk == 512 and plan.restore == "off" and plan.plan_id == PIN_013["big_long"]


def test_large_store_of_short_records_takes_the_quarter_step():
    plan = muster.derive(BIG_SHORT)
    T = plan.stats["tokens"]
    assert T > 10 * 4096
    assert (plan.branch, plan.chunk, plan.U, plan.restore) == ("record", 128, 40, "records")
    assert plan.mscale == _parents(T // 128, 128)[0] == 1024
    assert plan.trace[-1].startswith(REFINE_TRACE) and sum(t.startswith(REFINE_TRACE) for t in plan.trace) == 1
    assert plan.plan_id != PIN_013["big_short"]
    assert plan.with_(chunk=512, mscale=4096, U=10, restore="off", trace=plan.trace[:-1]).plan_id == PIN_013["big_short"]
    assert Plan.from_dict(plan.to_dict()) == plan and muster.derive(BIG_SHORT).plan_id == plan.plan_id


def test_records_longer_than_the_refined_unit_keep_their_013_plan():
    recs = _paragraph_records()
    text, spans = muster.serialize(recs)
    assert store_stats(text, 128).record_share >= RECORD_TAU      # the blank-line blocks would pass
    plan = muster.derive(recs)
    assert plan.stats["tokens"] > 10 * 4096
    assert plan.chunk == 512 and plan.restore == "off" and plan.plan_id == PIN_013["paragraph_records"]
    coll = muster.derive_collection([recs, recs])
    assert coll.restore == "off" and coll.chunk == 512


def test_collection_refines_on_its_pooled_statistics():
    big = muster.build_collection([BIG_SHORT, BIG_SHORT], embed=False)
    assert (big.plan.chunk, big.plan.U, big.plan.restore, big.plan.source) == (128, 40, "records", "derived:collection")
    assert all(len(m.cut.units) > 400 for m in big)
    small = muster.build_collection([SMALL, SMALL, SMALL], embed=False)
    assert small.plan.restore == "off" and small.plan.plan_id == PIN_013["small_collection"]


def test_restore_returns_whole_records_within_the_selection_tokens():
    mem = muster.build(_mixed_records(), embed=False)
    assert mem.plan.restore == "records" and mem.plan.chunk == 128
    mem.unit_records(0)
    spread = {}
    for i, ks in enumerate(mem._ur):
        for k in ks:
            spread.setdefault(k, []).append(i)
    spread = {k: v for k, v in spread.items() if len(v) > 1}
    assert spread
    k, units = sorted(spread.items())[0]
    n = mem.cut.n_o200k
    filler = [i for i in range(len(mem.cut.units)) if i not in units and len(mem._ur[i]) and
              all(len([j for j, ks in enumerate(mem._ur) if kk in ks]) == 1 for kk in mem._ur[i])][:12]
    ids = [units[0]] + filler
    out, budget, used, restored = muster.restore_records(mem, ids)
    assert budget == sum(n[i] for i in ids) and used <= budget
    if sum(n[j] for j in units) <= budget:
        assert out[:len(units)] == sorted(units) and restored >= 1
    assert len(set(out)) == len(out)
    alone, b1, u1, r1 = muster.restore_records(mem, [units[0]])   # the record does not fit one slice's tokens
    assert alone == [units[0]] and (b1, u1, r1) == (n[units[0]], n[units[0]], 0)


def test_search_restores_only_on_refined_plans():
    models = Models(encoder=FakeEncoder(16), reranker=FakeReranker())
    mem = muster.build(_mixed_records(), models=models)
    sel = mem.search("Where is Maria's engine report?", models=models)
    assert sel.selected and sel.restored is not None
    assert list(sel.ids) == muster.restore_records(mem, sel.selected)[0]
    assert mem.render(sel).text == mem.render(list(sel.ids)).text
    plain = muster.build(SMALL, models=models)
    s0 = plain.search("Where is Maria's engine report?", models=models)
    assert s0.selected == () and s0.restored is None


def test_plain_string_store_refines_without_records_to_restore():
    mem = muster.build(BIG_SHORT, embed=False)
    assert mem.plan.restore == "records"
    ids = list(range(0, 40))
    budget = sum(mem.cut.n_o200k[i] for i in ids)
    assert budget > 0
    assert muster.restore_records(mem, ids) == (ids, budget, budget, 0)


@pytest.mark.parametrize("bad", ["on", "", None])
def test_restore_field_is_checked(bad):
    with pytest.raises(ValueError):
        muster.derive(SMALL).with_(restore=bad)
