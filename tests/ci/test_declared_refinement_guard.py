"""Declared units keep their original read count; restoration accounting is truthful.

CPU only. Models.fake exercises selection without loading neural weights.
"""
import pytest

import muster
from muster.derive import _derive_one


BODY = "source observation about alpha and beta with matching details"


def declaration():
    return muster.SourceOrder(
        source_sha256="a" * 64, source_locator="fixture.json#/records",
        kind="structured_array", declaration="Original source-array order.",
        namespace="declared-refinement-guard")


@pytest.fixture(scope="module")
def long_records():
    return [{"text": BODY} for _ in range(6000)]


def test_large_declared_record_keeps_original_U_and_real_selection(long_records):
    source = "\n\n".join(row["text"] for row in long_records)
    old = _derive_one(source, refine=False)
    refined = muster.derive(source)
    assert old.branch == "record" and old.U == 10
    assert old.stats["tokens"] > old.U * old.reference_chunk
    assert refined.U == 40 and refined.restore == "records"
    mem = muster.build(long_records, declared_order=declaration(), embed=False)
    assert mem.text == source and len(mem) == len(long_records)
    assert mem.units == tuple(row["text"] for row in long_records)
    assert mem.plan.U == old.U and mem.plan.restore == "off"
    assert mem.plan.chunk is None and mem.plan.mscale == 0
    assert mem.ordered_records.roundtrip() == long_records
    selected = mem.search("alpha observation", models=muster.Models.fake(16))
    assert selected.U == 10 and len(selected.ids) == 10
    assert selected.restored is None and selected.selected == ()


@pytest.mark.parametrize("bodies, branch, U", [
    ([BODY, "another source observation"], "record", 10),
    (["A continuous narrative about alpha and beta. " * 6000], "prose", 40),
])
def test_declared_default_keeps_base_record_or_prose_read_count(bodies, branch, U):
    source = "\n\n".join(bodies)
    original = _derive_one(source, refine=False)
    assert (original.branch, original.U) == (branch, U)
    mem = muster.build(bodies, declared_order=declaration(), embed=False)
    assert mem.plan.U == U and mem.plan.restore == "off"
    assert mem.text == source and mem.units == tuple(bodies)
    assert mem.plan.stats == original.stats


@pytest.mark.parametrize("U, restore", [(7, "off"), (23, "records")])
def test_declared_explicit_U_is_preserved_and_restore_is_disabled(U, restore):
    explicit = muster.derive(BODY).with_(U=U, restore=restore)
    before = explicit.to_dict()
    mem = muster.build([BODY, "another source observation"], explicit,
                       declared_order=declaration(), embed=False)
    assert mem.plan.U == U and mem.plan.restore == "off"
    assert mem.plan.chunk is None and mem.plan.atom is None
    assert explicit.to_dict() == before


def test_declared_exact_plan_is_still_refused():
    exact = muster.derive(BODY).with_(mode="exact", frozen=True)
    with pytest.raises(muster.PlanError, match="cannot change exact replay"):
        muster.build([BODY], exact, declared_order=declaration(), embed=False)


def test_raw_restore_keeps_ids_context_and_reports_actual_unit_budget():
    text = "\n\n".join([BODY] * 30)
    plan = muster.derive(text).with_(chunk=32, mscale=0, U=3, restore="records")
    raw = muster.build(text, plan, embed=False)
    off = muster.build(text, plan.with_(restore="off"), embed=False)
    assert raw.records is None and len(raw) > 3
    ids = [2, 0, 1]
    got, budget, used, restored = muster.restore_records(raw, ids)
    expected = sum(raw.cut.n_o200k[i] for i in ids)
    assert got == ids and expected > 0
    assert (budget, used, restored) == (expected, expected, 0)
    assert raw.render(got).text == off.render(ids).text
    selected = raw.search("alpha observation", models=muster.Models.fake(16))
    expected = sum(raw.cut.n_o200k[i] for i in selected.selected)
    assert tuple(selected.ids) == selected.selected
    assert selected.restored == {"budget": expected, "used": expected, "records": 0}
    assert raw.render(selected).text == off.render(selected.ids).text


def test_empty_raw_restore_still_reports_zero():
    mem = muster.build("", embed=False)
    assert muster.restore_records(mem, []) == ([], 0, 0, 0)


def test_open_explain_is_conditional_without_mutating_plan_trace():
    phrase = "a read restores whole records within its unit tokens"
    plan = muster.derive(BODY).with_(chunk=128, U=40, restore="records", trace=(phrase,))
    before = plan.to_dict()
    shown = muster.explain(plan)
    assert "typed records" in shown and "when their units fit" in shown
    assert "raw text" in shown and phrase not in shown
    assert plan.to_dict() == before and plan.trace == (phrase,)


def test_closed_explain_retains_original_wording_and_trace():
    plan = muster.derive(BODY).with_(trace=("original trace",))
    assert plan.restore == "off"
    expected = "\n".join("  %-10s %s" % pair for pair in (
        ("store", "record store"), ("units", "512 tokens, records whole"),
        ("parents", "none"), ("read", "10 units per query"),
        ("mode", "native"), ("source", plan.source)))
    assert muster.explain(plan) == expected + "\n  how:\n    - original trace"


def test_open_cli_describes_conditional_typed_restoration_only(capsys):
    from muster.cli import _decision, _show
    decision = _decision(0.0, 1.0, 100000, unit=128)
    _show("source.txt", 100000, decision)
    shown = capsys.readouterr().out
    assert "128 tokens, 4x as many" in shown
    assert "typed records restored whole when they fit" in shown
