"""Native session rounds preserve every turn without renumbering existing rounds."""
import ast
from collections import Counter
import json

import pytest

import muster
from muster.cut import cut
from muster.render import flat


def source_sessions():
    def turn(role, content):
        return dict(role=role, content=content, source="original producer field")

    return [
        "2026-11-11", [
            turn("system", "SYS_HEAD_α"),
            turn("assistant", "LEAD_β"),
            turn("user", "USER_A_γ"),
            turn("assistant", "ANSWER_A_δ"),
            turn("assistant", "EXTRA_ε"),
            turn("system", "SYS_MID_ζ"),
            turn("user", "USER_B_η"),
            turn("user", "USER_C_θ"),
            turn("system", "SYS_PAIRED_ι"),
            turn("assistant", "TAIL_κ"),
        ],
        "2026-11-12", [
            turn("assistant", "NEXT_HEAD_λ"),
            turn("user", "USER_D_μ"),
            turn("assistant", "ANSWER_D_ν"),
            turn("system", "SYS_TAIL_ξ"),
        ],
    ]


def exact_plan():
    return muster.presets.get("mab.A1")


EXPECTED_ROUNDS = (
    "[2026-11-11] USER_A_γ\nANSWER_A_δ",
    "[2026-11-11] USER_B_η",
    "[2026-11-11] USER_C_θ\nSYS_PAIRED_ι",
    "[2026-11-12] USER_D_μ\nANSWER_D_ν",
)
EXPECTED_KEYS = ("USER_A_γ", "USER_B_η", "USER_C_θ", "USER_D_μ")
SOURCE_IDS = (4, 5, 0, 6, 7, 1, 2, 8, 9, 3, 10)


@pytest.mark.parametrize("serialize", [repr, lambda value: json.dumps(value, ensure_ascii=False)])
def test_native_rounds_cover_every_source_turn_once_and_keep_raw_source(serialize):
    sessions = source_sessions()
    source = serialize(sessions)
    memory = muster.build(source, embed=False)
    assert memory.plan.branch == "rounds" and memory.plan.mode == "native"
    assert memory.text == source and ast.literal_eval(memory.text) == sessions
    assert memory.records is None
    assert memory.units[:4] == EXPECTED_ROUNDS and memory.cut.keys[:4] == EXPECTED_KEYS
    assert memory.cut.parent == (0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1)
    assert memory.cut.order == (2, 6, 7, 11, 0, 1, 4, 5, 9, 10, 13)
    assert memory.cut.diagnostics["n_user_rounds"] == 4
    assert memory.cut.diagnostics["orphan_units"] == tuple(range(4, 11))
    expected_positions = {(si, ti) for si, turns in enumerate(sessions[1::2]) for ti in range(len(turns))}
    covered = Counter(position for positions in memory.cut.diagnostics["source_turns"] for position in positions)
    assert set(covered) == expected_positions and set(covered.values()) == {1}
    for turns in sessions[1::2]:
        for turn in turns:
            assert "\n".join(memory.units).count(turn["content"]) == 1
    assert all(span is None for span in memory.cut.spans)


def test_exact_preset_retains_original_round_units_keys_ids_and_diagnostics():
    source = repr(source_sessions())
    plan = exact_plan()
    assert plan.mode == "exact" and plan.branch == "rounds"
    exact = cut(source, plan)
    native = cut(source, plan.with_(mode="native"))
    assert exact.units == EXPECTED_ROUNDS and exact.keys == EXPECTED_KEYS
    assert exact.parent == (0, 0, 0, 1) and exact.order == (0, 1, 2, 3)
    assert exact.spans == (None,) * 4 and exact.sep is None
    assert exact.diagnostics == dict(n_sessions=2, dates=("2026-11-11",) * 3 + ("2026-11-12",))
    for field in ("units", "keys", "parent", "spans", "n_o200k"):
        assert getattr(native, field)[:4] == getattr(exact, field)
    assert cut(source, plan.with_(mode="native"), exact=True) == exact
    assert cut(source, plan, exact=False) == native


@pytest.mark.parametrize("ids", [[], [3, 0], [2, 1, 0], [3, 2, 1, 0]])
def test_old_round_selection_keeps_its_complete_rendered_bytes(ids):
    source = repr(source_sessions())
    native = muster.build(source, embed=False)
    exact = muster.build(source, exact_plan(), embed=False)
    before, after = exact.render(ids), native.render(ids)
    assert after.text == before.text
    assert after.ids == before.ids and after.blocks == before.blocks
    assert after.messages("a fixed request") == before.messages("a fixed request")


def test_appended_turns_render_at_their_source_positions_with_their_own_session():
    memory = muster.build(repr(source_sessions()), embed=False)
    context = memory.render(reversed(range(len(memory))))
    assert context.ids == SOURCE_IDS
    assert context.text == "\n".join(memory.units[i] for i in SOURCE_IDS)
    assert [block.ids for block in context.blocks] == [(i,) for i in SOURCE_IDS]
    assert [block.parent for block in context.blocks] == [memory.cut.parent[i] for i in SOURCE_IDS]
    assert [block.index for block in context.blocks] == list(range(1, len(memory) + 1))
    selected = memory.render([3, 9, 0, 4, 6])
    assert selected.ids == (4, 0, 6, 9, 3)
    assert "SYS_HEAD_α" in selected.text and "EXTRA_ε" in selected.text and "NEXT_HEAD_λ" in selected.text


def test_optional_flat_order_leaves_existing_ordinary_and_exact_calls_unchanged():
    units, parent, order = ("later", "first", "middle"), ("a", "b", "a"), (20, 0, 10)
    assert flat([2, 0, 1], units, parent).ids == (0, 1, 2)
    assert flat([2, 0, 1], units, parent, order).ids == (1, 2, 0)
    memory = muster.build(["first record", "second record"], embed=False)
    ordinary = memory.render([1, 0]) if len(memory) > 1 else memory.render([0])
    assert ordinary.text == "\n".join(memory.units[i] for i in sorted(ordinary.ids))


def test_unpaired_whitespace_is_preserved_and_empty_turns_are_not_invented():
    sessions = ["2026-11-11", [
        dict(role="assistant", content=""),
        dict(role="system", content=" \t "),
        dict(role="user", content="  old user  "),
        dict(role="assistant", content="  old reply  "),
        dict(role="assistant", content="\n  extra reply  \t\n"),
    ]]
    source = repr(sessions)
    native = muster.build(source, embed=False)
    exact = muster.build(source, exact_plan(), embed=False)
    assert native.units[0] == exact.units[0] and native.cut.keys[0] == exact.cut.keys[0]
    assert native.units[1:] == ("[2026-11-11]  \t ", "[2026-11-11] \n  extra reply  \t\n")
    assert native.cut.order == (2, 1, 4)
    assert native.cut.diagnostics["source_turns"] == (((0, 2), (0, 3)), ((0, 1),), ((0, 4),))
    assert native.text == source


@pytest.mark.parametrize("user", ["", " \t "])
def test_native_blank_user_key_can_index_and_search_its_nonempty_reply(user):
    source = repr(["2026-11-11", [dict(role="user", content=user),
                                  dict(role="assistant", content="The amber token opens the archive.")]])
    exact = muster.build(source, exact_plan(), embed=False)
    assert exact.cut.keys == ("",)
    memory = muster.build(source, models=muster.Models.fake(16))
    assert memory.units == exact.units and memory.cut.keys == memory.units
    assert memory.index is not None
    context = memory.context("Which token opens the archive?")
    assert "amber token" in context.text and context.ids == (0,)


def test_collection_keeps_session_sources_independent_and_record_metadata_opaque():
    source = repr(source_sessions())
    collection = muster.build_collection([source, source], embed=False)
    assert collection.plan.branch == "rounds"
    assert all(member.plan is collection.plan for member in collection)
    assert all(member.text == source and member.cut.order == collection[0].cut.order for member in collection)
    assert all(member.render(range(len(member))).ids == SOURCE_IDS for member in collection)
    record = muster.Record(source, meta={"source": "original producer source"})
    typed = muster.build([record], embed=False)
    assert typed.plan.branch == "record" and typed.text == source
    assert typed.records == [record] and typed.records[0] is record
    assert typed.records[0].meta["source"] == "original producer source"
    assert typed.records[0].text == source


@pytest.mark.parametrize("value", [
    ["2026-11-11", [dict(role="assistant", content="a source without any user")]],
    ["2026-11-11", [dict(role="user", content="kept"), dict(role="tool", content="not a supported source role")]],
    ["2026-11-11", [dict(role="user", content="kept"), dict(role="assistant", content=42)]],
])
def test_native_turn_preservation_does_not_relax_auto_session_validation(value):
    source = repr(value)
    memory = muster.build(source, embed=False)
    assert memory.plan.branch != "rounds" and memory.text == source
