"""Public collection entry, source-bound recipes, and unsafe shape mixing."""
import pytest

import muster


def narrative():
    return "The travelers followed the river and watched the mountain road. " * 1800


def dot_log():
    return "\n".join("%d. Person%d lives in Place%d." % (i + 1, i % 8, i % 3) for i in range(40))


def lead_log():
    return "\n".join("Step %d: Person%d lives in Place%d." % (i, i % 8, i % 3) for i in range(40))


def conversation():
    return [muster.Record("I moved to the northern town.", speaker="Alice", time="2026-10-01", session="first"),
            muster.Record("The new address is near the library.", speaker="Bob", time="2026-10-02", session="second")]


def session_literal():
    return repr(["2026/10/01 10:30", [{"role": "user", "content": "I moved to the northern town."},
                                     {"role": "assistant", "content": "I will remember your address."}],
                 "2026/10/02 11:40", [{"role": "user", "content": "The new address is near the library."},
                                     {"role": "assistant", "content": "That is the updated address."}]])


def test_explicit_collection_preserves_existing_list_of_records_meaning():
    stores = ["First source has the orange key.", "Second source has the green key."]
    single = muster.build(stores, embed=False)
    collection = muster.build_collection(stores, embed=False)
    assert isinstance(collection, muster.MemoryCollection)
    assert single.text == "\n\n".join(stores)
    assert len(collection) == 2 and tuple(collection) == collection.memories
    assert [memory.text for memory in collection] == stores
    assert all(memory.plan is collection.plan for memory in collection)
    assert collection[0].render(range(len(collection[0]))).text == stores[0]
    assert "green" not in collection[0].text and "orange" not in collection[1].text


def test_sample_prefix_and_all_stores_have_explicit_different_scope():
    stores = ["short record\n\nsecond short record"] * 3 + [narrative()]
    prefix = muster.build_collection(stores, embed=False)
    whole = muster.build_collection(stores, sample=None, embed=False)
    assert (prefix.plan.branch, prefix.plan.policy, prefix.plan.render) == ("record", "flrs", "flat")
    assert (whole.plan.branch, whole.plan.policy, whole.plan.render) == ("prose", "flr", "r4")
    assert prefix.plan.stats["rows"] == 3 and whole.plan.stats["rows"] == 4
    assert "sample=3" in " ".join(prefix.plan.trace)
    assert "sample=None" in " ".join(whole.plan.trace)
    assert all(memory.plan is whole.plan for memory in whole)
    assert whole.plan == muster.derive_collection(stores, sample=None)


def test_shared_parent_uses_mean_store_size_instead_of_concatenated_size():
    source = "\n\n".join("record%d %s" % (i, "plain " * 40) for i in range(20))
    one = muster.derive_collection([source])
    many = muster.derive_collection([source] * 100, sample=None)
    assert one.stats["tokens"] == many.stats["tokens"]
    assert one.mscale == many.mscale == 0


def test_typed_and_original_dated_speaker_records_select_conversation_recipe():
    records = conversation()
    raw = muster.serialize(records)[0]
    for source in (records, raw):
        memory = muster.build(source, embed=False)
        assert (memory.plan.branch, memory.plan.policy, memory.plan.render) == ("record", "flr", "r4")
        assert "2026-10-01" in memory.text and "Alice:" in memory.text
    collection = muster.build_collection([records, conversation()], embed=False)
    assert (collection.plan.policy, collection.plan.render) == ("flr", "r4")


@pytest.mark.parametrize("source", [
    "A story mentions [2026-10-01] Alice: in its middle.\n\nAnother ordinary paragraph.",
    "Alice: I mentioned 2026-10-01.\n\nBob: The date is in the body, not a source timestamp.",
    "[2026-10-01] Alice: a quote.\n\nplain paragraph\n\nanother paragraph\n\nlast paragraph",
])
def test_incomplete_or_embedded_conversation_imitation_keeps_independent_recipe(source):
    plan = muster.derive(source)
    assert (plan.branch, plan.policy, plan.render) == ("record", "flrs", "flat")


def test_leading_serials_keep_priority_over_speaker_shaped_text():
    source = "[42] Narrator: a numbered category.\n\n[43] Author: another category."
    memory = muster.build(source, embed=False)
    assert (memory.plan.branch, memory.plan.serial) == ("numbered", "lead")
    assert (memory.plan.policy, memory.plan.render) == ("attprio_sess", "numbered")
    assert memory.cut.order == (42, 43)
    rendered = memory.render(range(len(memory))).text
    assert rendered.startswith("[42] Narrator:") and "[43] Author:" in rendered


def test_metadata_serials_and_inner_numbered_lines_do_not_supply_a_body_clock():
    def records():
        return [muster.Record("Quoted lines:\n1. old detail\n2. new detail", id="%d." % i,
                              time="2026-10-01", speaker="Alice") for i in range(8)]
    collection = muster.build_collection([records(), records()], embed=False)
    assert collection.plan.branch == "record" and collection.plan.supersede == "off"
    assert all("1. old detail" in memory.text and "2. new detail" in memory.text for memory in collection)


@pytest.mark.parametrize("source,serial", [(dot_log(), "dot"), (lead_log(), "lead")])
def test_numbered_collections_share_a_plan_without_merging_source_clocks(source, serial):
    collection = muster.build_collection(["", source, source], embed=False)
    assert collection.plan.branch == "numbered" and collection.plan.serial == serial
    assert collection.plan.policy == "attprio_sess" and collection.plan.render == "numbered"
    assert collection[0].plan.branch == "empty" and len(collection[0]) == 0
    assert collection[0].plan.U == 0
    assert collection[1].plan is collection.plan and collection[2].plan is collection.plan
    assert collection[1].cut.order == collection[2].cut.order
    assert collection[1].cut.order == muster.build(source, embed=False).cut.order


def test_typed_lead_records_use_only_body_serials():
    records = [muster.Record(line, id="%d." % (1000 + i), speaker="Alice", time="2026-10-01")
               for i, line in enumerate(lead_log().splitlines())]
    collection = muster.build_collection([records, records], embed=False)
    assert collection.plan.branch == "numbered" and collection.plan.serial == "lead"
    assert collection[0].cut.diagnostics["serial_source"] == "record_bodies"


@pytest.mark.parametrize("stores", [
    [dot_log(), lead_log()],
    [dot_log(), "ordinary records"],
    ["ordinary records", dot_log()],
    [session_literal(), "ordinary records"],
])
def test_incompatible_source_boundaries_are_refused_even_outside_the_sample(stores):
    with pytest.raises(muster.PlanError, match="collection mixes"):
        muster.build_collection(stores, sample=1, embed=False)


def test_mixed_conversation_and_independent_text_can_share_a_non_structural_recipe():
    collection = muster.build_collection([conversation(), "ordinary records"], embed=False)
    assert collection.plan.branch == "record"
    assert (collection.plan.policy, collection.plan.render) == ("flrs", "flat")
    assert all(member.plan is collection.plan for member in collection)
    assert "mixed dated-speaker" in " ".join(collection.plan.trace)


def test_empty_collections_and_empty_sampling_do_not_force_a_nonempty_shape():
    empty = muster.build_collection([], embed=False)
    assert len(empty) == 0 and empty.plan.branch == "empty" and empty.plan.U == 0
    blanks = muster.build_collection(["", "\n\n"], embed=False)
    assert len(blanks) == 2 and all(memory.plan.branch == "empty" for memory in blanks)
    with pytest.raises(muster.PlanError, match="sample contains only empty"):
        muster.build_collection(["", "ordinary record"], sample=1, embed=False)
    assert muster.build_collection(["", "ordinary record"], sample=None, embed=False).plan.branch == "record"


def test_nested_record_iterators_are_consumed_once():
    collection = muster.build_collection(((value for value in ["alpha", "beta"]) for _ in range(2)), embed=False)
    assert [memory.text for memory in collection] == ["alpha\n\nbeta", "alpha\n\nbeta"]


def test_default_embedding_shares_one_model_container_across_memories(monkeypatch):
    import muster.models as model_module
    original = model_module.Models
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return original(encoder=model_module.FakeEncoder(16), reranker=model_module.FakeReranker())

    monkeypatch.setattr(model_module, "Models", factory)
    collection = muster.build_collection(["orange key", "green key"])
    assert len(calls) == 1
    assert collection[0].models is collection[1].models
    assert all(memory.vecs.shape == (1, 16) for memory in collection)


def test_session_literal_is_a_source_format_in_normal_build_and_collection():
    source = session_literal()
    memory = muster.build(source, embed=False)
    assert (memory.plan.branch, memory.plan.atom, memory.plan.chunk, memory.plan.mscale) == ("rounds", None, None, 0)
    assert (memory.plan.U, memory.plan.policy, memory.plan.render) == (10, "flrs", "flat")
    assert len(memory) == 2 and memory.parent == (0, 1)
    assert "I will remember" in memory.units[0] and "updated address" in memory.units[1]
    collection = muster.build_collection([source, source], embed=False)
    assert collection.plan.branch == "rounds" and all(member.plan is collection.plan for member in collection)
    explicit = muster.derive("plain record").with_(source="custom", frozen=True)
    assert muster.build(source, explicit, embed=False).plan is explicit


@pytest.mark.parametrize("value", [
    [], ["date"], ["", [{"role": "user", "content": "hello"}]],
    ["date", []], ["date", [{"role": "assistant", "content": "hello"}]],
    ["date", [{"role": "user", "content": 42}]],
    ["date", [{"role": "tool", "content": "hello"}]],
    ["date", [{"role": "user", "other": "hello"}]],
])
def test_partial_session_literal_imitations_are_not_rounds(value):
    assert muster.derive(repr(value)).branch != "rounds"


@pytest.mark.parametrize("source", [
    r'["2026-01-01", [{"role": "user", "content": "\ud800"}]]',
    r'["\ud800", [{"role": "user", "content": "ordinary turn"}]]',
    r'["2026-01-01", [{"role": "user", "content": "\udc00"}]]',
])
def test_escaped_surrogates_in_decoded_sessions_preserve_the_valid_source_literal(source):
    source.encode("utf-8")
    memory = muster.build(source, embed=False)
    assert memory.plan.branch != "rounds"
    assert memory.text == source and "".join(memory.units) == source
    collection = muster.build_collection([source, source], embed=False)
    assert collection.plan.branch != "rounds"
    assert all(member.text == source for member in collection)


@pytest.mark.parametrize("P", [1, 127, 1024, 8193])
def test_native_positive_resource_sizes_are_supported(P):
    assert muster.build("a source record", reference_chunk=P, embed=False).plan.reference_chunk == P
    assert muster.build_collection(["a", "b"], reference_chunk=P, embed=False).plan.reference_chunk == P


@pytest.mark.parametrize("P", [True, False, 0, -1, 1.5, "512", None, float("inf")])
def test_invalid_native_resource_sizes_are_refused(P):
    with pytest.raises(muster.PlanError, match="reference_chunk"):
        muster.derive("a", reference_chunk=P)
    with pytest.raises(muster.PlanError, match="reference_chunk"):
        muster.build_collection(["a"], reference_chunk=P, embed=False)


@pytest.mark.parametrize("sample", [True, False, 0, -1, 1.5, "3"])
def test_invalid_native_sample_sizes_are_refused(sample):
    with pytest.raises(muster.PlanError, match="sample"):
        muster.build_collection(["a"], sample=sample, embed=False)


@pytest.mark.parametrize("stores", ["raw text", {"text": "record"}, {"a", "b"}, None])
def test_outer_collection_must_be_an_ordered_collection_of_stores(stores):
    with pytest.raises(muster.PlanError, match="ordered collection"):
        muster.build_collection(stores, embed=False)


def test_explicit_plans_and_historical_exact_collection_recipe_are_not_rederived():
    explicit = muster.presets.get("mab.A").with_(mode="native")
    assert muster.build(narrative(), explicit, reference_chunk=0, embed=False).plan is explicit
    historical = muster.derive_collection([narrative()] * 3, mode="exact")
    assert historical.branch == "prose" and (historical.policy, historical.render) == ("flrs", "r4")
