"""Native numbered logs: a later line retires an earlier one of the same subject and relation, whatever shape the value has."""
import sys
import time

import pytest

import muster
from muster.cut import cut, line_relation, slot_of, value_slots


FACTS = [
    "Kobe Bryant is associated with the sport of basketball.",
    "Lionel Messi is associated with the sport of association football.",
    "Kobe Bryant is associated with the sport of cricket.",
]
PEOPLE = ["Alice Reed", "Bob Stone", "Carol Hart", "Dan Cole", "Erin Ford", "Finn Page", "Gina Lowe", "Hugo Dale", "Iris Vane", "Jack Moss", "Kira Webb", "Liam Shaw"]


def bodies():
    # Capitalised values give the update evidence the numbered check needs; the sport lines hold lower-case values.
    out = []
    for who in PEOPLE:
        out += ["%s lives in Paris." % who, "%s plays the sport of tennis." % who, "%s lives in Rome." % who, "%s plays the sport of golf." % who]
    return out


def text():
    return "\n".join("%d. %s" % (i, b) for i, b in enumerate(bodies()))


def test_a_lower_case_value_does_not_stay_in_the_slot():
    keys = value_slots(FACTS)
    assert keys[0] == keys[2] == "v:kobe bryant||is associated with the sport"
    assert keys[1] == "v:lionel messi||is associated with the sport"
    assert slot_of(FACTS[0]) != slot_of(FACTS[2])


def test_an_oddly_shaped_value_does_not_split_a_slot():
    lines = ["The univeristy where Bill Frisell was educated is Berklee College of Music.",
             "The univeristy where Bill Frisell was educated is The University of the Arts.",
             "The univeristy where Ann Lee was educated is Yale."]
    keys = value_slots(lines)
    assert keys[0] == keys[1] == "v:bill frisell|_ univeristy where|was educated is"
    assert keys[2] == "v:ann lee|_ univeristy where|was educated is"
    assert slot_of(lines[0]) != slot_of(lines[1])


def test_a_name_closing_the_sentence_is_the_value():
    lines = ["the door was painted by Alice.", "the door was painted by Bob.", "the gate was painted by Carol."]
    keys = value_slots(lines)
    assert keys == ["v:the door||was painted by", "v:the door||was painted by", "v:the gate||was painted by"]


def test_a_bare_attribute_keeps_the_public_slot():
    lines = ["Alice is a nurse.", "Alice is 30 years old.", "Bob is a pilot."]
    assert line_relation(lines[0]) is None and line_relation(lines[1]) is None
    assert value_slots(lines) == [slot_of(x) for x in lines]


def test_a_line_without_a_relation_word_keeps_the_public_slot():
    lines = ["Alice smiled.", "Bob smiled."]
    assert [line_relation(x) for x in lines] == [None, None]
    assert value_slots(lines) == [slot_of(x) for x in lines]


def test_a_relation_of_one_subject_keeps_the_public_slot():
    lines = ["User lives in Paris.", "User lives in rome.", "User works at Acme.", "User works at globex."]
    assert value_slots(lines) == [slot_of(x) for x in lines]


def test_many_lines_per_changed_subject_read_as_events():
    places = ["store", "park", "gym", "office", "garden"]
    lines = ["%s went to the %s." % (who, p) for who in ("Alice", "Bob") for p in places]
    assert value_slots(lines) == [slot_of(x) for x in lines]


def test_native_read_shows_the_newest_value_only():
    mem = muster.build(text(), models=muster.Models.fake(dim=16))
    assert mem.plan.branch == "numbered"
    shown = mem.context("Which sport does Alice Reed play?", U=len(mem)).text
    assert "Alice Reed plays the sport of golf." in shown and "Alice Reed lives in Rome." in shown
    assert "Alice Reed plays the sport of tennis." not in shown and "Alice Reed lives in Paris." not in shown


def test_exact_cut_keeps_the_harness_slot():
    plan = muster.derive(text())
    assert plan.branch == "numbered"
    assert cut(text(), plan, exact=True).parent == tuple("s:" + slot_of(b) for b in bodies())
    native = cut(text(), plan, exact=False).parent
    assert native == tuple("s:" + k for k in value_slots(bodies()))
    assert len(set(native)) == len(PEOPLE) * 2


def test_plain_text_and_records_take_the_same_slots():
    lines = text().split("\n")
    fake = muster.Models.fake(dim=16)
    assert muster.build("\n".join(lines), models=fake).parent == muster.build([dict(text=x) for x in lines], models=fake).parent


def test_the_store_reads_the_same_slots_as_build(tmp_path):
    store = muster.Store(tmp_path / "s.db", models=muster.Models.fake(dim=16))
    store.add([dict(text=x) for x in text().split("\n")])
    assert store.head().parent == muster.build([dict(text=x) for x in text().split("\n")], models=muster.Models.fake(dim=16)).parent


@pytest.mark.parametrize("limit", [None, 640])
def test_a_serial_past_the_integer_limit_does_not_stop_the_library(limit):
    if limit and not hasattr(sys, "set_int_max_str_digits"):
        pytest.skip("this Python has no integer-string limit")
    before = sys.get_int_max_str_digits() if limit else None
    if limit:
        sys.set_int_max_str_digits(limit)
    try:
        huge = "9" * 5000
        lines = ["%d. %s" % (i, b) for i, b in enumerate(bodies())] + ["%s. Alice Reed lives in Lima." % huge]
        source = "\n".join(lines)
        assert muster.derive(source).branch == "numbered"
        mem = muster.build(source, models=muster.Models.fake(dim=16))
        shown = mem.context("Where does Alice Reed live?", U=len(mem)).text
        assert "Alice Reed lives in Lima." in shown and "Alice Reed lives in Rome." not in shown
    finally:
        if limit:
            sys.set_int_max_str_digits(before)


@pytest.mark.parametrize("line", [
    "Alice lives " + "," * 200 + " x in Paris.",
    "Alice lives " + "." * 200 + " x in the park",
    "Alice lives" + " " * 200 + "x in Paris.",
    "Alice lives in the of " + "of " * 20000 + "Paris",
    "of Alice a " + "Xx a " * 40000,
])
def test_a_long_or_odd_line_takes_linear_time(line):
    started = time.perf_counter()
    line_relation(line)
    value_slots([line, line.replace("Alice", "Bob")])
    assert time.perf_counter() - started < 2.0


def test_a_number_before_a_name_stays_with_the_name():
    keys = value_slots([
        "2009 Six Nations Championship is associated with the sport of rugby.",
        "2004 Six Nations Championship is associated with the sport of cricket.",
        "2009 Six Nations Championship is associated with the sport of tennis.",
        "1977 FA Cup Final is associated with the sport of football.",
    ])
    assert keys[0] == keys[2] == "v:2009 six nations championship|_|is associated with the sport"
    assert keys[1] == "v:2004 six nations championship|_|is associated with the sport"
    assert keys[3] == "v:1977 fa cup final|_|is associated with the sport"


def test_a_number_in_a_year_phrase_keeps_the_years_apart():
    keys = value_slots([
        "In 2014, the president of France is Hollande.",
        "In 2015, the president of France is Valls.",
        "In 2014, the president of Spain is Rajoy.",
    ])
    assert keys[0] != keys[1]
