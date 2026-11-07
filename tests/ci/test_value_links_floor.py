"""A subject leads nowhere only when it holds more than three slots and more than 5% of them, the floor the entity graph
uses for hubs. A small log about one person still walks."""
from muster.index import ValueLinks


def links(lines):
    return ValueLinks(list(lines), {i: i for i in range(len(lines))})


ALICE = ["Alice is a citizen of Place%d." % i for i in range(10)]
OTHERS = ["Bob%d is a citizen of Town%d." % (i, i) for i in range(100)]


def test_three_slots_of_one_subject_still_lead():
    assert ("alice",) in links(ALICE[:3]).slots_of


def test_a_fourth_slot_in_a_small_log_makes_a_hub():
    assert ("alice",) not in links(ALICE[:4]).slots_of


def test_in_a_log_of_100_slots_the_cut_is_5_percent():
    assert ("alice",) in links(ALICE[:5] + OTHERS[:95]).slots_of
    assert ("alice",) not in links(ALICE[:6] + OTHERS[:94]).slots_of
