"""A dot serial keeps its place however long it is. Up to 600 digits it is its number; a longer one is keyed by its digits
read as one big-endian integer, which sorts like its number, ties with no other serial and prints as its digits, in time
linear in its length. Before, every serial past 600 digits read as one value, so a newer line no longer replaced an older
one and a log written newest first stopped being numbered (found by Codex on candidate v12)."""
import time

import pytest

import muster
from muster.cut import serial_text, serial_value

FAKE = muster.Models.fake(16)
NAMES = ["Ada", "Ben", "Cyd", "Dee", "Eli", "Fay", "Gus", "Hal", "Ivy", "Jon"]
OLD = ["Rome", "Oslo", "Bern", "Riga", "Graz", "Kiel", "Pisa", "Lund", "Bonn", "Lyon"]
NEW = ["Nice", "Cork", "Bari", "Ghent", "Leeds", "Perth", "Tampa", "Reno", "Nome", "Delft"]
BODIES = ["%s lives in %s." % (n, c) for n, c in zip(NAMES, OLD)] + ["%s lives in %s." % (n, c) for n, c in zip(NAMES, NEW)]


def serials(width):
    return ["1" + "%0*d" % (width - 1, k) for k in range(len(BODIES))]


def test_serials_up_to_600_digits_are_their_numbers():
    assert serial_value("0042") == 42 and serial_value("9" * 600) == 10 ** 600 - 1
    assert serial_value("\u0661\u0662") == 12
    assert serial_text(42) == "42" and serial_text(-1) == "-1" and serial_text(10 ** 600 - 1) == "9" * 600


def test_longer_serials_sort_by_their_number_and_never_tie():
    xs = ["1" + "0" * 600, "1" + "0" * 599 + "1", "2" + "0" * 600, "1" + "0" * 700, "0" * 50 + "3" + "0" * 650, "9" * 600]
    keys = [serial_value(x) for x in xs]
    by_number = sorted(range(len(xs)), key=lambda i: (len(xs[i].lstrip("0")), xs[i].lstrip("0")))
    assert sorted(range(len(xs)), key=lambda i: keys[i]) == by_number
    assert len(set(keys)) == len(xs)
    assert [serial_text(k) for k in keys] == [x.lstrip("0") for x in xs]


def test_leading_zeros_and_other_scripts_past_600_digits():
    assert serial_value("0" * 700 + "42") == 42
    arabic = "".join(chr(0x0660 + int(d)) for d in "1" + "0" * 700)
    assert serial_value(arabic) == serial_value("1" + "0" * 700)


@pytest.mark.parametrize("width", [601, 5000])
@pytest.mark.parametrize("newest_first", [False, True])
def test_a_newer_long_serial_replaces_an_older_line(width, newest_first):
    lines = ["%s. %s" % (s, b) for s, b in zip(serials(width), BODIES)]
    mem = muster.build("\n".join(reversed(lines) if newest_first else lines), models=FAKE)
    assert mem.plan.branch == "numbered"
    assert len(set(mem.cut.order)) == len(BODIES) and mem.cut.diagnostics["distinct_serial_share"] == 1.0
    text = mem.context("Where does Ada live?").text
    assert "[%s] Ada lives in Nice." % serials(width)[10] in text
    assert "Ada lives in Rome." not in text


def test_long_serials_read_and_print_in_linear_time():
    s = "7" * 2_000_000
    t = time.perf_counter()
    assert serial_text(serial_value(s)) == s
    assert time.perf_counter() - t < 1.0
