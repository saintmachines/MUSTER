"""Regression: a long run of whitespace costs linear time, and the patterns rewritten for it match what they matched before."""
import random
import re
import time

import pytest

import muster
from muster.cut import _CAPRE
from muster.index import _CAP
from muster.stats import LEAD

# The patterns as the paper's harness wrote them. One run of whitespace could go to either of two adjacent repeats.
HARNESS_NAME = re.compile(r"\b([A-Z][\w.'-]+(?:\s+(?:of|the|and|de|van|von|di)?\s*[A-Z][\w.'-]+)*)\b")
HARNESS_LEAD = re.compile(r"^(\s*[\[(#]?\s*(?:[A-Za-z]+\s*)?)(\d{1,8})([\]).:]?\s+)(?=\S)")
PIECES = ["Alice", "Bob", "A", "ab", "of", "the", "and", "de", "van", "von", "di", " ", "  ", "\t", "\n", " ", " ", ".", ",", "'", "-",
          "[", "(", "#", "]", ")", ":", "7", "42", "123456789", "x", "O'Neil", "J.R.", "_", "٣"]


def spans(pattern, text):
    return [(m.start(), m.end(), m.groups()) for m in pattern.finditer(text)]


def test_the_rewritten_patterns_match_what_they_matched_before():
    rng = random.Random(20261107)
    for _ in range(40000):
        text = "".join(rng.choice(PIECES) for _ in range(rng.randint(0, 16)))
        assert spans(_CAPRE, text) == spans(HARNESS_NAME, text) == spans(_CAP, text), repr(text)
        old, new = HARNESS_LEAD.match(text), LEAD.match(text)
        assert (old is None) == (new is None), repr(text)
        if old:
            assert (old.span(), old.groups()) == (new.span(), new.groups()), repr(text)


RUN = 200000


@pytest.mark.parametrize("line", [
    "Alice" + " " * RUN + "lives in Rome",
    "Alice" + "\t" * RUN + "lives in Rome",
    " " * RUN + "Alice lives in Rome",
    "[" + " " * RUN + "Alice lives in Rome",
], ids=["spaces after a name", "tabs after a name", "spaces first", "bracket then spaces"])
@pytest.mark.parametrize("shape", ["numbered", "prose"])
def test_a_long_run_of_whitespace_takes_linear_time(shape, line):
    if shape == "numbered":
        source = "\n".join(["%d. Person%d lives in City%d." % (i, i % 40, i) for i in range(60)] + ["60. " + line])
    else:
        source = "Plain prose about a person.\n\n" + line + "\n\nMore plain prose about a person."
    started = time.perf_counter()
    plan = muster.derive(source)
    mem = muster.build(source, plan, models=muster.Models.fake(dim=16))
    mem.context("Where does Alice live?")
    assert time.perf_counter() - started < 10.0
