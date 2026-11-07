"""A numbered log under its own serial spelling ("[step 3] body") keeps the paper's reading in native mode, as in 0.1.0:
the paper's slot and no value walk. Both native numbered-log rules (exact.md N8, N9) read dot-serial logs only."""
import muster
from muster.cut import slot_of

FAKE = muster.Models.fake(16)
PEOPLE = ["Alice Reed", "Bob Stone", "Carol Hart", "Dan Cole", "Erin Ford", "Finn Page"]
PLACES = ["kitchen", "hallway", "office", "garden"]
CHAIN = ["Steve Sax is associated with the sport of association football.",
         "association football was created in the country of Italy.",
         "The capital of Italy is Rome."]
FILLER = ["%s was born in the city of %s." % (a, b) for a, b in (
    ("Mira Ona", "Lund"), ("Pavel Rusk", "Oslo"), ("Hana Sol", "Bern"), ("Ivo Tamm", "Riga"), ("Lena Voss", "Graz"),
    ("Omar Wynn", "Kiel"), ("Rita Yule", "Pisa"), ("Sven Zorn", "Turin"), ("Tove Arden", "Porto"), ("Uma Blake", "Cork"),
    ("Vera Crane", "Ghent"), ("Wim Doyle", "Malmo"), ("Xena Ellis", "Bari"), ("Yuri Fane", "Bonn"), ("Zoe Grant", "Basel"))]


def moves():
    # three moves each to lower-case places: the subject-relation slot would keep only each person's last move
    return ["%s moved to the %s." % (who, PLACES[(j + k) % 4]) for k in range(3) for j, who in enumerate(PEOPLE)]


def build(fmt, bodies):
    return muster.build("\n".join(fmt % (i + 1, b) for i, b in enumerate(bodies)), models=FAKE)


def test_a_step_log_keeps_the_papers_slot():
    lead, dot = build("[step %d] %s", moves()), build("%d. %s", moves())
    assert (lead.plan.branch, lead.plan.serial) == ("numbered", "lead"), lead.plan.trace
    assert (dot.plan.branch, dot.plan.serial) == ("numbered", "dot"), dot.plan.trace
    assert list(lead.parent) == ["s:" + slot_of(b) for b in moves()]
    assert any(p.startswith("s:v:") for p in dot.parent)  # the same lines under a dot serial take the subject-relation slot


def test_a_step_log_is_not_walked():
    q = "What is the capital of the country where Steve Sax's sport was created?"
    bodies = CHAIN + FILLER + [f.replace(" of ", " of New ") for f in FILLER[:4]]
    lead, dot = build("[step %d] %s", bodies), build("%d. %s", bodies)
    assert lead.plan.serial == "lead" and dot.plan.serial == "dot"
    assert dot.search(q).heads["walk"]  # the walk is live on the dot form of the same lines
    sel = lead.search(q)
    assert sel.heads["walk"] == [] and "first" not in sel.heads
