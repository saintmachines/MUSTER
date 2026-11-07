"""A log numbered in its own style ("Step 3: ...", "#3 ...") is read as a numbered log; the paper's stores are not."""
import json

import muster
from muster.cli import main as cli_main
from muster.cut import cut, slot_of, value_slots

NAMES = ["Mary", "John", "Sandra", "Daniel", "Ines", "Omar", "Priya", "Tomas", "Yuki", "Zofia", "Lena", "Marco", "Nadia",
         "Pavel", "Rosa"]
CITIES = ["Lisbon", "Porto"]


def _log(fmt, n=30, start=0):
    # Each person's city is stated, then revised: the update evidence a numbered plan needs.
    return "\n".join(fmt % (start + i, "%s lives in %s." % (NAMES[i % 15], CITIES[i // 15 % 2])) for i in range(n))


def test_step_spelling_is_numbered_and_shown_as_written():
    text = _log("Step %d: %s")
    plan = muster.derive(text)
    assert plan.branch == "numbered" and plan.serial == "lead" and plan.policy == "attprio_sess"
    c = cut(text, plan)
    assert c.order == tuple(range(30)) and c.units[0] == "Mary lives in Lisbon."
    assert c.prefixes[3] == "Step 3: "
    mem = muster.build(text, embed=False)
    assert mem.render([3, 1]).text == "Step 1: John lives in Lisbon.\nStep 3: Daniel lives in Lisbon."


def test_a_short_log_counts_line_pairs():
    assert muster.derive(_log("(%d) %s", n=18)).serial == "lead"      # 17 of 17 pairs step by one


def test_same_records_cut_alike_under_both_spellings():
    # Units and order match. The log's own spelling keeps the paper's slot; a dot-serial log takes the native
    # subject-relation slot (exact.md N8), which here retires the same lines.
    a, b = _log("#%d %s"), _log("%d. %s")
    ca, cb = cut(a, muster.derive(a)), cut(b, muster.derive(b))
    assert (ca.units, ca.order) == (cb.units, cb.order)
    assert ca.parent == tuple("s:" + slot_of(u) for u in ca.units)
    assert cb.parent == tuple("s:" + k for k in value_slots(list(cb.units)))
    assert len(set(ca.parent)) == len(set(cb.parent))
    assert cb.prefixes is None


def test_every_listed_style_is_read_alike():
    for fmt in ("Step %d: %s", "#%d %s", "(%d) %s", "[%d] %s", "[step %d] %s"):
        plan = muster.derive(_log(fmt))
        assert plan.branch == "numbered" and plan.serial == "lead", fmt


def test_dot_plans_keep_their_json():
    plan = muster.derive(_log("%d. %s", start=1))
    assert plan.serial == "dot" and "serial" not in plan.to_dict()
    lead = muster.derive(_log("Step %d: %s"))
    assert lead.to_dict()["serial"] == "lead"
    for p in (plan, lead):
        assert muster.Plan.from_dict(json.loads(p.to_json())).plan_id == p.plan_id


def test_long_lines_session_keys_dates_and_resets_stay_off():
    chapters = "\n".join("[%d] Chapter %d %s" % (i, i, "the long road home " * 8) for i in range(1, 40))
    sessions = "\n".join("D1:%d Caroline: hello there, how are you?" % i for i in range(1, 40))
    dates = "\n".join("2023-05-%02d Mary moved to the kitchen." % (i % 28 + 1) for i in range(60))
    resets = "\n".join("Step %d: Mary moved to the kitchen." % (i % 5) for i in range(40))   # 32 of 39 pairs
    for text in (chapters, sessions, dates, resets):
        assert muster.derive(text).serial == "dot" and muster.derive(text).branch != "numbered"


def test_cli_derive_reports_the_spelling(tmp_path, capsys):
    p = tmp_path / "log.txt"
    p.write_text(_log("#%d %s"), encoding="utf-8")
    try:
        cli_main(["derive", str(p), "--json"])
    except SystemExit as e:
        assert not e.code
    d = json.loads(capsys.readouterr().out)
    assert d["store"] == "numbered log" and d["leading_serial_share"] >= 0.95 and d["median_line_tokens"] <= 20
