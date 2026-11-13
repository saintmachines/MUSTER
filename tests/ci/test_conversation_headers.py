"""Source date inheritance does not confuse message bodies with source headers."""
import importlib
import json

import pytest

import muster


derive_module = importlib.import_module("muster.derive")


def exported_turns(date="March-15-2024"):
    parts = []
    for i in range(5):
        head = date + " | session,turn 1,1" if i == 0 else "session,turn 1,%d" % (i + 1)
        parts.append("%d. [%s] USER: Remember the amber cabinet." % (2 * i, head))
        parts.append("%d. ASSISTANT: Here are the details.\n\n## Details\n\n" % (2 * i + 1)
                     + "\n".join("padding: keep the ordinary body line %d" % k for k in range(20))
                     + "\n\n```python\nname: ordinary code content\n```\n\nThe cabinet is near the door.")
    return "\n".join(parts)


@pytest.mark.parametrize("date", ["March-15-2024", "March 15, 2024", "Mar. 15 2024", "15 March 2024",
                                  "8 May, 2023", "MARCH-15-2024", "2024-03-15"])
def test_session_header_date_carries_into_multiline_messages(date):
    source = exported_turns(date)
    assert derive_module._conversation_share(source) == 1
    plan = muster.derive_collection([source, source])
    assert (plan.branch, plan.policy, plan.render) == ("record", "flr", "r4")
    assert plan.U == 10


@pytest.mark.parametrize("date", ["March 2024", "Monday", "you may 3 of them", "session,turn 1,1"])
def test_incomplete_or_body_dates_do_not_supply_a_session_clock(date):
    source = exported_turns(date)
    assert derive_module._conversation_share(source) == 0
    assert (muster.derive(source).policy, muster.derive(source).render) == ("flrs", "flat")


@pytest.mark.parametrize("source", [
    "\n\n".join("text: review number %d mentions March 15, 2024\nlabel: positive" % i for i in range(20)),
    "\n\n".join("date: 2024-03-15\nuser: person%d\nrating: 5\nitem: amber cabinet" % i for i in range(20)),
    "[2024-03-15] Holmes: a quoted line.\n\n" + "\n\n".join("Holmes: ordinary quoted line %d" % i for i in range(40)),
])
def test_examples_fields_and_sparse_quoted_headers_keep_independent_recipe(source):
    assert derive_module._conversation_share(source) < .8
    plan = muster.derive(source)
    assert (plan.branch, plan.policy, plan.render) == ("record", "flrs", "flat")


def test_numbered_priority_is_unchanged_for_short_dated_speaker_lines():
    source = "\n".join("%d. [March-15-2024] USER: entry %d" % (i, i) for i in range(20))
    plan = muster.derive(source)
    assert (plan.branch, plan.policy, plan.render) == ("numbered", "attprio_sess", "numbered")


@pytest.mark.parametrize("stamp,conversation", [("March-15-2024", True), ("8 May, 2023", True),
                                                ("March 2024", False), ("you may 3 of them", False)])
def test_metadata_and_original_serialized_headers_agree(stamp, conversation):
    records = [muster.Record("The amber cabinet is near the door.", time=stamp, speaker="Alice"),
               muster.Record("The silver key is in that cabinet.", time=stamp, speaker="Bob")]
    raw = muster.serialize(records)[0]
    expected = ("flr", "r4") if conversation else ("flrs", "flat")
    assert (muster.derive(records).policy, muster.derive(records).render) == expected
    assert (muster.derive(raw).policy, muster.derive(raw).render) == expected


def test_cli_reports_the_actual_native_rounds_and_unpaired_turns(tmp_path, capsys):
    from muster.cli import main
    source = repr(["2024-03-15", [dict(role="assistant", content="A leading answer."),
                                  dict(role="user", content="A user turn."),
                                  dict(role="assistant", content="Its paired answer."),
                                  dict(role="assistant", content="An extra answer.")]])
    path = tmp_path / "sessions.txt"
    path.write_text(source)
    main(["derive", str(path), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["units"] == len(muster.build(source, embed=False)) == 3
    assert report["user_rounds"] == 1 and report["unpaired_turns"] == 2
    main(["derive", str(path)])
    assert "2 unpaired turns (3 units)" in capsys.readouterr().out
