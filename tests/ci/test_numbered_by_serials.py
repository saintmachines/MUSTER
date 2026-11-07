"""The serial fraction alone makes a numbered log, in derive, the command line and every serial style, as in 0.1.0. A log
whose lines never revise a slot is still numbered; newest-wins then replaces nothing."""
import json

import pytest

import muster
from muster import cli

STYLES = ["%d. %s", "Step %d: %s", "#%d %s", "[%d] %s"]


@pytest.mark.parametrize("style", STYLES)
def test_serials_without_updates_make_a_numbered_log(style):
    text = "\n".join(style % (i, "Person%d opened door%d." % (i, i)) for i in range(30))
    plan = muster.derive(text)
    assert plan.branch == "numbered" and plan.supersede == "numbered"
    assert not any("update share" in line for line in plan.trace)


def test_the_command_line_shows_no_update_numbers(tmp_path, capsys):
    f = tmp_path / "log.txt"
    f.write_text("\n".join("%d. Person%d opened door%d." % (i, i, i) for i in range(30)), encoding="utf-8")
    cli.main(["derive", str(f), "--json"])
    d = json.loads(capsys.readouterr().out)
    assert "update_share" not in d and "lines_per_changed_slot" not in d
