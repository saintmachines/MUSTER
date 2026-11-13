"""C4: derive and `muster derive` against scripts/derive.py, run as a subprocess on generated stores."""
import json
import os
import random
import subprocess
import sys

import pytest

import muster
from muster.cli import main as cli_main

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SD = os.path.join(ROOT, "scripts", "derive.py")


def _stores():
    rnd = random.Random(7)
    words = "alpha beta gamma delta river stone light Paris Maria engine report seven".split()
    sent = lambda n: " ".join(rnd.choice(words) for _ in range(n)).capitalize() + "."
    out = {}
    out["records"] = "\n\n".join("D%d:%d [2023-05-0%d] Ann: %s" % (i // 9, i, 1 + i % 9, sent(12)) for i in range(400))
    out["prose"] = " ".join(sent(rnd.randint(8, 30)) for _ in range(3000))
    out["numbered"] = "\n".join("%d. Unit%d is led by %s." % (i, i // 2, rnd.choice(words).capitalize()) for i in range(1, 900))
    out["mixed_crlf"] = out["records"][:20000].replace("\n", "\r\n")
    out["small"] = "one line only."
    out["giant_records"] = "\n\n".join(sent(2000) for _ in range(6))
    out["tabs"] = "\n\n".join("\t" + sent(30) + "\n \t \n" for _ in range(300))
    for k in range(6):
        n = rnd.randint(1, 4000)
        out["rand%d" % k] = ("\n\n" if k % 2 else " ").join(sent(rnd.randint(3, 40)) for _ in range(n))
    return out


@pytest.mark.parametrize("name", sorted(_stores()))
def test_cli_equals_scripts_derive(tmp_path, name, capsys):
    p = tmp_path / ("%s.txt" % name)
    p.write_text(_stores()[name], encoding="utf-8")
    for extra in ([], ["--json"]):
        ref = subprocess.run([sys.executable, SD, str(p)] + extra, capture_output=True, text=True)
        try:
            cli_main(["derive", str(p)] + extra)
            code = 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            if isinstance(e.code, str):
                print(e.code, file=sys.stderr)
        got = capsys.readouterr()
        if extra:
            # The historical oracle owns the original fields; the normal API
            # now also reports its actual policy, renderer and read budget.
            wanted, actual = json.loads(ref.stdout), json.loads(got.out)
            assert {key: actual[key] for key in wanted} == wanted
        else:
            historical = "\n".join(line for line in got.out.splitlines()
                                   if not line.startswith(("  policy", "  render", "  read"))) + "\n"
            assert historical == ref.stdout
        assert (code != 0) == (ref.returncode != 0)


@pytest.mark.parametrize("name", sorted(_stores()))
def test_plan_matches_scripts_derive(tmp_path, name):
    text = _stores()[name]
    p = tmp_path / "s.txt"
    p.write_text(text, encoding="utf-8")
    ref = subprocess.run([sys.executable, SD, str(p), "--json"], capture_output=True, text=True)
    plan = muster.derive(text)
    if ref.returncode:
        assert plan.branch == "empty"
        return
    d = json.loads(ref.stdout)
    want = {"numbered log": "numbered", "record store": "record", "prose": "prose"}[d["store"]]
    assert plan.branch == want
    if want != "numbered":
        assert plan.chunk == d["unit_tokens"]
        assert plan.mscale == (d["parent_tokens"] or 0)


def test_thresholds_are_inclusive():
    # 4 of 5 lines numbered = 0.80 exactly -> numbered; 0.75 -> not
    head = "1. Ann lives in Rome.\n2. Ann lives in Oslo.\n3. Bob lives in Rome."
    assert muster.derive(head + "\n4. Bob lives in Oslo.\nplain line").branch == "numbered"
    assert muster.derive(head + "\nplain\nplain").branch != "numbered"


def test_records_input():
    recs = [muster.Record(text="I moved to Paris in May.", id="D1:%d" % i, time="2023-05-08", speaker="Caroline")
            for i in range(50)]
    plan = muster.derive(recs)
    assert plan.branch == "record" and plan.chunk == 512
    text, spans = muster.serialize(recs)
    assert text.split("\n\n")[0] == "D1:0 [2023-05-08] Caroline: I moved to Paris in May."
    assert spans[1][0] == len(text.split("\n\n")[0]) + 2


@pytest.mark.parametrize("text", [
    "first record\n\nsecond record",
    "A long uninterrupted narrative. " * 1800,
    "[2026-10-01] Alice: first message.\n\n[2026-10-02] Bob: second message.",
    repr(["2026-10-01", [{"role": "user", "content": "first turn"},
                          {"role": "assistant", "content": "first answer"}]]),
    "\n".join("Step %d: Person%d lives in Place%d." % (i, i % 8, i % 3) for i in range(40)),
])
def test_cli_reports_the_actual_normal_api_recipe(tmp_path, capsys, text):
    path = tmp_path / "source.txt"
    path.write_text(text, encoding="utf-8")
    cli_main(["derive", str(path), "--json"])
    output = json.loads(capsys.readouterr().out)
    plan = muster.derive(text)
    assert (output["policy"], output["render"], output["U"], output["mode"], output["reference_chunk"]) == (
        plan.policy, plan.render, plan.U, plan.mode, plan.reference_chunk)
    if plan.branch == "rounds":
        memory = muster.build(text, embed=False)
        assert output["store"] == "conversation rounds" and output["units"] == len(memory)
        assert output["parents"] == 1


def test_cli_human_output_explains_source_session_rounds(tmp_path, capsys):
    text = repr(["2026-10-01", [{"role": "user", "content": "first turn"},
                                {"role": "assistant", "content": "first answer"}]] )
    path = tmp_path / "sessions.txt"
    path.write_text(text, encoding="utf-8")
    cli_main(["derive", str(path)])
    output = capsys.readouterr().out
    assert "conversation rounds" in output and "1 rounds" in output
    assert "flrs" in output and "flat" in output and "10 units per query" in output
