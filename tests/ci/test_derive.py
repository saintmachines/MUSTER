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
    out["numbered"] = "\n".join("%d. %s is %s" % (i, rnd.choice(words).capitalize(), sent(5)) for i in range(1, 900))
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
        assert got.out == ref.stdout
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
    assert muster.derive("1. a b\n2. c d\n3. e f\n4. g h\nplain line").branch == "numbered"
    assert muster.derive("1. a b\n2. c d\n3. e f\nplain\nplain").branch != "numbered"


def test_records_input():
    recs = [muster.Record(text="I moved to Paris in May.", id="D1:%d" % i, time="2023-05-08", speaker="Caroline")
            for i in range(50)]
    plan = muster.derive(recs)
    assert plan.branch == "record" and plan.chunk == 512
    text, spans = muster.serialize(recs)
    assert text.split("\n\n")[0] == "D1:0 [2023-05-08] Caroline: I moved to Paris in May."
    assert spans[1][0] == len(text.split("\n\n")[0]) + 2
