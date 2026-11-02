"""`muster verify cuts` on a synthetic data directory: equal tables pass, a changed store fails, a file whose hash does
not match is skipped rather than compared."""
import hashlib
import json
import os

import pytest

from muster import presets
from muster.compat.verify import row_signature, verify_cuts

pd = pytest.importorskip("pandas")


def _write(tmp_path, context):
    rel = presets.legs()["lz_shqa"]["parq"]
    p = tmp_path / "data" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"context": [context]}).to_parquet(p)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / "data.sha256").write_text("%s  %s\n" % (sha, rel))
    return str(tmp_path / "data")


def test_verify_cuts_pass_fail_skip(tmp_path):
    text = "\n\n".join("D1:%d [2026-01-01] Ann: record number %d about Lisbon and the Atlas project." % (i, i)
                       for i in range(400))
    data = _write(tmp_path, text)
    table = {"legs": {"lz_shqa": {"0": row_signature(text, presets.get("lz_shqa"))}}}
    tp = tmp_path / "t.json"
    tp.write_text(json.dumps(table))
    lines = []
    assert verify_cuts(data, str(tp), legs={"lz_shqa"}, out=lines.append)
    table["legs"]["lz_shqa"]["0"]["units"] += 1
    tp.write_text(json.dumps(table))
    assert not verify_cuts(data, str(tp), legs={"lz_shqa"}, out=lines.append)
    table["legs"]["lz_shqa"]["0"]["units"] -= 1
    table["legs"]["lz_shqa"]["0"]["text_sha256"] = "0" * 64       # another store text: skipped, never compared
    tp.write_text(json.dumps(table))
    lines = []
    assert not verify_cuts(data, str(tp), legs={"lz_shqa"}, out=lines.append)
    assert any("store text differs" in x for x in lines)
