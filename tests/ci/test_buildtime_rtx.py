"""harness/buildtime_rtx/ stays the driver of the reported build: its copied blocks equal the harness line by line, its stores
carry the reported legs' env, and its records name the files of this tree. No model, no data, no network."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "harness"
DRIVER = HARNESS / "buildtime_rtx"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_copied_blocks_equal_the_harness_line_by_line():
    r = subprocess.run([sys.executable, "-B", str(DRIVER / "verify_copies.py"), "--src", str(HARNESS)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("against the shipped source lines: OK") == 2


def test_the_registry_is_the_harness_file():
    assert (DRIVER / "cells_registry.py").read_bytes() == (HARNESS / "cells_registry.py").read_bytes()


def test_every_store_runs_the_env_of_its_reported_legs():
    legs = {row["tag"]: row["env"] for row in _json(HARNESS / "len_legs.json")["legs"]}
    stores = _json(DRIVER / "stores.json")["stores"]
    assert sorted(stores) == ["beam100k", "beam10m", "beam1m"]
    for store in stores.values():
        u20, u40 = (legs[tag] for tag in store["legs"])
        assert store["env"] == u20
        assert {k for k in u20 if u20[k] != u40[k]} == {"U"}
        assert [store["env"][k] for k in ("ATOM", "CHUNK", "MSCALE", "EMB_MAXLEN")] == ["blank", "512", "4096", "2048"]


def test_expected_names_the_files_of_this_tree():
    expected = _json(DRIVER / "expected.json")
    sources = {"lme_pf29.py": "lme_pf29.py", "derive_config.py": "derive_config.py", "core2.py": "core2.py", "cells_registry.py": "cells_registry.py",
               "real_e1_conflict_killer.py": "src/real_e1_conflict_killer.py", "len_legs.tsv": "len_legs.tsv", "len_legs.json": "len_legs.json"}
    assert {name: _sha(HARNESS / rel) for name, rel in sources.items()} == expected["harness"]
    assert {name: _sha(DRIVER / name) for name in expected["package_verbatim"]} == expected["package_verbatim"]
    assert _sha(DRIVER / "stores.json") == expected["inputs"]["stores.json"]["sha256"]
    for name, rec in _json(DRIVER / "MANIFEST_copies.json")["sources"].items():
        assert _sha(ROOT / rec["path"]) == rec["sha256"], name


def test_the_requirements_hold_the_versions_of_the_reported_run():
    pins = dict(re.findall(r"^([\w-]+)==([\d.]+)$", (DRIVER / "requirements.txt").read_text(encoding="utf-8"), re.M))
    report = (ROOT / "gate" / "build_time_rtx.md").read_text(encoding="utf-8")
    assert not [n for n in ("sentence-transformers", "transformers", "tokenizers", "rank-bm25") if f"{n} {pins[n]}" not in report]


def test_the_driver_reads_its_arguments_before_it_needs_torch():
    r = subprocess.run([sys.executable, "-B", str(DRIVER / "time_beam_build_rtx.py"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--out" in r.stdout, r.stdout + r.stderr
