"""Release checks that examine the built wheel and exercise it outside the checkout."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("muster", "tests", "tools", "examples", "docs/library")
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".hypothesis", "dist", "build"}
FORBIDDEN_IMPORTS = {"harness", "openai", "anthropic", "requests"}
PATTERNS = {
    "home path": re.compile(r"(?:/" + r"Users|/" + r"home)/[^/\s\"'<>]+"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b"),
    "provider key": re.compile(r"\bsk-(?:(?:proj|ant)-)?[A-Za-z0-9_-]{20,}\b"),
    "cloud key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "repository key": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def _run(args, *, cwd, timeout=60):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    assert result.returncode == 0, "subprocess failed: " + result.stdout[-2000:] + result.stderr[-2000:]
    return result


def _package_files():
    return {p.relative_to(ROOT).as_posix(): p for p in (ROOT / "muster").rglob("*") if p.is_file()
            and not (set(p.parts) & SKIP_DIRS) and p.suffix not in (".pyc", ".pyo") and p.name != ".DS_Store"}


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    dest = tmp_path_factory.mktemp("wheel")
    # The default build produces an sdist, then builds the wheel from that sdist.
    _run([sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dest)], cwd=ROOT)
    wheels = list(dest.glob("*.whl"))
    sdists = list(dest.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1
    return wheels[0], sdists[0]


@pytest.fixture(scope="module")
def wheel(distributions):
    return distributions[0]


def test_wheel_contains_exactly_package_sources(wheel):
    expected = _package_files()
    with zipfile.ZipFile(wheel) as archive:
        files = [n for n in archive.namelist() if not n.endswith("/")]
        assert len(files) == len(set(files)), "wheel contains duplicate names"
        actual = {n for n in files if n.startswith("muster/")}
        assert actual == set(expected), "wheel/source member mismatch"
        assert all(".dist-info/" in n for n in set(files) - actual), "non-package file in wheel"
        for name in files:
            assert not name.startswith("/") and ".." not in Path(name).parts
        for name, source in expected.items():
            assert not source.is_symlink(), "symlink in package: " + name
            assert archive.read(name) == source.read_bytes(), "wheel/source bytes differ: " + name
        assert "muster/presets/paper.json" in actual


def test_sdist_contains_only_library_documentation_and_metadata(distributions):
    expected = dict(_package_files())
    for path in (ROOT / "docs/library").rglob("*"):
        if path.is_file() and not (set(path.parts) & SKIP_DIRS) and path.name != ".DS_Store":
            expected[path.relative_to(ROOT).as_posix()] = path
    # hatchling always ships the root .gitignore in an sdist (it reuses it to exclude files when the wheel is
    # built from the sdist), so it is part of the contract
    expected.update({name: ROOT / name for name in (".gitignore", "LICENSE", "NOTICE", "README.md", "pyproject.toml")})
    with tarfile.open(distributions[1], "r:gz") as archive:
        members = archive.getmembers()
        assert not any(m.issym() or m.islnk() for m in members), "link in sdist"
        files = [m for m in members if m.isfile()]
        assert all(not m.name.startswith("/") and ".." not in Path(m.name).parts for m in files)
        actual = {m.name.split("/", 1)[1]: m for m in files}
        assert len(actual) == len(files), "duplicate name in sdist"
        assert set(actual) == set(expected) | {"PKG-INFO"}, "sdist/source member mismatch"
        for name, path in expected.items():
            assert archive.extractfile(actual[name]).read() == path.read_bytes(), "sdist bytes differ: " + name


def _private_patterns():
    filename = os.environ.get("MUSTER_PRIVATE_PATTERNS")
    if not filename:
        return []
    path = Path(filename).resolve()
    assert not path.is_relative_to(ROOT), "private patterns must stay outside the checkout"
    assert path.is_file(), "private pattern file is missing"
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]


def _violations(text, private=()):
    found = [name for name, pattern in PATTERNS.items() if pattern.search(text)]
    if any(pattern in text for pattern in private):
        found.append("private pattern")
    return found


def test_wheel_metadata_and_contents_have_no_private_patterns(wheel):
    private = _private_patterns()
    with zipfile.ZipFile(wheel) as archive:
        violations = [(name, _violations(archive.read(name).decode("utf-8", errors="replace"), private))
                      for name in archive.namelist() if not name.endswith("/")]
    assert not [(name, kinds) for name, kinds in violations if kinds], "private pattern in wheel; values suppressed"


def test_optional_private_pattern_file_stays_outside_the_checkout(tmp_path, monkeypatch):
    path = tmp_path / "private-patterns.txt"
    marker = "synthetic " + "identifier 9427"
    path.write_text("# one literal per line\n" + marker + "\n", encoding="utf-8")
    monkeypatch.setenv("MUSTER_PRIVATE_PATTERNS", str(path))
    assert _private_patterns() == [marker]
    assert _violations(marker, _private_patterns()) == ["private pattern"]
    monkeypatch.setenv("MUSTER_PRIVATE_PATTERNS", str(ROOT / "pyproject.toml"))
    with pytest.raises(AssertionError, match="outside the checkout"):
        _private_patterns()


def test_public_sources_contain_no_private_patterns():
    private = _private_patterns()
    violations = []
    for folder in SCAN_ROOTS:
        for path in (ROOT / folder).rglob("*"):
            if set(path.parts) & SKIP_DIRS or path.suffix in (".pyc", ".pyo") or path.name == ".DS_Store":
                continue
            relative = path.relative_to(ROOT).as_posix()
            assert not path.is_symlink(), "symlink in public source: " + relative
            if not path.is_file():
                continue
            # Do not read a credential file if one is accidentally placed under a scanned tree.
            assert not (path.name.startswith(".env") or path.suffix in (".key", ".pem")
                        or path.name == ".keys.sh"), "credential-shaped filename: " + relative
            text = path.read_bytes().decode("utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), 1):
                for kind in _violations(line, private):
                    violations.append("%s:%d (%s)" % (relative, number, kind))
    assert not violations, "privacy scan failed; values suppressed:\n" + "\n".join(violations)


@pytest.mark.parametrize("kind", list(PATTERNS) + ["private pattern"])
def test_privacy_scanner_has_positive_controls(kind):
    examples = {
        "home path": "/" + "Users" + "/fixture/project",
        "email": "fixture" + chr(64) + "example.invalid",
        "provider key": "sk-" + "x" * 40,
        "cloud key": "AKIA" + "A" * 16,
        "repository key": "ghp_" + "x" * 36,
        "private key": "-----BEGIN " + "PRIVATE KEY-----",
        "private pattern": "fixture marker 9427",
    }
    assert kind in _violations(examples[kind], private=[examples["private pattern"]])
    assert not _violations("A public synthetic memory fixture.")


def test_package_has_no_direct_provider_or_harness_imports():
    violations = []
    for name, path in _package_files().items():
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            for module in modules:
                if module.split(".")[0] in FORBIDDEN_IMPORTS:
                    violations.append("%s:%d" % (name, node.lineno))
    assert not violations, "forbidden direct import: " + ", ".join(violations)


def test_harness_oracle_is_available_from_checkout():
    import oracle

    assert Path(oracle.PF29).is_file()
    ns = oracle.cut_namespace(ATOM="newline", CHUNK=512)
    assert ns["_atom_split"]("first\n\nsecond", "newline") == ["first", "second"]
    assert callable(ns["load_row"])


def test_installed_wheel_runs_without_checkout_or_provider_imports(wheel, tmp_path):
    target = tmp_path / "installed"
    _run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile", "--target", str(target), str(wheel)],
         cwd=tmp_path)
    script = r'''
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
class BlockProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'harness', 'openai', 'anthropic', 'requests', 'agents', 'langgraph',
                                     'langchain_core', 'torch', 'sentence_transformers'}:
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockProviders())
import muster
from muster.models import Models
assert Path(muster.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert muster.presets.names()
with muster.Store.open('wheel.muster', models=Models.fake()) as store:
    store.add({'text': 'The public fixture is in the blue cabinet.'})
    context = store.context('Where is the public fixture?')
    output = context.text
    assert 'blue cabinet' in output
    print(json.dumps({'context': output, 'style': context.style, 'version': store.version}))
'''
    result = _run([sys.executable, "-I", "-c", script, str(target)], cwd=tmp_path)
    receipt = json.loads(result.stdout)
    assert receipt["version"] == 1 and receipt["style"] == "flat"
    assert "blue cabinet" in receipt["context"]
