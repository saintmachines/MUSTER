"""The original headline and native-library blocks each follow their named data.
Generated figures and text retain their sources and carry no private pattern."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from test_packaging import _private_patterns, _violations

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = (ROOT / "README.md", ROOT / "docs" / "data", ROOT / "docs" / "assets" / "readme",
          ROOT / "docs" / "assets" / "readme_page" / "MANIFEST.json")


def _run(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, timeout=120)


def test_generated_blocks_and_figures_are_current():
    result = _run(ROOT / "tools" / "make_readme.py", "--check")
    assert result.returncode == 0, result.stdout + result.stderr


def test_readme_and_its_data_carry_no_private_pattern():
    private = _private_patterns()
    files = [p for root in PUBLIC for p in ([root] if root.is_file() else sorted(root.rglob("*"))) if p.is_file()]
    assert len(files) >= 8
    bad = ["%s:%d" % (p.relative_to(ROOT).as_posix(), i)
           for p in files for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if _violations(line, private)]
    assert not bad, "privacy scan failed; values suppressed:\n" + "\n".join(bad)


def test_readme_pictures_are_the_rendered_page():
    # The README shows the project page's own panels, rendered by tools/make_readme_media.js. Its MANIFEST.json records
    # the headline data they were made from and each file's hash, so new numbers or a hand-edited picture fail here.
    folder = ROOT / "docs" / "assets" / "readme_page"
    manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
    named = set(re.findall(r'(?:src|srcset)="(docs/assets/[^"]+)"', (ROOT / "README.md").read_text(encoding="utf-8")))
    assert not [p for p in named if not (ROOT / p).is_file()], named
    assert {p for p in named if p.startswith("docs/assets/readme_page/")} == {"docs/assets/readme_page/" + f for f in manifest["files"]}
    data = hashlib.sha256((ROOT / "docs" / "data" / "headline.json").read_bytes()).hexdigest()
    assert manifest["inputs"]["docs/data/headline.json"] == data, "headline.json changed: run node tools/make_readme_media.js"
    for name, made in manifest["files"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == made["sha256"], name


def test_pages_and_citation_follow_the_headline_data():
    data = json.loads((ROOT / "docs" / "data" / "headline.json").read_text(encoding="utf-8"))
    build = next(r for r in data["build_10m"]["rows"] if r["system"] == "MUSTER")["build"]
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert build.replace(" ", "&nbsp;") in page and build in readme
    title = data["paper"]["title"]
    assert title in page and title in readme and title in citation
    assert "withheld" not in page and "withheld" not in readme
    assert "when the paper is public" not in page and "when the paper is public" not in readme


def test_every_statement_of_the_build_time_follows_the_headline_data():
    data = json.loads((ROOT / "docs" / "data" / "headline.json").read_text(encoding="utf-8"))
    build = next(r for r in data["build_10m"]["rows"] if r["system"] == "MUSTER")["build"].split()[0]
    stated = re.compile(r'MUSTER[ ,0-9.a-z]*? in (\d+) s\b|(\d+) s · 0 LLM calls|MUSTER takes (\d+) seconds|built in (\d+) seconds'
                        r'|<th scope="row">MUSTER</th>(?:<td class="n">[^<]*</td>){7}<td class="n">(\d+) s</td></tr>')
    files = [ROOT / "README.md", ROOT / "docs" / "index.html", *sorted((ROOT / "docs" / "assets").rglob("*.svg"))]
    texts = {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8").replace("&nbsp;", " ") for p in files}
    found = [(name, m[m.lastindex]) for name, text in texts.items() for m in stated.finditer(text)]
    # The R landing renders its charts from embedded canonical data. Keep the
    # ten static README/SVG checks and check every embedded value, rather than
    # requiring three copies of a phrase from the previous page template.
    assert len([f for f in found if f[0] != "docs/index.html"]) >= 10
    embedded = re.search(r'<script id="hd" type="application/json">(.*?)</script>',
                         texts["docs/index.html"], re.S)
    assert embedded, "the landing must include the canonical headline data"
    assert json.loads(embedded.group(1)) == data
    assert not [f for f in found if f[1] != build], found
    # The whole set of times in seconds, so a spelling the pattern above does not know still fails here.
    # An animation pacing explanation in a JS source comment is not a reported
    # build time. The unrounded value is covered by full embedded-data equality.
    seconds = {m[1] for text in texts.values()
               for m in re.finditer(r"(?<![\w.,])(\d+(?:\.\d+)?) ?(?:s|seconds)\b",
                                    re.sub(r"(?m)^\s*//[^\n]*", "", text))}
    assert seconds == {"1", "7", "8", "10", build,
                       str(data["mechanism"]["detect"]["seconds"])}


def test_landing_page_links_stay_inside_what_pages_serves():
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    assert 'href="../' not in page and 'src="../' not in page
    for local in re.findall(r'(?:href|src)="(assets/[^"#]+)"', page):
        assert (ROOT / "docs" / local).is_file(), local
    for target in re.findall(r'https://github\.com/saintmachines/MUSTER/(?:blob|tree)/main/([^"#]+)', page):
        assert (ROOT / target).exists(), target


@pytest.mark.skipif(not os.environ.get("MUSTER_PAPER_TEX"), reason="MUSTER_PAPER_TEX names the paper's tex source")
def test_headline_cells_equal_the_paper():
    result = _run(ROOT / "tools" / "check_headline_tex.py", os.environ["MUSTER_PAPER_TEX"])
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not os.environ.get("MUSTER_PAPER_TEX"), reason="MUSTER_PAPER_TEX names the paper's tex source")
def test_every_readme_number_is_in_the_paper_or_a_named_source():
    result = _run(ROOT / "scripts" / "check_readme_numbers.py", ROOT / "README.md", os.environ["MUSTER_PAPER_TEX"],
                  ROOT / "docs" / "data" / "headline.json", ROOT / "gate" / "stage0d_render.md",
                  ROOT / "docs" / "data" / "native_mab_0_1_2.json")
    assert result.returncode == 0, result.stdout + result.stderr
