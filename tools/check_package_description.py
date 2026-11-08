#!/usr/bin/env python3
"""Render and verify the source/wheel/sdist descriptions before PyPI publication.

Requires readme_renderer[md]==46.0. Uses only public GET requests and writes
rendered HTML/receipts to the explicitly selected report directory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
from html.parser import HTMLParser
import importlib.metadata
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urldefrag, urlsplit
from urllib.request import Request, urlopen
import zipfile


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Surface(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []
        self.targets: set[str] = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.targets.add(values["id"])
        if tag == "a":
            if values.get("name"):
                self.targets.add(values["name"])
            if "href" in values:
                self.hrefs.append(values["href"] or "")


def surface(html: str) -> Surface:
    parsed = Surface()
    parsed.feed(html)
    return parsed


def description(raw: bytes, expected: str) -> str:
    meta = BytesParser().parsebytes(raw)
    if meta.get("Name") != "muster-memory" or meta.get("Version") != expected:
        raise ValueError("Distribution name or version does not match the reviewed release")
    if not (meta.get("Description-Content-Type") or "").lower().startswith("text/markdown"):
        raise ValueError("Distribution description must declare text/markdown")
    payload = meta.get_payload(decode=True)
    if payload is None:
        raise ValueError("Distribution has no long-description body")
    return payload.decode("utf-8")


def normalized(text: str) -> str:
    # Tolerate only transport line endings and trailing newlines.
    return text.replace("\r\n", "\n").rstrip("\n") + "\n"


def sources(readme: Path, dist: Path, expected: str) -> dict[str, tuple[str, str]]:
    wheels, sdists = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Require exactly one wheel and one sdist")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("Require one wheel METADATA member")
        wheel = description(archive.read(names[0]), expected)
    with tarfile.open(sdists[0], "r:gz") as archive:
        names = [m for m in archive.getmembers() if m.isfile() and m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")]
        if len(names) != 1:
            raise ValueError("Require one top-level sdist PKG-INFO member")
        sdist = description(archive.extractfile(names[0]).read(), expected)
    return {"source": (str(readme), readme.read_text(encoding="utf-8")),
            "wheel": (wheels[0].name, wheel), "sdist": (sdists[0].name, sdist)}


def check_destination(href: str) -> dict:
    url, fragment = urldefrag(href)
    result = {"href": href, "request_url": url, "fragment": unquote(fragment), "method": "GET"}
    for attempt in range(1, 4):
        try:
            request = Request(url, headers={"User-Agent": "muster-memory-description-gate/0.1", "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"})
            with urlopen(request, timeout=30) as response:
                result.update(status=response.status, final_url=response.url,
                              content_type=response.headers.get("Content-Type", ""))
                body = response.read(12 * 1024 * 1024 + 1)
                if len(body) > 12 * 1024 * 1024:
                    raise ValueError("Destination exceeds the explicit 12 MiB inspection limit")
                charset = response.headers.get_content_charset() or "utf-8"
            if not 200 <= result["status"] < 300:
                raise ValueError("Destination is not successful")
            result.update(response_sha256=digest(body), bytes=len(body), attempts=attempt)
            if fragment:
                if "html" not in result["content_type"].lower():
                    raise ValueError("A fragment destination is not HTML")
                targets = surface(body.decode(charset, errors="strict")).targets
                fragment = unquote(fragment)
                candidates = [fragment]
                # GitHub namespaces rendered Markdown heading IDs this way.
                if urlsplit(result["final_url"]).hostname == "github.com":
                    candidates.append("user-content-" + fragment)
                matches = [value for value in candidates if value in targets]
                if not matches:
                    result["fragment_validation"] = "unverified_in_fetched_HTML"
                    raise ValueError("Destination fragment not verified in fetched response HTML: " + fragment)
                result["matched_target"] = matches[0]
            result["passed"] = True
            return result
        except HTTPError as error:
            result.update(status=error.code, error=str(error))
            if error.code in {429, 502, 503, 504} and attempt < 3:
                time.sleep(2)
                continue
            break
        except (URLError, TimeoutError, ValueError, UnicodeError) as error:
            result["error"] = str(error)
            break
    result["passed"] = False
    return result


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, required=True)
    cli.add_argument("--dist", type=Path, required=True)
    cli.add_argument("--expected-version", required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "status": "failed", "documents": [], "destinations": [], "issues": [],
              "new_model_paid_or_GPU_calls": 0, "publication_performed": False,
              "scope": "Rendered long-description and live href/anchor verification only"}
    try:
        import readme_renderer.markdown as markdown
        versions = {}
        for name in ["readme_renderer", "comrak", "cmarkgfm", "nh3", "docutils"]:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        report["tool_versions"] = versions
        if versions.get("readme_renderer") != "46.0":
            raise ValueError("Install the reviewed readme_renderer[md]==46.0")
        documents = sources(args.source, args.dist, args.expected_version)
        baseline = normalized(documents["source"][1])
        external = set()
        for label, (origin, text) in documents.items():
            warnings = io.StringIO()
            rendered = markdown.render(text, stream=warnings, variant="GFM")
            if not rendered:
                raise ValueError("Actual Markdown rendering failed for " + label + ": " + warnings.getvalue())
            (args.out / (label + ".rendered.html")).write_text(rendered, encoding="utf-8")
            parsed = surface(rendered)
            item = {"kind": label, "origin": origin, "description_sha256": digest(text.encode()),
                    "normalized_description_sha256": digest(normalized(text).encode()),
                    "rendered_html_sha256": digest(rendered.encode()), "rendered_hrefs": parsed.hrefs,
                    "anchor_targets": sorted(parsed.targets), "renderer_warnings": warnings.getvalue()}
            report["documents"].append(item)
            if normalized(text) != baseline:
                report["issues"].append({"kind": "description_mismatch", "document": label})
            for href in parsed.hrefs:
                parsed_url = urlsplit(href)
                if href.startswith("#"):
                    fragment = unquote(href[1:])
                    if fragment and fragment not in parsed.targets:
                        report["issues"].append({"kind": "missing_local_anchor", "document": label, "href": href})
                elif parsed_url.scheme in {"http", "https"} and parsed_url.netloc:
                    external.add(href)
                elif parsed_url.scheme == "mailto" and parsed_url.path:
                    pass
                else:
                    report["issues"].append({"kind": "relative_or_invalid_href", "document": label, "href": href})
        # Fail locally before making requests when the source or rendered links are invalid.
        if not report["issues"]:
            with ThreadPoolExecutor(max_workers=4) as pool:
                report["destinations"] = list(pool.map(check_destination, sorted(external)))
            report["issues"].extend({"kind": "destination_check_failed", **row} for row in report["destinations"] if not row["passed"])
        report["status"] = "passed" if not report["issues"] else "failed"
    except Exception as error:
        report["issues"].append({"kind": "gate_error", "error": str(error)})
    report["source_wheel_sdist_checked"] = len(report["documents"]) == 3
    report["unique_live_destination_checks"] = len(report["destinations"])
    (args.out / "DESCRIPTION_GATE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "documents": len(report["documents"]),
                      "destination_checks": len(report["destinations"]), "issues": report["issues"]}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
