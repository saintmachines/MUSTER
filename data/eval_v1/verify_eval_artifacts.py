#!/usr/bin/env python3
"""Verify a MUSTER evaluation download offline; never invokes a model or API."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_rows(path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open(encoding="utf-8") as f:
        return json.load(f)["results"]


def verify(root, source_root=None):
    root = Path(root).resolve()
    with (root / "MANIFEST.tsv").open(encoding="utf-8", newline="") as f:
        manifest = list(csv.DictReader(f, delimiter="\t"))
    require(len(manifest) == 108, "expected the frozen release's 108 result files")
    paths = [r["path"] for r in manifest]
    require(len(set(paths)) == len(paths), "duplicate manifest path")
    actual = {p.relative_to(root).as_posix() for p in (root / "results").rglob("*") if p.is_file()}
    require(actual == set(paths), "manifest and results/ have different files")
    totals = collections.defaultdict(lambda: {"files": 0, "bytes": 0, "rows": 0})
    ids = {}
    duplicate_ids = 0
    for entry in manifest:
        rel = PurePosixPath(entry["path"])
        require(not rel.is_absolute() and ".." not in rel.parts and rel.parts[0] == "results", "unsafe manifest path")
        path = root / rel
        require(not path.is_symlink() and path.resolve().is_relative_to(root), "symlink outside artifact root")
        require(path.stat().st_size == int(entry["bytes"]), "size mismatch: " + entry["path"])
        require(digest(path) == entry["sha256"], "SHA mismatch: " + entry["path"])
        if source_root is not None:
            original = Path(source_root) / rel
            require(original.is_file() and digest(original) == entry["sha256"], "original-copy mismatch: " + entry["path"])
        records = read_rows(path)
        require(len(records) == int(entry["rows"]), "row count mismatch: " + entry["path"])
        group = entry["group"]
        totals[group]["files"] += 1
        totals[group]["bytes"] += int(entry["bytes"])
        totals[group]["rows"] += len(records)
        if group in ("beam_answer", "beam_judgment"):
            keys = [(r["row"], r["qi"]) for r in records]
            duplicate_ids += len(keys) - len(set(keys))
            require(len(keys) == len(set(keys)), "duplicate question ID: " + entry["path"])
            require(len(keys) in (200, 400, 700), "unexpected/partial run: " + entry["path"])
            ids[path.name] = set(keys)
    aligned = 0
    for entry in manifest:
        if entry["group"] != "beam_answer":
            continue
        name = Path(entry["path"]).name
        judge = name.removesuffix(".jsonl") + "_bj_gpt-4.1-mini.jsonl"
        require(judge in ids, "missing judgment file for: " + name)
        require(ids[name] == ids[judge], "answer/judge question IDs differ: " + name)
        aligned += 1
    require({k: v["files"] for k, v in totals.items()} == {"beam_answer": 51, "beam_judgment": 51, "honcho_call": 6}, "unexpected groups")
    return {"status": "passed", "files": len(manifest), "groups": dict(totals),
            "total_result_bytes": sum(v["bytes"] for v in totals.values()),
            "answer_judge_pairs_aligned_by_row_qi": aligned, "duplicate_question_ids": duplicate_ids,
            "source_copy_identity_checked": source_root is not None,
            "manifest_sha256": digest(root / "MANIFEST.tsv"), "new_model_or_api_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify(args.root, args.source_root)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
