#!/usr/bin/env python3
"""Check five-reader SDK MAB data, exact paper rows, and baseline preservation.

Usage: python tools/check_native_mab_tex.py [--data docs/data/native_mab_0_1_2.json]
       [--tex paper/MUSTER_native_20261109.tex]
       [--baseline paper/MUSTER_followup_20261107.tex]

--source-only requires explicit --tex, --baseline, --direct-baseline and
--source-receipt. It verifies source before PDF packaging, emits an explicit
non-publication status and never substitutes for the default full artifact gate.
This verifier does not run a benchmark or import a model. A fresh --report may
be written on request. Display uses Decimal ROUND_HALF_UP; duplicate reader
labels are parsed within their explicit implementation scopes.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, getcontext
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile
from native_mab_contract import (AUTO_MARKER, DIRECT_V6_SHA, PP_MARKER,
    PUBLIC_B_SHA, READ_CAPTION_AFTER, READ_CAPTION_BEFORE, READERS as READER_SPECS,
    SDK_MARKER, SOURCE_RECEIPT_SCHEMA, TABLE_SOURCES,
    ContractError, validate_data)

ROOT = Path(__file__).resolve().parents[1]
getcontext().prec = 50
AGGREGATES = ("AR", "MCC", "TTL", "LRU", "SF", "Overall")
COUNTS = {
    "shqa": 100, "mhqa": 100, "lme": 300, "evq": 100,
    "b77": 100, "clin": 100, "nlu": 100, "trcC": 100,
    "trcF": 100, "rec": 200, "summ": 100, "det": 71,
    "fcsh": 100, "fcmh": 100,
}
CELL_LABELS = {
    "SH-QA": "shqa", "MH-QA": "mhqa", "LongMemEval": "lme",
    "EventQA": "evq", "Banking77": "b77", "CLINC150": "clin",
    "NLU": "nlu", "TREC-coarse": "trcC", "TREC-fine": "trcF",
    "Recsys": "rec", "InfBench-Sum": "summ", "DetectiveQA": "det",
    "FC-SH": "fcsh", "FC-MH": "fcmh",
}
READERS = {key: ("SDK, " + spec[0], spec[1], spec[2]) for key, spec in READER_SPECS.items()}
ORIGINAL_BOARD_READERS = ("4o-mini", "4.1-mini", "5-mini")
CHANGED_TABLES = set(TABLE_SOURCES)
TOLERANCE = Decimal("0.0000000001")


class CheckError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise CheckError(message)


def repo_path(value, role, source_only=False):
    candidate = Path(value)
    path = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    require(source_only or path.is_relative_to(ROOT), role + " must stay inside the repository")
    require(path.is_file(), role + " file is missing")
    return path


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, context):
    require(not isinstance(value, bool), context + ": boolean is not a metric")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise CheckError(context + ": invalid number") from None
    require(result.is_finite() and 0 <= result <= 100, context + ": outside0..100")
    return result


def shown(value, places):
    quantum = Decimal(1).scaleb(-places)
    return format(value.quantize(quantum, rounding=ROUND_HALF_UP), ".%df" % places)


def average(values):
    return sum(values, Decimal(0)) / Decimal(len(values))


def recompute(cells):
    mcc = average([cells[c] for c in ("b77", "clin", "nlu", "trcC", "trcF")])
    ar = average([cells[c] for c in ("shqa", "mhqa", "lme", "evq")])
    ttl = average([mcc, cells["rec"]])
    lru = average([cells["summ"], cells["det"]])
    sf = average([cells["fcsh"], cells["fcmh"]])
    return {"AR": ar, "MCC": mcc, "TTL": ttl, "LRU": lru, "SF": sf,
            "Overall": average([ar, ttl, lru, sf])}


def validated_record(record, expected_id, scope):
    require(record["id"] == expected_id, scope + ": wrong reader id")
    require(record["request_model"] == READERS[expected_id][1], scope + ": wrong requested model")
    require(isinstance(record["label"], str) and record["label"].strip(), scope + ": missing label")
    require(set(record["cells"]) == set(COUNTS), scope + ": expected exactly14 cells")
    values = {"original": {}, "native": {}}
    for cell, count in COUNTS.items():
        entry = record["cells"][cell]
        require(type(entry["n"]) is int and entry["n"] == count, scope + "/" + cell + ": wrong n")
        require(type(entry["repeats"]) is int and entry["repeats"] == (3 if cell == "det" else 1),
                scope + "/" + cell + ": wrong repeat count")
        for view in values:
            values[view][cell] = number(entry[view + "_percent"], scope + "/" + cell + "/" + view)
    require(sum(entry["n"] for entry in record["cells"].values()) == 1671, scope + ": wrong denominator total")
    aggregates = {}
    for view in values:
        require(set(record[view]) == set(AGGREGATES), scope + "/" + view + ": aggregate keys differ")
        computed = recompute(values[view])
        for metric in AGGREGATES:
            saved = number(record[view][metric], scope + "/" + view + "/" + metric)
            require(abs(saved - computed[metric]) <= TOLERANCE,
                    scope + "/" + view + "/" + metric + ": aggregate does not match14 cells")
        aggregates[view] = computed
    return {"cells": values, "aggregates": aggregates}


def uncomment(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def canonical(text):
    return re.sub(r"\s+", " ", uncomment(text)).strip()


def section_headings(text):
    text = uncomment(text)
    result = []
    for match in re.finditer(r"\\((?:sub)*section)(\*?)\{", text):
        end, depth = match.end(), 1
        while depth and end < len(text):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            end += 1
        require(depth == 0, "Unclosed section heading")
        result.append((match[1], match[2], canonical(text[match.end():end - 1])))
    return result


def environments(text, names):
    token = re.compile(r"\\(begin|end)\{([^}]+)\}")
    stack, result = [], []
    for match in token.finditer(text):
        action, name = match.groups()
        if name not in names:
            continue
        if action == "begin":
            stack.append((name, match.start()))
        else:
            require(bool(stack) and stack[-1][0] == name, "Unbalanced TeX environment " + name)
            _, start = stack.pop()
            if not stack:
                result.append(text[start:match.end()])
    require(not stack, "Unclosed TeX environment")
    return result


def table_map(text):
    result = {}
    for block in environments(text, {"table", "table*"}):
        labels = re.findall(r"\\label\{([^}]+)\}", block)
        require(bool(labels), "Table without a label")
        require(labels[0] not in result, "Repeated table label " + labels[0])
        result[labels[0]] = block
    return result


def read_caption_span(block):
    """Locate the one actual caption body; a matching phrase elsewhere is invalid."""
    prefix = r"\caption{"
    require(block.count(prefix) == 1, "fig:read must contain one ordinary caption")
    start = block.index(prefix) + len(prefix)
    end, depth = start, 1
    while end < len(block) and depth:
        if block[end] == "\\":
            end += 2
            continue
        depth += (block[end] == "{") - (block[end] == "}")
        end += 1
    require(depth == 0, "Unclosed fig:read caption")
    return start, end - 1


def normalize_read_caption(block):
    """Reverse only the authorized rerank substring inside fig:read's caption."""
    require(re.findall(r"\\label\{([^}]+)\}", block) == ["fig:read"], "Wrong fig:read label scope")
    start, end = read_caption_span(block)
    caption = block[start:end]
    require(caption.count(READ_CAPTION_AFTER) == block.count(READ_CAPTION_AFTER) == 1,
            "Expected the exact released rerank clause once inside the fig:read caption")
    require(READ_CAPTION_BEFORE not in block, "Old one-rerank clause remains in current fig:read")
    return block[:start] + caption.replace(READ_CAPTION_AFTER, READ_CAPTION_BEFORE, 1) + block[end:]


def compare_figure_blocks(before, after):
    """Keep full blocks, images and labels; permit exactly the declared caption delta."""
    require(bool(before) and len(before) == len(after), "Changed figure inventory")
    corrected = 0
    for old, new in zip(before, after):
        old_labels = re.findall(r"\\label\{([^}]+)\}", old)
        require(old_labels == re.findall(r"\\label\{([^}]+)\}", new), "Changed figure labels/order")
        if "fig:read" in old_labels:
            require(old_labels == ["fig:read"], "Ambiguous baseline fig:read label")
            start, end = read_caption_span(old)
            require(old[start:end].count(READ_CAPTION_BEFORE) == old.count(READ_CAPTION_BEFORE) == 1,
                    "Baseline fig:read does not contain the exact historical clause")
            require(normalize_read_caption(new) == old, "fig:read changed beyond the authorized caption clause")
            corrected += 1
        else:
            require(new == old, "Changed non-exempt figure block")
    require(corrected == 1, "Expected exactly one scoped fig:read correction")
    return len(before)


def unwrap(text, command):
    prefix = "\\" + command + "{"
    while prefix in text:
        start = text.index(prefix)
        end, depth = start + len(prefix), 1
        while depth and end < len(text):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            end += 1
        require(depth == 0, "Unclosed TeX command " + command)
        text = text[:start] + text[start + len(prefix):end - 1] + text[end:]
    return text


def clean(cell):
    cell = re.sub(r"\\cite\{[^}]*\}|\\ours|\\smash|\\phantom\{0\}", "", cell)
    cell = unwrap(cell, "textbf").replace("\\sys{}", "MUSTER").replace("\\parprio{}", "parent-priority")
    cell = cell.replace("{,}", ",").replace("\\,", "").replace("$", "")
    cell = cell.replace("\\\\", "").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", cell).strip()


def row_cells(line):
    return [clean(cell) for cell in line.split("&")]


def board_sections(block):
    sections = {"original": [], "native": [], "auto": []}
    scope = "original"
    for line in block.splitlines():
        plain = clean(line)
        if SDK_MARKER in plain:
            require(scope == "original", "SDK board section is repeated or reordered")
            require("\\multicolumn{16}{l}" in line, "SDK marker must scope the board")
            scope = "native"
        elif AUTO_MARKER in plain:
            require(scope == "native", "Automatic board section is repeated or reordered")
            require("\\multicolumn{16}{l}" in line, "Automatic marker must scope the board")
            scope = "auto"
        elif "&" in line:
            sections[scope].append((line, row_cells(line)))
    require(sections["native"] and sections["auto"], "Missing native or auto board section")
    return sections


def board_values(validated, view):
    c, a = validated["cells"][view], validated["aggregates"][view]
    return [c["shqa"], c["mhqa"], c["lme"], c["evq"], a["MCC"], c["rec"], c["summ"], c["det"],
            c["fcsh"], c["fcmh"], a["AR"], a["TTL"], a["LRU"], a["SF"], a["Overall"]]


def compare_board_row(cells, values, context):
    require(len(cells) == 16, context + ": board needs15 values")
    expected = [shown(v, 1) for v in values]
    require(cells[1:] == expected, context + ": one-decimal values differ")


def compare_policy_row(line, label, values, emphasized):
    """Check the named SDK policy, exact one-decimal values and its emphasis."""
    row = row_cells(line)
    require(row[0] == label and len(row) == 6, "Wrong SDK policy row identity")
    expected = [shown(number(values[k], label + "/" + k), 1) for k in ("AR", "TTL", "LRU", "SF", "Overall")]
    require(row[1:] == expected, "SDK policy one-decimal row differs: " + label)
    raw = line.split("&")
    if emphasized:
        require(raw[0].strip() == r"\ours \textbf{SDK, structure-matched gate}", "SDK gate row must retain its exact highlight")
        require(all(cell.strip().startswith(r"\textbf{" + value + "}") for cell, value in zip(raw[1:], expected)),
                "SDK gate values must retain their highlight")
    else:
        require(raw[0].strip() == r"SDK, \parprio{} everywhere", "SDK PP row must use the declared policy label")
        require(r"\ours" not in line and r"\textbf" not in line, "SDK PP control must be a normal row")


def expanded_rows(block):
    result = []
    for line in block.splitlines():
        if "&" not in line:
            continue
        cells = row_cells(line)
        if cells[0] in CELL_LABELS or cells[0] == "Overall":
            result.append(cells)
    require(len(result) == 15 and len({r[0] for r in result}) == 15, "Expanded table needs14 distinct cells and Overall")
    require({r[0] for r in result} == set(CELL_LABELS) | {"Overall"}, "Expanded cell names differ")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="docs/data/native_mab_0_1_2.json")
    parser.add_argument("--tex")
    parser.add_argument("--baseline")
    parser.add_argument("--direct-baseline")
    parser.add_argument("--source-receipt")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--report", type=Path, help="Write a new receipt; existing receipts are never overwritten")
    args = parser.parse_args()
    if args.source_only:
        require(all((args.tex, args.baseline, args.direct_baseline, args.source_receipt)),
                "Source-only needs explicit TeX, public B, direct v6 and source receipt")
    require(args.report is None or not args.report.exists(), "Report already exists; preserve completed checks")
    data_path = repo_path(args.data, "JSON", args.source_only)
    data = json.loads(data_path.read_text(encoding="utf-8"), parse_float=Decimal)
    validate_data(data, require_final_paper=not args.source_only)
    tex_path = repo_path(args.tex or data["paper"]["tex"], "SDK TeX", args.source_only)
    baseline_path = repo_path(args.baseline or data["paper"]["baseline_tex"], "Public B TeX", args.source_only)
    require(file_sha(baseline_path) == PUBLIC_B_SHA, "Public B is not the locked baseline")
    artifact_checks = {}
    if not args.source_only:
        require(tex_path == repo_path(data["paper"]["tex"], "Bound TeX"), "Full check must use the bound v9 TeX")
        for field, digest_field in (("tex", "tex_sha256"), ("pdf", "clean_pdf_sha256"),
                                   ("marked_pdf", "marked_pdf_sha256"), ("source_zip", "source_zip_sha256")):
            artifact = repo_path(data["paper"][field], field)
            require(file_sha(artifact) == data["paper"][digest_field], field + ": v9 artifact digest differs")
            artifact_checks[field] = data["paper"][digest_field]
        with zipfile.ZipFile(repo_path(data["paper"]["source_zip"], "Source ZIP")) as archive:
            tex_members = [name for name in archive.namelist() if name.endswith(".tex")]
            require(tex_members == [tex_path.name], "Source ZIP must contain the single bound TeX member")
            require(archive.read(tex_members[0]) == tex_path.read_bytes(), "Source ZIP TeX differs from checked v9 source")
    require(tex_path != baseline_path, "SDK and baseline TeX must be different files")
    records = data["paper_configuration_readers"]
    primary = {r["id"]: validated_record(r, r["id"], "primary/" + r["id"]) for r in records}
    auto = validated_record(data["protocol_auto_4omini"], "4o-mini", "auto/4o-mini")
    require(auto["cells"]["original"] == primary["4o-mini"]["cells"]["original"], "Auto and primary original4o-mini cells differ")
    current = tex_path.read_text(encoding="utf-8")
    baseline = baseline_path.read_text(encoding="utf-8")
    require("leads the reported retrieval and memory systems on every competency" not in canonical(current),
            "The SDK's highest Overall must not be presented as leading every competency")
    now, before = table_map(current), table_map(baseline)
    require(len(before) == 30 and len(now) == 31 and set(now) == set(before) | {"tab:expandedfrontier"},
            "Expected public B's30 tables and only the new frontier table")
    old_headings, new_headings = section_headings(baseline), section_headings(current)
    require(old_headings == new_headings, "Section structure changed; do not restore a removed appendix")
    require(CHANGED_TABLES <= set(now), "Expected six source-bound result tables are missing")
    unchanged = sorted(set(before) - CHANGED_TABLES)
    require(len(unchanged) == 25, "Expected25 unchanged tables")
    for label in unchanged:
        require(before[label] == now[label], "Changed full public B table block " + label)
    preservation = {}
    preservation["figures"] = compare_figure_blocks(environments(baseline, {"figure", "figure*"}),
                                                    environments(current, {"figure", "figure*"}))
    old_algorithms, new_algorithms = environments(baseline, {"algorithm", "algorithm*"}), environments(current, {"algorithm", "algorithm*"})
    require(old_algorithms and old_algorithms == new_algorithms, "Changed algorithms")
    preservation["algorithms"] = len(old_algorithms)
    source_receipt_path = None
    if args.source_receipt:
        source_receipt_path = repo_path(args.source_receipt, "v9 source receipt", args.source_only)
        source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
        require(source_receipt["schema"] == SOURCE_RECEIPT_SCHEMA, "Not a v9 representative source receipt")
        require(source_receipt["revision"] == "v9", "Source receipt revision is not v9")
        require(source_receipt["allowed_modified_tables"] == list(TABLE_SOURCES), "Source table mapping differs")
        require(source_receipt["added_vs_public_B_tables"] == ["tab:expandedfrontier"], "Source added-table mapping differs")
        require(sorted(source_receipt["unchanged_table_labels"]) == unchanged, "Source unchanged-table receipt differs")
        for key, sha, count in (("public_B", PUBLIC_B_SHA, 30), ("direct_v6", DIRECT_V6_SHA, 31)):
            meta = source_receipt["comparison_baselines"][key]
            require(meta["sha256"] == sha and meta["tables"] == count, "Source baseline metadata differs: " + key)
        require(source_receipt["primary_metric_source_sha256"] == data["provenance"]["primary_metric_sha256"], "Source primary pin differs")
        require(source_receipt["PP_metric_source_sha256"] == data["sdk_parent_priority"]["source_sha256"], "Source PP pin differs")
    direct_path = None
    if args.direct_baseline:
        direct_path = repo_path(args.direct_baseline, "Direct v6 TeX", args.source_only)
        require(file_sha(direct_path) == DIRECT_V6_SHA, "Direct v6 is not the locked baseline")
        direct_text = direct_path.read_text(encoding="utf-8")
        direct_tables = table_map(direct_text)
        require(len(direct_tables) == 31 and set(direct_tables) == set(now), "Direct v6 table labels differ")
        require(sorted(set(direct_tables) - CHANGED_TABLES) == unchanged, "Direct v6 unchanged-table mapping differs")
        for label in unchanged:
            require(direct_tables[label] == now[label], "Changed full direct v6 table block " + label)
        compare_figure_blocks(environments(direct_text, {"figure", "figure*"}), environments(current, {"figure", "figure*"}))
        require(environments(direct_text, {"algorithm", "algorithm*"}) == new_algorithms, "Changed direct v6 algorithms")
        require(section_headings(direct_text) == new_headings, "Changed direct v6 section structure")
        if source_receipt_path:
            require(set(source_receipt["comparison_baselines"]["direct_v6"]["table_labels"]) == set(direct_tables), "Receipt direct v6 table inventory differs")
    if source_receipt_path:
        require(set(source_receipt["comparison_baselines"]["public_B"]["table_labels"]) == set(before), "Receipt public B table inventory differs")
    sections = board_sections(now["tab:board"])
    old_rows = [line for line in before["tab:board"].splitlines() if "&" in line]
    def original_label_only(line):
        prefix, amp, rest = line.partition("&")
        for reader in ORIGINAL_BOARD_READERS:
            if clean(prefix) == "Original, " + READER_SPECS[reader][0]:
                require(prefix.count("Original,") == 1, "Ambiguous original row label")
                prefix = prefix.replace("Original,", "\\sys{},", 1)
        return canonical(prefix + amp + rest)
    require([canonical(line) for line in old_rows] == [original_label_only(line) for line, _ in sections["original"]],
            "Original board rows changed beyond the three explicit Original reader labels")
    native_rows = sections["native"]
    require(len(native_rows) == 5 and len(sections["auto"]) == 1, "Expected exactly5 SDK paper and1 automatic rows")
    for reader, spec in READERS.items():
        matches = [cells for _, cells in native_rows if cells[0] == spec[0]]
        require(len(matches) == 1, "Missing/duplicate SDK paper reader " + reader)
        compare_board_row(matches[0], board_values(primary[reader], "native"), "SDK paper board/" + reader)
    for reader in ORIGINAL_BOARD_READERS:
        original = [cells for _, cells in sections["original"] if cells[0] == "Original, " + READER_SPECS[reader][0]]
        require(len(original) == 1, "Missing/duplicate original board reader " + reader)
        compare_board_row(original[0], board_values(primary[reader], "original"), "original board/" + reader)
    auto_row = sections["auto"][0][1]
    require(auto_row[0] == "Auto, " + READER_SPECS["4o-mini"][0], "Wrong auto board reader")
    compare_board_row(auto_row, board_values(auto, "native"), "auto board/4o-mini")
    for reader in ORIGINAL_BOARD_READERS:
        spec = READERS[reader]
        original_rows = expanded_rows(before[spec[2]])
        native_detail = expanded_rows(now[spec[2]])
        require([r[0] for r in original_rows] == [r[0] for r in native_detail], "Expanded rows reordered for " + reader)
        for old_row, row in zip(original_rows, native_detail):
            require(len(old_row) == 4 and len(row) == (6 if reader == "4o-mini" else 5), "Expanded column count differs for " + reader)
            require([row[0], row[1], row[2], row[-1]] == old_row, "Original/Infini-style expanded values changed for " + reader + "/" + row[0])
            if row[0] == "Overall":
                old_value = primary[reader]["aggregates"]["original"]["Overall"]
                native_value = primary[reader]["aggregates"]["native"]["Overall"]
                auto_value = auto["aggregates"]["native"]["Overall"]
                require(row[1] == "", "Expanded Overall must not invent an n")
            else:
                cell = CELL_LABELS[row[0]]
                require(row[1] == str(COUNTS[cell]), "Expanded n differs for " + cell)
                old_value = primary[reader]["cells"]["original"][cell]
                native_value = primary[reader]["cells"]["native"][cell]
                auto_value = auto["cells"]["native"][cell]
            require(row[2] == shown(old_value, 2), "Original expanded JSON/TeX differs for " + reader + "/" + row[0])
            require(row[3] == shown(native_value, 2), "Native expanded JSON/TeX differs for " + reader + "/" + row[0])
            if reader == "4o-mini":
                require(row[4] == shown(auto_value, 2), "Auto expanded JSON/TeX differs for " + row[0])
    # GLM/Luna are one new six-column table, not two interchangeable readers.
    frontier = expanded_rows(now["tab:expandedfrontier"])
    for row in frontier:
        require(len(row) == 6, "Frontier table needs Cell,n,GLM original/SDK,Luna original/SDK")
        if row[0] == "Overall":
            require(row[1] == "", "Frontier Overall must not invent an n")
            vv = [primary[r]["aggregates"][view]["Overall"] for r in ("glmh", "luna") for view in ("original", "native")]
        else:
            cell = CELL_LABELS[row[0]]
            require(row[1] == str(COUNTS[cell]), "Frontier n differs for " + cell)
            vv = [primary[r]["cells"][view][cell] for r in ("glmh", "luna") for view in ("original", "native")]
        require(row[2:] == [shown(v, 2) for v in vv], "Frontier source-bound values differ: " + row[0])
    gate_original, gate_sdk, in_sdk = [], [], False
    for line in now["tab:gate"].splitlines():
        if PP_MARKER in clean(line):
            require(not in_sdk and "\\multicolumn{6}{l}" in line, "Repeated or misplaced SDK policy marker")
            in_sdk = True
        elif "&" in line:
            (gate_sdk if in_sdk else gate_original).append(line)
    require(in_sdk and len(gate_sdk) == 2, "SDK policy block needs exactly PP and structure-matched rows")
    require([canonical(line) for line in gate_original] == [canonical(line) for line in before["tab:gate"].splitlines() if "&" in line],
            "Original policy controls changed")
    for line, label, values, emphasized in zip(gate_sdk, ("SDK, parent-priority everywhere", "SDK, structure-matched gate"),
                                   (data["sdk_parent_priority"]["competencies"], primary["4o-mini"]["aggregates"]["native"]), (False, True)):
        compare_policy_row(line, label, values, emphasized)
    checked_sources = [("data", data_path), ("SDK_tex", tex_path), ("public_B_tex", baseline_path)]
    if direct_path:
        checked_sources.append(("direct_v6_tex", direct_path))
    if source_receipt_path:
        checked_sources.append(("source_receipt", source_receipt_path))
    result = {
        "status": "passed_independent_six_table_source_audit" if args.source_only else "passed_five_reader_mab_full_artifact_gate",
        "mode": "source_only_not_publication_ready" if args.source_only else "full_public_artifact_check",
        "publication_ready": False,
        "aggregate_sets_recomputed": len(records) * 2 + 3,
        "aggregate_values_checked": (len(records) * 2 + 3) * len(AGGREGATES),
        "table1_new_rows": len(native_rows) + len(sections["auto"]),
        "table1_new_one_decimal_values": (len(native_rows) + len(sections["auto"])) * 15,
        "expanded_GPT_SDK_two_decimal_values": len(ORIGINAL_BOARD_READERS) * 15,
        "expanded_frontier_two_decimal_values": len(frontier) * 4,
        "expanded_auto_two_decimal_values": 15,
        "SDK_policy_rows": len(gate_sdk), "SDK_policy_one_decimal_values": len(gate_sdk) * 5,
        "SDK_policy_normal_PP_and_highlighted_gate": True,
        "allowed_modified_tables": list(TABLE_SOURCES), "added_vs_public_B_tables": ["tab:expandedfrontier"],
        "source_labels": data["paper_source_labels"],
        "unchanged_full_table_blocks": len(unchanged), "unchanged_table_labels": unchanged,
        "public_B_table_counts": [len(before), len(now)],
        "direct_v6_checked": direct_path is not None,
        "direct_v6_table_counts": [31, 31] if direct_path else None,
        "original_board_rows_preserved_except_three_explicit_reader_labels": len(old_rows),
        "original_and_Infini_style_expanded_rows_preserved": 45,
        "section_headings_preserved": len(old_headings),
        "figure_blocks_preserved_except_one_exact_caption_clause": preservation["figures"],
        "unmodified_figure_blocks": preservation["figures"] - 1,
        "figure_caption_exception": {"label": "fig:read", "before": READ_CAPTION_BEFORE, "after": READ_CAPTION_AFTER,
                                     "scope": "one exact substring inside the actual caption; all other block bytes equal"},
        "figure_assets_and_labels_preserved": True, "algorithms_preserved": preservation["algorithms"],
        "display_rounding": "Decimal ROUND_HALF_UP", "new_quality_or_publication_claim": False,
        "v9_final_artifact_hashes_checked": artifact_checks,
        "artifacts": {role: {"file": path.name if args.source_only else path.relative_to(ROOT).as_posix(), "sha256": file_sha(path)}
                      for role, path in checked_sources},
    }
    if args.report:
        with args.report.open("x", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (CheckError, ContractError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
    except (KeyError, TypeError, ValueError, OSError, InvalidOperation, IndexError, zipfile.BadZipFile) as error:
        # Keep filesystem errors and arbitrary data strings out of public logs.
        print(json.dumps({"status": "failed", "error": "Malformed or unreadable artifact", "error_type": type(error).__name__}), file=sys.stderr)
        sys.exit(1)
