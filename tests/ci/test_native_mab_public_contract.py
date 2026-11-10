"""Research-number regressions: named cells, scopes and final v9 artifacts.

These document checks do not measure a model or call a service. In particular,
a stale SDK value must fail even if it occurs in an original row elsewhere.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def contract():
    spec = importlib.util.spec_from_file_location("mab_contract_document_test", ROOT / "tools/native_mab_contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_checker(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("mab_source_document_test", ROOT / "tools/check_native_mab_tex.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def data():
    return json.loads((ROOT / "docs/data/native_mab_0_1_2.json").read_text(encoding="utf-8"))


def run_tool(name, *args):
    return subprocess.run([sys.executable, str(ROOT / "tools" / name), *map(str, args)],
                          capture_output=True, text=True, timeout=120)


def test_five_reader_rows_full_v9_artifacts_and_original_headline_are_current(contract, data):
    contract.validate_data(data, require_final_paper=True)
    for name, args in (("check_native_mab_tex.py", ()), ("make_native_mab.py", ("--check",)),
                       ("check_headline_tex.py", (ROOT / data["paper"]["tex"],))):
        result = run_tool(name, *args)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("reader", ("4o-mini", "4.1-mini", "5-mini", "glmh", "luna"))
def test_stale_sdk_cell_is_rejected_even_when_aggregates_are_recomputed(contract, data, reader):
    record = next(r for r in data["paper_configuration_readers"] if r["id"] == reader)
    assert record["cells"]["fcmh"]["native_percent"] != record["cells"]["fcmh"]["original_percent"]
    record["cells"]["fcmh"]["native_percent"] = record["cells"]["fcmh"]["original_percent"]
    values = {cell: item["native_percent"] for cell, item in record["cells"].items()}
    record["native"] = {key: float(value) for key, value in contract.recompute(values).items()}
    with pytest.raises(contract.ContractError, match="source-locked cell"):
        contract.validate_data(data)


def test_frontier_readers_are_not_interchangeable(contract, data):
    records = {r["id"]: r for r in data["paper_configuration_readers"]}
    records["glmh"]["cells"], records["luna"]["cells"] = records["luna"]["cells"], records["glmh"]["cells"]
    with pytest.raises(contract.ContractError, match="source-locked cell"):
        contract.validate_data(data)


def test_source_only_paper_status_cannot_satisfy_public_artifact_gate(contract, data):
    data["paper"] = {"revision": "v9", "status": "awaiting_v9_final_receipt"}
    contract.validate_data(data)
    with pytest.raises(contract.ContractError, match="final paper receipt"):
        contract.validate_data(data, require_final_paper=True)


@pytest.mark.parametrize("old_sha", (
    "cabe57f083623fd6abcc93ea7fbe3724139fb4be7fcfc8e0220861233b406e7e",
    "0f4ccc3c793e4c3e2a18558e7e92e224d0649f350b40ffc2a0b3405a5dce3279",
    "8c5bb0c5178dc68022f307fafe47c03ffe152ff772fd9cf39a7647c5d645ca78",
    "3b5030bf7af82f4616e86f0cffca6b09547463303f76def6278c022eee70b64b",
    "6896e3379fdcbadb9f62638c3a80874e00a9de2b95b2293fe557aad3f613b920",
    "6694014de38ef0678ed8883cf3b543b8486e08534d63ac4388f3e7a4a2173b84",
    "9896afaaa81b2cac4ded183db675eb4fff3ec8e2df46c9bed995a56b8b273f8e",
))
def test_earlier_paper_digest_cannot_be_relabelled_v9(contract, data, old_sha):
    data["paper"]["clean_pdf_sha256"] = old_sha
    with pytest.raises(contract.ContractError, match="mislabeled as v9"):
        contract.validate_data(data, require_final_paper=True)


@pytest.mark.parametrize("old_revision", ("v7", "v8"))
def test_earlier_metadata_is_not_accepted_as_the_current_paper(contract, data, old_revision):
    data["paper"]["revision"] = old_revision
    data["paper"]["status"] = "verified_" + old_revision + "_final_receipt"
    with pytest.raises(contract.ContractError, match="Expected v9 paper metadata"):
        contract.validate_data(data, require_final_paper=True)


def figure_fixture(clause):
    return (r"\begin{figure}[t]" + "\n" + r"\includegraphics[width=\textwidth]{fig_read3_v122.pdf}" + "\n"
            + r"\caption{Read flow: " + clause + r" Nothing is written back.}" + "\n"
            + r"\label{fig:read}" + "\n" + r"\end{figure}")


def test_only_exact_released_read_caption_clause_is_normalized(contract, source_checker):
    before, after = figure_fixture(contract.READ_CAPTION_BEFORE), figure_fixture(contract.READ_CAPTION_AFTER)
    assert source_checker.compare_figure_blocks([before], [after]) == 1
    assert source_checker.normalize_read_caption(after) == before


@pytest.mark.parametrize("mutation", ("image", "label", "other_caption_text", "outside_caption", "second_figure"))
def test_caption_exception_does_not_hide_other_figure_changes(contract, source_checker, mutation):
    before, after = figure_fixture(contract.READ_CAPTION_BEFORE), figure_fixture(contract.READ_CAPTION_AFTER)
    if mutation == "image":
        after = after.replace("fig_read3_v122.pdf", "different_image.pdf")
    elif mutation == "label":
        after = after.replace("fig:read", "fig:other")
    elif mutation == "other_caption_text":
        after = after.replace("Nothing is written back.", "Other evidence is now written back.")
    elif mutation == "outside_caption":
        after = after.replace(contract.READ_CAPTION_AFTER, "Retrieval.") + "\n" + contract.READ_CAPTION_AFTER
    if mutation == "second_figure":
        other = r"\begin{figure}\caption{Other result.}\label{fig:other}\end{figure}"
        with pytest.raises(source_checker.CheckError):
            source_checker.compare_figure_blocks([before, other], [after, other.replace("Other result.", "Changed result.")])
    else:
        with pytest.raises(source_checker.CheckError):
            source_checker.compare_figure_blocks([before], [after])


def test_pp_control_is_bound_to_its_own_axis(contract, data):
    primary = next(r for r in data["paper_configuration_readers"] if r["id"] == "4o-mini")
    data["sdk_parent_priority"]["cells"] = {c: x["native_percent"] for c, x in primary["cells"].items()}
    with pytest.raises(contract.ContractError, match="PP source-locked cell"):
        contract.validate_data(data)


def test_incompatible_response_accounting_groups_cannot_be_flattened(contract, data):
    data["provenance"]["response_accounting_groups"] = {"combined": deepcopy(data["provenance"]["response_accounting_groups"])}
    with pytest.raises(contract.ContractError, match="separate groups"):
        contract.validate_data(data)


def test_original_headline_does_not_fall_back_to_a_duplicate_sdk_label(data, tmp_path):
    source = (ROOT / data["paper"]["tex"]).read_text(encoding="utf-8")
    original = r"\textbf{Original, GPT-4o-mini}"
    assert source.count(original) == 1
    altered = source.replace(original, r"\textbf{SDK, GPT-4o-mini}", 1)
    path = tmp_path / "missing_explicit_original_scope.tex"
    path.write_text(altered, encoding="utf-8")
    result = run_tool("check_headline_tex.py", path)
    assert result.returncode != 0
    assert "original implementation needs exactly one row" in result.stdout + result.stderr


def test_sdk_policy_rows_keep_one_decimal_values_and_distinct_emphasis(data, source_checker):
    pp = r"SDK, \parprio{} everywhere & 70.5 & 47.7 & 39.4 & 76.5 & 58.5 \\"
    gate = (r"\ours \textbf{SDK, structure-matched gate} & \textbf{74.1} & \textbf{47.5} & "
            + r"\textbf{44.0} & \textbf{76.5} & \textbf{60.5} \\")
    primary = next(r for r in data["paper_configuration_readers"] if r["id"] == "4o-mini")
    source_checker.compare_policy_row(pp, "SDK, parent-priority everywhere", data["sdk_parent_priority"]["competencies"], False)
    source_checker.compare_policy_row(gate, "SDK, structure-matched gate", primary["native"], True)
    with pytest.raises(source_checker.CheckError, match="one-decimal"):
        source_checker.compare_policy_row(pp.replace("70.5", "70.50", 1), "SDK, parent-priority everywhere", data["sdk_parent_priority"]["competencies"], False)
    with pytest.raises(source_checker.CheckError):
        source_checker.compare_policy_row(r"\ours " + pp, "SDK, parent-priority everywhere", data["sdk_parent_priority"]["competencies"], False)
    with pytest.raises(source_checker.CheckError, match="highlight"):
        source_checker.compare_policy_row(gate.replace(r"\ours ", "", 1), "SDK, structure-matched gate", primary["native"], True)
