"""Research-number regressions: named cells, scopes and final v10 artifacts.

These document checks do not measure a model or call a service. In particular,
a stale SDK value must fail even if it occurs in an original row elsewhere.
"""
from copy import deepcopy
import csv
from decimal import Decimal, ROUND_HALF_UP
import importlib.util
import io
import json
from pathlib import Path
import re
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
def generator(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("mab_generator_document_test", ROOT / "tools/make_native_mab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def data():
    return json.loads((ROOT / "docs/data/native_mab_0_1_2.json").read_text(encoding="utf-8"), parse_float=Decimal)


def run_tool(name, *args):
    return subprocess.run([sys.executable, str(ROOT / "tools" / name), *map(str, args)],
                          capture_output=True, text=True, timeout=120)


def test_thirteen_reader_rows_full_v10_artifacts_and_original_headline_are_current(contract, data):
    contract.validate_data(data, require_final_paper=True)
    for name, args in (("check_native_mab_tex.py", ()), ("make_native_mab.py", ("--check",)),
                       ("check_headline_tex.py", (ROOT / data["paper"]["tex"],))):
        result = run_tool(name, *args)
        assert result.returncode == 0, result.stdout + result.stderr
        if name == "check_native_mab_tex.py":
            audit = json.loads(result.stdout)
            assert audit["table1_new_rows"] == 8
            assert audit["expanded_auto_two_decimal_values"] == 45
            assert audit["auto_extension_csv_rows"] == 30
            assert audit["existing_auto4omini_csv_rows"] == 14
            assert audit["source_zip_auto_csv_bytes_checked"] is True
            assert audit["previous_v9_numeric_table_blocks"] == 3
            assert audit["previous_v9_caption_only_table_blocks"] == 1
            assert audit["previous_v9_unchanged_full_table_blocks"] == 27
            assert audit["unchanged_full_table_blocks"] == 25
            assert audit["original_and_Infini_style_expanded_rows_preserved"] == 45
            scope = audit["final_v10_scope_exceptions"]
            assert len(scope["fig:read"]["caption"]) == 2
            assert len(scope["alg:muster"]["caption"]) == 2
            assert len(scope["alg:muster"]["algorithmic"]) == 4
            assert audit["other_figure_and_algorithm_block_bytes_preserved"] is True


def test_readme_and_landing_bind_all_five_metrics_to_thirteen_scoped_rows(data):
    primary = data["paper_configuration_readers"]
    auto = data["protocol_auto_readers"]
    expected = []
    for scope, configuration, records, view in (
        ("original", "Original implementation", primary, "original"),
        ("sdk-paper", "SDK (paper settings)", primary, "native"),
        ("sdk-auto", "SDK (automatic settings)", auto, "native"),
    ):
        for record in records:
            values = [format(Decimal(str(record[view][metric])).quantize(Decimal(".01"), rounding=ROUND_HALF_UP), ".2f")
                      for metric in ("AR", "TTL", "LRU", "SF", "Overall")]
            expected.append((scope, record["id"], record["label"], configuration, values))
    assert len(expected) == 13
    start, end = "<!-- BEGIN native-mab -->", "<!-- END native-mab -->"
    html = (ROOT / "docs/index.html").read_text(encoding="utf-8").split(start)[1].split(end)[0]
    matches = re.findall(r'<tr data-result-scope="([^"]+)" data-reader="([^"]+)">'
                         r'<th scope="row">([^<]+)</th><td>([^<]+)</td>(.*?)</tr>', html, re.S)
    actual = [(scope, reader, label, config, re.findall(r'<td class="n">([^<]+)</td>', cells))
              for scope, reader, label, config, cells in matches]
    assert html.count("data-result-scope=") == len(actual) == 13
    assert actual == expected
    readme = (ROOT / "README.md").read_text(encoding="utf-8").split(start)[1].split(end)[0]
    rows = [line[2:-2].split(" | ") for line in readme.splitlines()
            if line.startswith("| ") and not line.startswith("| Reader |")]
    assert rows == [[label, config, *values] for _, _, label, config, values in expected]


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


@pytest.mark.parametrize("reader", ("4.1-mini", "5-mini"))
def test_stale_sdk_summ_cannot_replace_an_auto_cell(contract, data, reader):
    auto = next(r for r in data["protocol_auto_readers"] if r["id"] == reader)
    paper = next(r for r in data["paper_configuration_readers"] if r["id"] == reader)
    assert auto["cells"]["summ"]["native_percent"] != paper["cells"]["summ"]["native_percent"]
    auto["cells"]["summ"]["native_percent"] = paper["cells"]["summ"]["native_percent"]
    auto["native"] = contract.recompute({cell: entry["native_percent"] for cell, entry in auto["cells"].items()})
    with pytest.raises(contract.ContractError, match="source-locked cell"):
        contract.validate_data(data)


def test_auto_4omini_alias_cannot_drift(contract, data):
    data["protocol_auto_4omini"]["native"]["Overall"] = 60
    with pytest.raises(contract.ContractError, match="automatic alias differs"):
        contract.validate_data(data)


def test_three_auto_readers_have_a_fixed_order(contract, data):
    data["protocol_auto_readers"][1:] = reversed(data["protocol_auto_readers"][1:])
    with pytest.raises(contract.ContractError, match="three automatic readers in fixed order"):
        contract.validate_data(data)


def test_auto_extension_source_pins_are_bound_to_their_reader(contract, data):
    pins = data["provenance"]["auto_extension_metric_sha256"]
    pins["4.1-mini"], pins["5-mini"] = pins["5-mini"], pins["4.1-mini"]
    with pytest.raises(contract.ContractError, match="Auto extension source pins differ"):
        contract.validate_data(data)


def test_auto_fc_reuse_cannot_be_reported_as_fresh_responses(contract, data):
    record = data["protocol_auto_readers"][1]
    record["read_provenance"]["completed_extension_reader_response"] += record["latest_SDK_FC_saved_request_reuses"]
    record["latest_SDK_FC_saved_request_reuses"] = 0
    with pytest.raises(contract.ContractError, match="Auto extension response provenance differs"):
        contract.validate_data(data)


def test_auto_extension_csv_can_be_staged_before_pdf_receipt(generator, source_checker, data, tmp_path):
    data["paper"] = {"revision": "v10", "status": "awaiting_v10_final_receipt"}
    text = generator.render_auto_additional_csv(data)
    reader = csv.DictReader(io.StringIO(text))
    rows = {(row["reader"], row["cell"]): row for row in reader}
    assert len(rows) == 30
    assert rows["4.1-mini", "lme"]["old_percent"] == "61.66666666666667"
    assert rows["4.1-mini", "lme"]["new_percent"] == "61.666666666666664"
    assert rows["5-mini", "summ"]["new_percent"] == "25.27711738604311"
    assert all(rows[key, "Overall"]["n"] == rows[key, "Overall"]["repeats"] == "" for key in ("4.1-mini", "5-mini"))
    path = tmp_path / "additional.csv"
    path.write_text(text, encoding="utf-8")
    assert source_checker.check_auto_additional_csv(path, data) == 30
    path.write_text(text.replace("4.1-mini,Overall,,,", "4.1-mini,Overall,1671,,", 1), encoding="utf-8")
    with pytest.raises(source_checker.CheckError, match="Overall CSV must not invent an n"):
        source_checker.check_auto_additional_csv(path, data)


@pytest.mark.parametrize("mutation, error", (
    ("tabular", "caption-only tabular changed"),
    ("other_caption_text", "beyond the authorized clause"),
))
def test_v9_auto4omini_caption_exception_preserves_other_bytes(data, source_checker, mutation, error):
    label = "tab:expanded4o-mini"
    old = source_checker.table_map((ROOT / data["paper"]["previous_tex"]).read_text(encoding="utf-8"))[label]
    new = source_checker.table_map((ROOT / data["paper"]["tex"]).read_text(encoding="utf-8"))[label]
    source_checker.compare_auto_caption_table(old, new, label)
    if mutation == "tabular":
        altered = new.replace(r"\toprule", r"\midrule", 1)
    else:
        altered = new.replace("0--100 scale", "another scale", 1)
    assert altered != new
    with pytest.raises(source_checker.CheckError, match=error):
        source_checker.compare_auto_caption_table(old, altered, label)


def test_source_only_paper_status_cannot_satisfy_public_artifact_gate(contract, data):
    data["paper"] = {"revision": "v10", "status": "awaiting_v10_final_receipt"}
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
    "7ead1fcbc357864524931599bf22268f350be9d522f81958dbe8bde08c0af0a9",
    "b8b02cd7db115d6be578deb552b9be401bfae4b0e1f4f60ecb4e49e4ec0ea338",
    "83c7203dd2b313d03f258169ccd069c83bb563d773de79f54fbf9f830fe29d57",
    "784483e4317f167fd02dae32cb6106de683e467c838261652f92c1383066fae4",
))
def test_earlier_paper_digest_cannot_be_relabelled_v10(contract, data, old_sha):
    data["paper"]["clean_pdf_sha256"] = old_sha
    with pytest.raises(contract.ContractError, match="mislabeled as v10"):
        contract.validate_data(data, require_final_paper=True)


@pytest.mark.parametrize("old_revision", ("v7", "v8", "v9"))
def test_earlier_metadata_is_not_accepted_as_the_current_paper(contract, data, old_revision):
    data["paper"]["revision"] = old_revision
    data["paper"]["status"] = "verified_" + old_revision + "_final_receipt"
    with pytest.raises(contract.ContractError, match="Expected v10 paper metadata"):
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


@pytest.mark.parametrize("mutation, error", (
    ("figure_asset", "Changed v9 figure block"),
    ("duplicate_caption_clause", "once inside fig:read caption"),
    ("algorithm_parent_multiplier", "Changed v9 algorithm"),
    ("misplaced_algorithm_clause", "once inside alg:muster algorithmic body"),
))
def test_final_scope_restoration_preserves_unrelated_float_bytes(data, source_checker, mutation, error):
    previous = (ROOT / data["paper"]["previous_tex"]).read_text(encoding="utf-8")
    current = (ROOT / data["paper"]["tex"]).read_text(encoding="utf-8")
    if mutation in ("figure_asset", "duplicate_caption_clause"):
        figures = source_checker.environments(current, {"figure", "figure*"})
        block = next(block for block in figures if r"\label{fig:read}" in block)
        if mutation == "figure_asset":
            changed = block.replace("fig_read3_v122.pdf", "other_asset.pdf", 1)
        else:
            clause = (
                "The refined branch, LoCoMo, LoCoMo-Refined and BEAM use one container per parent "
                r"(Appendix~\ref{app:hyper})."
            )
            changed = block.replace(clause, clause + " " + clause, 1)
    else:
        algorithms = source_checker.environments(current, {"algorithm", "algorithm*"})
        block = next(block for block in algorithms if r"\label{alg:muster}" in block)
        if mutation == "algorithm_parent_multiplier":
            changed = block.replace(r"$8\times$ the unit", r"$9\times$ the unit", 1)
        else:
            clause = "select the top units after the initial reranking pass"
            changed = block.replace(clause, "select the top units after reranking", 1)
            changed = changed.replace(r"\begin{algorithmic}", clause + "\n" + r"\begin{algorithmic}", 1)
    assert changed != block
    with pytest.raises(source_checker.CheckError, match=error):
        source_checker.compare_previous_revision(previous, current.replace(block, changed, 1))


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
