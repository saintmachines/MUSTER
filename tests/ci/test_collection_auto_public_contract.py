"""Collection Auto rows bind named cells, archived sources, API proof and current SDK bytes."""
from decimal import Decimal
from html.parser import HTMLParser
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = ('README.md', 'docs/index.html', 'docs/library/README.md')


@pytest.fixture(scope='module')
def tool():
    spec = importlib.util.spec_from_file_location('collection_auto_public_tool', ROOT / 'tools/make_collection_auto.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def records(tool):
    data = tool.load(ROOT / tool.DATA_PATH)
    return (data, tool.load(ROOT / tool.CANONICAL_PATH), tool.load(ROOT / data['input_verification']['receipt']))


@pytest.fixture
def document_root(tmp_path, tool):
    data = tool.load(ROOT / tool.DATA_PATH)
    paths = [*DOCUMENTS, tool.DATA_PATH, tool.CANONICAL_PATH, Path(data['input_verification']['receipt'])]
    if 'quality_verification' in data:
        paths.append(Path(data['quality_verification']['receipt']))
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    shutil.copytree(ROOT / 'muster', tmp_path / 'muster', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    return tmp_path


def extract(text, name='collection-auto'):
    begin, end = '<!-- BEGIN ' + name + ' -->', '<!-- END ' + name + ' -->'
    assert text.count(begin) == text.count(end) == 1
    return text.split(begin, 1)[1].split(end, 1)[0].strip('\n')


class CollectionTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.columns = []
        self.rows = []
        self.row = None
        self.target = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'table' and attrs.get('data-collection-auto') == '0.1.3':
            self.active = True
        if not self.active:
            return
        if tag == 'th' and 'data-collection-column' in attrs:
            self.columns.append(attrs['data-collection-column'])
        elif tag == 'tr' and 'data-collection-reader' in attrs:
            self.row = dict(id=attrs['data-collection-reader'], label='', metrics={})
            self.rows.append(self.row)
        elif tag == 'th' and attrs.get('scope') == 'row' and self.row is not None:
            self.target = 'label'
        elif tag == 'td' and self.row is not None:
            metric = attrs['data-collection-metric']
            assert metric not in self.row['metrics']
            self.row['metrics'][metric] = ''
            self.target = metric

    def handle_data(self, data):
        if self.active and self.row is not None and self.target:
            if self.target == 'label':
                self.row['label'] += data
            else:
                self.row['metrics'][self.target] += data

    def handle_endtag(self, tag):
        if tag in ('th', 'td'):
            self.target = None
        elif tag == 'tr':
            self.row = None
        elif tag == 'table':
            self.active = False


def test_public_data_proof_and_current_sdk_are_one_verified_contract(tool, records):
    data, canonical, proof = records
    assert data['schema'] == 'muster.collection_auto.mab.v2', 'The final faithful-rounds release needs measured changed-input coverage'
    assert tool.validate_data(data, canonical, proof, ROOT) is data
    assert len(data['readers']) == 5 and sum(len(row['cells']) for row in data['readers']) == 70
    assert proof['actual_nonFC_search_replays'] == 1471
    assert proof['exact_complete_reader_messages'] == 1450 and proof['changed_complete_reader_messages'] == 21
    assert len(proof['sdk_source_sha256']) == 37
    report = tool.generate(ROOT, check=True)
    assert report['status'] == 'current' and report['changed'] == []


def test_every_displayed_reader_and_metric_is_bound_in_each_destination(tool, records):
    data, _, _ = records
    expected = [(row['id'], row['label'], {metric: tool.display(row['metrics'][metric]) for metric in tool.METRICS})
                for row in data['readers']]
    landing = extract((ROOT / 'docs/index.html').read_text())
    parser = CollectionTable()
    parser.feed(landing)
    assert parser.columns == list(tool.METRICS)
    assert [(row['id'], row['label'], row['metrics']) for row in parser.rows] == expected
    assert 'data-reader=' not in landing and 'data-result-scope=' not in landing
    for relative in ('README.md', 'docs/library/README.md'):
        block = extract((ROOT / relative).read_text())
        table_rows = [line for line in block.splitlines() if line.startswith('| ')]
        assert table_rows[0].split('|')[1:-1] == [' Reader ', *[' ' + metric + ' ' for metric in tool.METRICS]]
        assert [[part.strip() for part in line.split('|')[1:-1]] for line in table_rows[1:]] == [
            [label, *[values[metric] for metric in tool.METRICS]] for _, label, values in expected]
        assert all(url.startswith('https://github.com/saintmachines/MUSTER/') for url in re.findall(r'\]\(([^)]+)\)', block))
        assert 'neural FC search was not rerun' in block
        assert 'Input replay alone does not measure answer quality' in block
        assert '279 archived judgments' in block and '21 changed inputs' in block
        assert 'ordinary `build` remains one store of records' in block


def test_decimal_half_up_preserves_the_published_two_decimal_contract(tool):
    assert tool.display(Decimal('47.525')) == '47.53'
    assert tool.display(Decimal('49.995')) == '50.00'
    for invalid in (True, Decimal('NaN'), Decimal('Infinity'), -1, 101):
        with pytest.raises(tool.ContractError):
            tool.display(invalid)


@pytest.mark.parametrize('field', ['label', 'request_model'])
def test_reader_identity_cannot_be_reassigned_to_another_existing_row(tool, records, field):
    data, canonical, proof = records
    data['readers'][0][field] = data['readers'][1][field]
    with pytest.raises(tool.ContractError, match='Reader label/model'):
        tool.validate_data(data, canonical, proof, ROOT)


def test_cell_swap_is_rejected_even_when_competency_and_overall_are_unchanged(tool, records):
    data, canonical, proof = records
    cells = data['readers'][0]['cells']
    current_ar = tool.recompute(cells)['AR']
    cells['shqa']['percent'], cells['mhqa']['percent'] = cells['mhqa']['percent'], cells['shqa']['percent']
    assert tool.recompute(cells)['AR'] == current_ar
    with pytest.raises(tool.ContractError, match='Canonical cell differs'):
        tool.validate_data(data, canonical, proof, ROOT)


def canonical_to_cells(row):
    return {cell: dict(percent=entry['native_percent']) for cell, entry in row['cells'].items()}


def test_metric_column_cannot_take_a_number_that_exists_under_another_column(tool, records):
    data, canonical, proof = records
    data['readers'][0]['metrics']['TTL'] = data['readers'][0]['metrics']['AR']
    with pytest.raises(tool.ContractError, match='Canonical aggregate differs'):
        tool.validate_data(data, canonical, proof, ROOT)


@pytest.mark.parametrize('mutation', ['question-count', 'repeat-count', 'reader-order', 'missing-cell'])
def test_reader_cell_denominators_and_order_are_fixed(tool, records, mutation):
    data, canonical, proof = records
    if mutation == 'question-count':
        data['readers'][1]['cells']['lme']['n'] = 299
    elif mutation == 'repeat-count':
        data['readers'][2]['cells']['det']['repeats'] = 1
    elif mutation == 'reader-order':
        data['readers'][0], data['readers'][1] = data['readers'][1], data['readers'][0]
    else:
        del data['readers'][3]['cells']['fcsh']
    with pytest.raises(tool.ContractError):
        tool.validate_data(data, canonical, proof, ROOT)


@pytest.mark.parametrize('field', ['exact_complete_messages', 'exact_selection_ids_and_pools', 'exact_contexts'])
def test_one_proof_cell_mismatch_cannot_hide_behind_the_1471_summary(tool, records, field):
    data, canonical, proof = records
    proof['cells']['lme'][field] -= 1
    with pytest.raises(tool.ContractError, match='lme/'):
        tool.validate_data(data, canonical, proof, ROOT)


@pytest.mark.parametrize('mutation', ['default-sample', 'resource-override', 'session-adapter', 'new-reader', 'new-quality', 'fc-search', 'fc-trace', 'fc-function'])
def test_reuse_scope_cannot_be_promoted_to_new_or_overridden_measurement(tool, records, mutation):
    data, canonical, proof = records
    if mutation == 'default-sample':
        data['sample_default'] = None
    elif mutation == 'resource-override':
        proof['benchmark_reference_chunk_overrides'] = 1
    elif mutation == 'session-adapter':
        proof['explicit_LME_adapter'] = True
    elif mutation == 'new-reader':
        proof['new_reader_calls'] = 1
    elif mutation == 'new-quality':
        proof['new_answer_quality_measurement'] = True
    elif mutation == 'fc-search':
        data['input_verification']['FC_search_recomputed'] = True
    elif mutation == 'fc-trace':
        proof['fc']['cells']['mg_fcsh_struct']['context_bytes_equal'] = 99
    else:
        proof['fc']['unchanged_function_ast_sha256']['memory.py:search'] = '0' * 64
    with pytest.raises(tool.ContractError):
        tool.validate_data(data, canonical, proof, ROOT)


@pytest.mark.parametrize('mutation', ['code-byte', 'extra-code-file', 'unlisted-proof-file'])
def test_proof_is_bound_to_the_exact_current_sdk_tree(tool, document_root, mutation):
    data = tool.load(document_root / tool.DATA_PATH)
    canonical = tool.load(document_root / tool.CANONICAL_PATH)
    proof = tool.load(document_root / data['input_verification']['receipt'])
    if mutation == 'code-byte':
        path = document_root / 'muster/memory.py'
        path.write_bytes(path.read_bytes() + b'\n# source changed\n')
    elif mutation == 'extra-code-file':
        (document_root / 'muster/unlisted.py').write_text('value = 1\n')
    else:
        del proof['sdk_source_sha256']['muster/collection.py']
    with pytest.raises(tool.ContractError, match='SDK37'):
        tool.validate_data(data, canonical, proof, document_root)


@pytest.mark.parametrize('mutation', ['original-score', 'old-auto-score', 'provenance'])
def test_historical_score_and_response_metadata_are_preserved(tool, records, mutation):
    data, canonical, proof = records
    if mutation == 'original-score':
        data['readers'][0]['original_implementation']['Overall'] += Decimal('.01')
    elif mutation == 'old-auto-score':
        data['readers'][0]['previous_sdk_0_1_2_per_store_auto']['Overall'] += Decimal('.01')
    else:
        data['historical_scope']['source_provenance']['primary_metric_sha256'] = '0' * 64
    with pytest.raises(tool.ContractError, match='Historical'):
        tool.validate_data(data, canonical, proof, ROOT)


def test_generation_preserves_old_native_blocks_and_check_is_read_only(tool, document_root):
    historical = {name: extract((document_root / name).read_text(), 'native-mab') for name in DOCUMENTS[:2]}
    landing = document_root / 'docs/index.html'
    text = landing.read_text()
    block = extract(text)
    landing.write_text(text.replace(block, block.replace('Source-derived collection configuration.', 'A stale title.'), 1))
    before_check = {name: (document_root / name).read_bytes() for name in DOCUMENTS}
    report = tool.generate(document_root, check=True)
    assert report['status'] == 'stale' and report['changed'] == ['docs/index.html']
    assert before_check == {name: (document_root / name).read_bytes() for name in DOCUMENTS}
    tool.generate(document_root)
    assert tool.generate(document_root, check=True)['status'] == 'current'
    assert historical == {name: extract((document_root / name).read_text(), 'native-mab') for name in DOCUMENTS[:2]}


def test_first_insert_and_library_draft_replacement_are_scoped(tool, document_root):
    html_path, library_path = document_root / 'docs/index.html', document_root / 'docs/library/README.md'
    remove = re.compile(re.escape(tool.BEGIN) + r'.*?' + re.escape(tool.END) + r'\n?', re.S)
    html_before = remove.sub('', html_path.read_text())
    html_path.write_text(html_before)
    draft = 'The separate [0.1.3 collection Auto draft](https://github.com/saintmachines/MUSTER#collection-auto) is awaiting the input proof.'
    library = library_path.read_text()
    library_path.write_text(re.sub(re.escape(tool.BEGIN) + r'.*?' + re.escape(tool.END), lambda _: draft, library, flags=re.S))
    report = tool.generate(document_root)
    assert report['changed'] == ['docs/index.html', 'docs/library/README.md']
    updated = html_path.read_text()
    assert updated.index(tool.BEGIN) < updated.index(tool.NATIVE_BEGIN)
    assert remove.sub('', updated) == html_before
    assert draft not in library_path.read_text()
    assert tool.generate(document_root, check=True)['status'] == 'current'


def test_generation_failure_rolls_back_every_replaced_document(tool, document_root, monkeypatch):
    for relative in DOCUMENTS:
        path = document_root / relative
        text = path.read_text()
        block = extract(text)
        path.write_text(text.replace(block, block.replace('For FC, 200 frozen selection traces', 'Stale frozen traces', 1), 1))
    data = tool.load(document_root / tool.DATA_PATH)
    updates = tool.planned_updates(document_root, data)
    assert len(updates) == 3
    before = {relative: (document_root / relative).read_bytes() for relative in DOCUMENTS}
    replace = tool.os.replace
    calls = []

    def interrupted(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError('injected interrupted publication')
        return replace(source, destination)

    monkeypatch.setattr(tool.os, 'replace', interrupted)
    with pytest.raises(OSError, match='injected'):
        tool.atomic_write(updates)
    assert before == {relative: (document_root / relative).read_bytes() for relative in DOCUMENTS}
    assert not list(document_root.rglob('*.collection-auto-*'))


def test_concurrent_document_edit_is_detected_before_any_replace(tool, document_root):
    path = document_root / 'README.md'
    text = path.read_text()
    block = extract(text)
    path.write_text(text.replace(block, block.replace('For FC, 200 frozen selection traces', 'Stale frozen traces', 1), 1))
    updates = tool.planned_updates(document_root, tool.load(document_root / tool.DATA_PATH))
    path.write_bytes(path.read_bytes() + b'\nAn independent owner edit.\n')
    expected = path.read_bytes()
    with pytest.raises(tool.ContractError, match='changed during generation'):
        tool.atomic_write(updates)
    assert path.read_bytes() == expected


@pytest.mark.parametrize('mutation', ['old-answer', 'old-messages', 'duplicate-question', 'missing-question', 'wrong-judge-payload', 'wrong-source-qid'])
def test_changed_lme_score_coverage_is_bound_to_each_source_question(tool, document_root, mutation):
    data = tool.load(document_root / tool.DATA_PATH)
    canonical = tool.load(document_root / tool.CANONICAL_PATH)
    proof = tool.load(document_root / data['input_verification']['receipt'])
    quality_path = document_root / tool.QUALITY_PATH
    quality = tool.load(quality_path)
    changed = {tool.question_key(row) for row in quality['changed_input_inventory']}
    questions = quality['readers'][0]['questions']
    fresh = next(row for row in questions if tool.question_key(row) in changed)
    if mutation == 'old-answer':
        fresh['answer_source'] = 'archived'
    elif mutation == 'old-messages':
        fresh['reader_request_messages_sha256'] = fresh['old_messages_sha256']
    elif mutation == 'duplicate-question':
        questions[-1] = dict(questions[0])
    elif mutation == 'missing-question':
        questions.pop()
    elif mutation == 'wrong-judge-payload':
        fresh['judge_source'] = 'identical_payload_archive'
        fresh['archived_judge_request_sha256'] = '0' * 64
        assert fresh['judge_request_sha256'] != fresh['archived_judge_request_sha256']
    else:
        unchanged = next(row for row in questions if tool.question_key(row) not in changed)
        source_key = tool.question_key(unchanged)
        for reader in quality['readers']:
            row = next(row for row in reader['questions'] if tool.question_key(row) == source_key)
            row['qid'] += '.wrong-source'
    quality_path.write_text(json.dumps(quality, default=str, ensure_ascii=False) + '\n')
    # Rebind the outer file hash: rejection must inspect the source/score
    # semantics, rather than succeeding only because the JSON bytes changed.
    data['quality_verification']['receipt_sha256'] = hashlib.sha256(quality_path.read_bytes()).hexdigest()
    with pytest.raises(tool.ContractError):
        tool.validate_data(data, canonical, proof, document_root)


def test_question_identity_uses_source_position_and_keeps_the_original_qid(tool, records):
    _, _, proof = records
    rows = proof['lme_input_rows']
    rows[-1]['qid'] = rows[0]['qid']
    assert tool.question_key(rows[-1]) != tool.question_key(rows[0])
    tool.validate_proof_v3(proof, ROOT)
    rows[-1]['row'], rows[-1]['qi'] = rows[0]['row'], rows[0]['qi']
    rows[-1]['qid'] += '.another-id-at-the-same-position'
    with pytest.raises(tool.ContractError, match='positions are not unique'):
        tool.validate_proof_v3(proof, ROOT)


def test_mixed_score_schema_cannot_change_any_other_task_cell(tool, records):
    data, canonical, proof = records
    row = data['readers'][0]
    row['cells']['summ']['percent'] += Decimal('.01')
    row['metrics'] = tool.recompute(row['cells'])
    with pytest.raises(tool.ContractError, match='Canonical cell differs'):
        tool.validate_data(data, canonical, proof, ROOT)


@pytest.mark.parametrize('mutation', ['all-messages-equal', 'unknown-cpu-zero', 'cpu-progress-hidden', 'mps-zero', 'mps-hidden', 'final-cache-nonzero'])
def test_replay_scope_keeps_changed_inputs_and_actual_local_work_visible(tool, records, mutation):
    data, canonical, proof = records
    if mutation == 'all-messages-equal':
        proof['exact_complete_reader_messages'] = 1471
    elif mutation == 'unknown-cpu-zero':
        proof['local_cpu_warming']['stage']['encoder_calls'] = 0
    elif mutation == 'cpu-progress-hidden':
        proof['local_cpu_warming']['stage']['completed_inputs'] = 0
    elif mutation == 'mps-zero':
        proof['local_gpu_assistance']['retrieval_model_calls']['reranker_calls'] = 0
    elif mutation == 'mps-hidden':
        proof['local_gpu_assistance'] = None
    else:
        proof['final_cache_only_replay']['retrieval_model_calls']['encoder_encode_calls'] = 1
    with pytest.raises(tool.ContractError):
        tool.validate_data(data, canonical, proof, ROOT)
