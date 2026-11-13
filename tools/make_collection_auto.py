#!/usr/bin/env python3
"""Validate source-bound collection Auto score reuse and generate its three public blocks."""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import html
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path('docs/data/auto_0_1_3.json')
CANONICAL_PATH = Path('docs/data/native_mab_0_1_2.json')
PROOF_PATH = Path('data/auto_0_1_3/API_INPUT_IDENTITY.json')
REPLAY_PATH = Path('data/auto_0_1_3/API_INPUT_REPLAY.json')
QUALITY_PATH = Path('data/auto_0_1_3/LME_REFRESH.json')
BEGIN, END = '<!-- BEGIN collection-auto -->', '<!-- END collection-auto -->'
NATIVE_BEGIN = '<!-- BEGIN native-mab -->'
READERS = ('4o-mini', '4.1-mini', '5-mini', 'glmh', 'luna')
CELLS = ('shqa', 'mhqa', 'lme', 'evq', 'b77', 'trcC', 'clin', 'nlu', 'trcF', 'rec', 'summ', 'det', 'fcsh', 'fcmh')
COUNTS = dict(zip(CELLS, (100, 100, 300, 100, 100, 100, 100, 100, 100, 200, 100, 71, 100, 100)))
METRICS = ('AR', 'TTL', 'LRU', 'SF', 'Overall')
AGGREGATES = ('AR', 'MCC', 'TTL', 'LRU', 'SF', 'Overall')
API = 'muster.build_collection(source_texts, embed=False)'
FC_METHOD = 'frozen selection-trace rendering plus unchanged numbered search/function/data path; no new neural FC search'
# Verified function pins use the receipt's Python AST serialization, including
# the exclusion of the unrelated flat-render branch from Memory.render.
FC_FUNCTION_PINS = {
    'memory.py:search': '71ac4e348c021edd90910e43230a707318770a378a6e608de225232ec33752b6',
    'memory.py:render': '61a33a6f2a8ae36ce63b91bba7b44eb18a3ef507369ee4723755d18a8b4f35f0',
    'render.py:numbered': 'ffbb300d77b285282ba8b4643b355095ff7d3caed85da35d2a9ca59c53bbc52d',
}
FC_FILES = ('bpe.py', 'cut.py', 'index.py', 'search.py', 'stats.py', 'plan.py', 'records.py', 'record_serial.py')
FC_NUMBERED_FUNCTION_PINS = {
    **FC_FUNCTION_PINS,
    'cut.py:entities': '017aad50f83808b90b78042bee7ba9ca33226178ddeaa80f6a918d31eb9c733a',
    'cut.py:slot_of': '0486905bf785425881dfb7b59c3944d878b73fd3c6456807a2c454f90a2a88ac',
    'cut.py:_relation_tail': 'cf538221d4f436410990045e4be03dfb130f5020c5c5b01fbd50179a3050c4a1',
    'cut.py:line_relation': 'b4883ec85ddfcff64f5d9394651ec5b232a2bd44e96ee75b3e83cee852d8be2a',
    'cut.py:value_slots': '0b7b5e6f64b6e4a2b698d79b5961def1f31a665cf3f77b3eb5dffc01e76adae4',
    'cut.py:serial_value': '5228ee589b2321b0a8592009392d64fc1f611eb863d315abcc61d9478425e091',
    'cut.py:serial_text': '2192d33151485222efd065c8f8e51c0563133920b4e808f4faa8a3ec47d66359',
    'cut.py:_numbered': 'e77fa7677797962c6e8942eb3585703c386b0e41ac52ac6c5992f2c2a0bb74db',
}
FC_CUT_INVARIANTS = {
    'module_assignments': '96d86762c6e924b4c743c0eab1090bedcaf14a89eb025d8ea2708806f7526cdf',
    'numbered_entry_branch': 'a056d7d92e963bef07acc9517ee56b72e8ab96d6ce876a6b4a0a6a1d4f7a23cc',
}
GITHUB = 'https://github.com/saintmachines/MUSTER/blob/main/'


class ContractError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'), parse_float=Decimal)


def number(value):
    require(not isinstance(value, bool), 'Boolean metric')
    try:
        result = Decimal(str(value))
    except Exception as error:
        raise ContractError('Invalid metric') from error
    require(result.is_finite() and 0 <= result <= 100, 'Metric outside 0..100')
    return result


def display(value):
    return format(number(value).quantize(Decimal('.01'), rounding=ROUND_HALF_UP), '.2f')


def count(value, expected, label):
    require(type(value) is int and value == expected, 'Wrong ' + label)


def digest(value, label):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value), 'Invalid ' + label + ' SHA256')


def bounded_count(value, maximum, label):
    require(type(value) is int and 0 <= value <= maximum, 'Invalid ' + label + ' count')


def recompute(cells):
    mean = lambda names: sum((number(cells[name]['percent']) for name in names), Decimal(0)) / len(names)
    ar, mcc = mean(('shqa', 'mhqa', 'lme', 'evq')), mean(('b77', 'clin', 'nlu', 'trcC', 'trcF'))
    ttl, lru, sf = (mcc + number(cells['rec']['percent'])) / 2, mean(('summ', 'det')), mean(('fcsh', 'fcmh'))
    return dict(AR=ar, MCC=mcc, TTL=ttl, LRU=lru, SF=sf, Overall=(ar + ttl + lru + sf) / 4)


def current_sdk_hashes(root):
    package = root / 'muster'
    require(package.is_dir(), 'Missing SDK source tree')
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob('*')) if path.is_file() and '__pycache__' not in path.parts}


def validate_proof_v3(proof, root):
    require(proof['schema'] == 'muster.collection_auto.input_replay.v3'
            and proof['status'] == 'passed_input_replay', 'Input replay not passed')
    require(proof['sdk_version'] == '0.1.3' and proof['api'] == API, 'Wrong replay SDK/API')
    count(proof['strategy_overrides'], 0, 'replay strategy overrides')
    count(proof['benchmark_reference_chunk_overrides'], 0, 'replay resource overrides')
    require(proof['explicit_LME_adapter'] is False, 'Explicit session adapter in replay')
    count(proof['actual_nonFC_search_replays'], 1471, 'actual non-FC replays')
    count(proof['exact_complete_reader_messages'], 1450, 'same complete messages')
    count(proof['changed_complete_reader_messages'], 21, 'changed complete messages')
    require(set(proof['cells']) == set(CELLS[:-2]), 'Missing/extra replay cell')
    for cell, row in proof['cells'].items():
        n = COUNTS[cell]
        changed = 21 if cell == 'lme' else 0
        count(row['questions'], n, cell + '/questions')
        count(row['exact_contexts'], n - changed, cell + '/same contexts')
        count(row['exact_complete_messages'], n - changed, cell + '/same messages')
        count(row['changed_complete_messages'], changed, cell + '/changed messages')
        for field in ('exact_selection_ids', 'exact_selection_pools', 'exact_selection_ids_and_pools'):
            bounded_count(row[field], n, cell + '/' + field)
        ids, pools, both = (row[field] for field in
                            ('exact_selection_ids', 'exact_selection_pools', 'exact_selection_ids_and_pools'))
        require(max(0, ids + pools - n) <= both <= min(ids, pools), 'Inconsistent selection identity counts')
        for field in ('reference_bundle_sha256', 'candidate_bundle_sha256'):
            digest(row[field], cell + '/' + field)
    input_rows = {}
    for row in proof['lme_input_rows']:
        key = question_key(row)
        require(key not in input_rows, 'Duplicate replay LME source question')
        for field in ('old_messages_sha256', 'new_messages_sha256'):
            digest(row[field], 'replay LME ' + field)
        input_rows[key] = row
    count(len(input_rows), 300, 'replay LME source questions')
    require(len({key[:2] for key in input_rows}) == 300,
            'Replay LME source positions are not unique')
    if 'lme_original_question_bundle_sha256' in proof:
        digest(proof['lme_original_question_bundle_sha256'], 'LME original question bundle')
        for row in input_rows.values():
            require(isinstance(row['original_question_id'], str) and bool(row['original_question_id'])
                    and isinstance(row['qt'], str) and bool(row['qt']), 'Missing original LME question identity/task')
            for field in ('question_sha256', 'gold_sha256'):
                digest(row[field], 'original LME ' + field)
    count(sum(row['old_messages_sha256'] != row['new_messages_sha256'] for row in input_rows.values()),
          21, 'replay LME changed source messages')
    fc = proof['fc']
    require(fc['method'] == FC_METHOD, 'FC replay must retain frozen-trace scope')
    count(fc['rows'], 200, 'FC frozen-trace rows')
    require(set(fc['cells']) == {'mg_fcsh_struct', 'mg_fcmh_struct'}, 'Wrong FC replay cells')
    for cell, row in fc['cells'].items():
        count(row['stores'], 1, cell + '/stores')
        count(row['render_trace_replays'], 100, cell + '/render traces')
        count(row['context_bytes_equal'], 100, cell + '/context identity')
        require(row['operational_plan_equal'] is True and row['all_cut_fields_equal'] is True, 'FC structural mismatch')
        digest(row['source_sha256'], cell + '/source')
    require(fc['unchanged_function_ast_sha256'] == FC_NUMBERED_FUNCTION_PINS, 'FC numbered functions differ')
    require(fc['cut_numbered_invariants'] == FC_CUT_INVARIANTS, 'FC numbered entry/constants differ')
    require(set(fc['unchanged_files']) == set(FC_FILES) - {'cut.py'}, 'Wrong FC unchanged-file scope')
    digest(fc['frozen_FC_contexts_sha256'], 'FC frozen contexts')
    pinned = proof['sdk_source_sha256']
    require(len(pinned) == 37 and pinned == current_sdk_hashes(root), 'Input replay SDK37 does not match current source code')
    for filename, value in fc['unchanged_files'].items():
        require(pinned['muster/' + filename] == value, 'FC unchanged file differs: ' + filename)
    require(proof['new_encoder_calls'] is None and proof['new_reranker_calls'] is None,
            'Unknown warming counters must remain null')
    cpu = proof['local_cpu_warming']
    require(cpu['device'] == 'cpu' and cpu['stage']['status'] == 'stopped_by_work_guard', 'Wrong CPU warming scope')
    require(cpu['stage']['encoder_calls'] is None and cpu['stage']['reranker_calls'] is None,
            'Unknown CPU counters must remain null')
    count(cpu['stage']['required_inputs'], 300, 'CPU required inputs')
    count(cpu['stage']['completed_inputs'], 281, 'CPU completed inputs')
    count(cpu['stage']['upper_bound_new_embedding_texts'], 3, 'CPU embedding work guard')
    count(cpu['stage']['upper_bound_reranker_pairs'], 1000, 'CPU reranker work guard')
    count(cpu['paid_calls'], 0, 'CPU paid calls')
    count(cpu['MPS_jobs'], 0, 'CPU MPS jobs')
    digest(cpu['recovery_receipt_sha256'], 'CPU recovery receipt')
    digest(cpu['cache_transfer_sha256'], 'cache transfer receipt')
    gpu = proof['local_gpu_assistance']
    require(isinstance(gpu, dict), 'Missing recorded local GPU assistance')
    require(gpu['status'] == 'passed' and gpu['device'] == 'mps' and gpu['shared_lock_held'] is True,
            'Wrong local GPU assistance scope')
    count(gpu['checks_before_start'], 3, 'GPU resource checks')
    count(gpu['disk_floor_GiB'], 10, 'GPU disk admission floor')
    require(gpu['CPU_outputs_preserved'] is True, 'CPU outputs were not preserved')
    for field, expected in (('encoder_encode_calls', 0), ('texts_encoded', 0),
                            ('reranker_calls', 21), ('reranker_pairs', 442)):
        count(gpu['retrieval_model_calls'][field], expected, 'GPU ' + field)
    for field in ('paid_calls', 'new_reader_calls', 'new_judge_calls'):
        count(gpu[field], 0, 'GPU ' + field)
    digest(gpu['manifest_sha256'], 'GPU manifest')
    final = proof['final_cache_only_replay']
    require(set(final['retrieval_model_calls']) == {'encoder_encode_calls', 'texts_encoded', 'reranker_calls', 'reranker_pairs'},
            'Incomplete final cache-only counters')
    for field, value in final['retrieval_model_calls'].items():
        count(value, 0, 'final cache-only ' + field)
    digest(final['manifest_sha256'], 'final cache-only manifest')
    for field in ('new_reader_calls', 'new_judge_calls'):
        count(proof[field], 0, 'input replay ' + field)
    require(proof['new_answer_quality_measurement'] is False, 'Input replay scope must exclude quality calls')
    require('unchanged model/provider/request and scorer contracts' in proof['reuse_condition'], 'Missing reuse contract')


def validate_proof(proof, root):
    if proof.get('schema') == 'muster.collection_auto.input_replay.v3':
        return validate_proof_v3(proof, root)
    require(proof['schema'] == 'muster.collection_auto.input_identity.v1' and proof['status'] == 'passed', 'Input proof not passed')
    require(proof['sdk_version'] == '0.1.3' and proof['api'] == API, 'Wrong proof SDK/API')
    count(proof['strategy_overrides'], 0, 'proof strategy overrides')
    count(proof['benchmark_reference_chunk_overrides'], 0, 'proof reference chunk overrides')
    require(proof['explicit_LME_adapter'] is False, 'Explicit session adapter in proof')
    count(proof['actual_nonFC_search_replays'], 1471, 'actual non-FC replays')
    count(proof['exact_complete_reader_messages'], 1471, 'exact complete messages')
    require(set(proof['cells']) == set(CELLS[:-2]), 'Missing/extra non-FC proof cell')
    for cell, row in proof['cells'].items():
        for field in ('questions', 'exact_contexts', 'exact_complete_messages', 'exact_selection_ids_and_pools'):
            count(row[field], COUNTS[cell], cell + '/' + field)
        for field in ('reference_bundle_sha256', 'candidate_bundle_sha256'):
            digest(row[field], cell + '/' + field)
    fc = proof['fc']
    require(fc['method'] == FC_METHOD, 'FC proof must retain frozen-trace scope')
    count(fc['rows'], 200, 'FC frozen-trace rows')
    require(set(fc['cells']) == {'mg_fcsh_struct', 'mg_fcmh_struct'}, 'Wrong FC proof cells')
    for cell, row in fc['cells'].items():
        count(row['stores'], 1, cell + '/stores')
        count(row['render_trace_replays'], 100, cell + '/render traces')
        count(row['context_bytes_equal'], 100, cell + '/context identity')
        require(row['operational_plan_equal'] is True and row['all_cut_fields_equal'] is True, 'FC structural mismatch')
        digest(row['source_sha256'], cell + '/source')
    digest(fc['frozen_FC_contexts_sha256'], 'FC frozen contexts')
    require(set(fc['unchanged_files']) == set(FC_FILES), 'Incomplete FC unchanged-file proof')
    require(fc['unchanged_function_ast_sha256'] == FC_FUNCTION_PINS, 'FC numbered-function proof differs')
    pinned = proof['sdk_source_sha256']
    require(len(pinned) == 37 and pinned == current_sdk_hashes(root), 'Input proof SDK37 does not match current source code')
    for filename, value in fc['unchanged_files'].items():
        require(pinned['muster/' + filename] == value, 'FC unchanged file differs: ' + filename)
    for field in ('new_encoder_calls', 'new_reranker_calls', 'new_reader_calls', 'new_judge_calls'):
        count(proof[field], 0, 'proof ' + field)
    require(proof['new_answer_quality_measurement'] is False, 'Input replay is not a new quality measurement')
    require('unchanged model/provider/request and scorer contracts' in proof['reuse_condition'], 'Missing request/scorer reuse condition')


def question_key(row):
    require(type(row['row']) is int and row['row'] >= 0 and type(row['qi']) is int and row['qi'] >= 0,
            'Invalid source question position')
    require(isinstance(row['qid'], str) and bool(row['qid']), 'Missing source question ID')
    return row['row'], row['qi'], row['qid']


def binary_score(value):
    require(type(value) is int and value in (0, 1), 'LME score must be the official binary judgment')
    return value


def validate_lme_refresh(data, canonical, proof, root):
    """A full 300-Q merge per reader, with changed inputs bound to new requests.

    This receipt is produced from the executed answer/judge records. Input
    replay alone cannot supply it or turn a paper-settings target into a score.
    """
    verification = data['quality_verification']
    require(verification['receipt'] == QUALITY_PATH.as_posix(), 'Wrong LME quality receipt path')
    path = root / QUALITY_PATH
    require(hashlib.sha256(path.read_bytes()).hexdigest() == verification['receipt_sha256'], 'LME quality receipt bytes differ')
    quality = load(path)
    require(quality['schema'] == 'muster.collection_auto.lme_refresh.v1' and quality['status'] == 'complete',
            'LME quality coverage incomplete')
    require(quality['sdk_source_sha256'] == proof['sdk_source_sha256'], 'Quality receipt SDK differs')
    require(quality['input_replay_sha256'] == hashlib.sha256((root / REPLAY_PATH).read_bytes()).hexdigest(),
            'Quality receipt input replay differs')
    for field in ('reference_bundle_sha256', 'candidate_bundle_sha256'):
        require(quality[field] == proof['cells']['lme'][field], 'LME quality source bundle differs')
    count(quality['changed_questions'], 21, 'LME changed questions')
    sources = quality['source_artifact_sha256']
    require(isinstance(sources, dict) and bool(sources), 'Missing LME score source artifacts')
    for name, value in sources.items():
        require(isinstance(name, str) and bool(name) and not Path(name).is_absolute(), 'Invalid public source artifact name')
        digest(value, 'LME source artifact ' + name)
    inventory = {}
    for row in quality['changed_input_inventory']:
        key = question_key(row)
        require(key not in inventory, 'Duplicate changed source question')
        for field in ('old_messages_sha256', 'new_messages_sha256'):
            digest(row[field], 'changed ' + field)
        require(row['old_messages_sha256'] != row['new_messages_sha256'], 'Changed question has unchanged messages')
        inventory[key] = row
    count(len(inventory), 21, 'changed source question inventory')
    require(len({key[:2] for key in inventory}) == 21,
            'Changed source positions are not unique')
    expected_inputs = {question_key(row): row for row in proof['lme_input_rows']}
    expected_changed = {key: row for key, row in expected_inputs.items()
                        if row['old_messages_sha256'] != row['new_messages_sha256']}
    require(set(inventory) == set(expected_changed), 'LME quality changed questions differ from actual replay inputs')
    for key, row in inventory.items():
        require(all(row[field] == expected_changed[key][field] for field in ('old_messages_sha256', 'new_messages_sha256')),
                'LME changed inventory messages differ from actual replay inputs')
    require([row['id'] for row in quality['readers']] == list(READERS), 'LME quality reader identity/order differs')
    public = {row['id']: row for row in data['readers']}
    old = {row['id']: row for row in canonical['paper_configuration_readers']}
    all_keys = None
    for reader in quality['readers']:
        ident = reader['id']
        require(reader['request_model'] == old[ident]['request_model'], 'LME request model differs: ' + ident)
        for field in ('reader_contract_sha256', 'scorer_contract_sha256'):
            digest(reader[field], ident + '/' + field)
            require(reader[field] == reader['original_' + field], 'LME original request/scorer contract differs: ' + ident)
        rows = {}
        for row in reader['questions']:
            key = question_key(row)
            require(key not in rows, 'Duplicate LME score question: ' + ident)
            rows[key] = row
            before, after = binary_score(row['old_score']), binary_score(row['score'])
            for field in ('old_messages_sha256', 'new_messages_sha256', 'old_prediction_sha256', 'prediction_sha256',
                          'archived_score_record_sha256'):
                digest(row[field], ident + '/' + field)
            require(key in expected_inputs and all(row[field] == expected_inputs[key][field]
                    for field in ('old_messages_sha256', 'new_messages_sha256')),
                    'LME score question/messages differ from actual replay inputs')
            require(row['archived_score_source'] in sources, 'LME archived score source is unpinned')
            if key in inventory:
                source = inventory[key]
                require(all(row[field] == source[field] for field in ('old_messages_sha256', 'new_messages_sha256')),
                        'LME changed request belongs to another source question')
                require(row['answer_source'] in ('fresh', 'exact_request_archive'), 'Changed LME input reused its old answer')
                require(row['reader_request_messages_sha256'] == row['new_messages_sha256'], 'Fresh LME answer used old messages')
                for field in ('reader_request_sha256', 'reader_response_record_sha256', 'judge_request_sha256', 'judge_record_sha256'):
                    digest(row[field], ident + '/' + field)
                require(row['reader_response_source'] in sources and row['judge_record_source'] in sources,
                        'Changed LME response/judgment source is unpinned')
                require(row['judge_source'] in ('fresh', 'identical_payload_archive'), 'Changed LME judgment has no payload lineage')
                if row['judge_source'] == 'identical_payload_archive':
                    require(row['judge_request_sha256'] == row['archived_judge_request_sha256'],
                            'LME judge reuse payload differs')
            else:
                require(row['old_messages_sha256'] == row['new_messages_sha256'], 'Unlisted changed LME input')
                require(row['answer_source'] == row['judge_source'] == 'archived', 'Unchanged LME row changed its provenance')
                require(before == after and row['old_prediction_sha256'] == row['prediction_sha256'],
                        'Unchanged LME prediction/judgment changed')
        count(len(rows), 300, ident + '/merged LME questions')
        require(len({key[:2] for key in rows}) == 300,
                'Merged LME source positions are not unique')
        require(set(inventory) <= set(rows), 'Missing changed LME question coverage')
        require(set(rows) == set(expected_inputs), 'Merged LME questions differ from actual replay inputs')
        if all_keys is None:
            all_keys = set(rows)
        require(set(rows) == all_keys, 'LME reader question sets differ')
        old_percent = sum(row['old_score'] for row in rows.values()) * Decimal(100) / 300
        new_percent = sum(row['score'] for row in rows.values()) * Decimal(100) / 300
        require(abs(old_percent - number(old[ident]['cells']['lme']['native_percent'])) <= Decimal('1e-10'),
                'LME archived question scores do not reconstruct their canonical cell')
        require(abs(new_percent - number(reader['lme_percent'])) <= Decimal('1e-10'), 'Merged LME cell differs from question scores')
        require(number(public[ident]['cells']['lme']['percent']) == number(reader['lme_percent']),
                'Public LME cell differs from measured source coverage')
    return quality


def validate_data(data, canonical, proof, root):
    mixed = data.get('schema') == 'muster.collection_auto.mab.v2'
    require(data['schema'] in ('muster.collection_auto.mab.v1', 'muster.collection_auto.mab.v2')
            and data['sdk_version'] == '0.1.3', 'Wrong collection data schema/version')
    expected_status = 'mixed_archived_and_measured_scores' if mixed else 'input_identity_verified_archived_scores_reused'
    require(data['configuration'] == 'source_collection_auto' and data['status'] == expected_status, 'Wrong collection score scope')
    require(data['api'] == API and data['reference_scale_override'] is False and data['strategy_override'] is False
            and data['explicit_LME_adapter'] is False, 'Collection API must use default strategy/resource arguments')
    count(data['sample_default'], 3, 'default collection sample')
    benchmark = data['benchmark']
    count(benchmark['cells'], 14, 'cell denominator')
    count(benchmark['questions_per_reader'], 1671, 'question denominator')
    count(benchmark['raw_slots_per_reader'], 1813, 'raw-slot denominator')
    require(benchmark['cell_order'] == list(CELLS), 'Cell order differs')
    require(canonical['sdk_version'] == '0.1.2', 'Canonical historical SDK version differs')
    historical = canonical['paper_configuration_readers']
    require([row['id'] for row in historical] == list(READERS), 'Canonical reader identity/order differs')
    require([row['id'] for row in data['readers']] == list(READERS), 'Collection reader identity/order differs')
    previous = {row['id']: row['native'] for row in canonical['protocol_auto_readers']}
    for row, old in zip(data['readers'], historical):
        reader = row['id']
        require((row['label'], row['request_model']) == (old['label'], old['request_model']), 'Reader label/model differs: ' + reader)
        require(set(row['cells']) == set(CELLS), 'Missing/extra task cell: ' + reader)
        for cell, entry in row['cells'].items():
            count(entry['n'], COUNTS[cell], reader + '/' + cell + '/n')
            count(entry['repeats'], 3 if cell == 'det' else 1, reader + '/' + cell + '/repeats')
            if not mixed or cell != 'lme':
                require(number(entry['percent']) == number(old['cells'][cell]['native_percent']), 'Canonical cell differs: ' + reader + '/' + cell)
            else:
                number(entry['percent'])
            require(entry['n'] == old['cells'][cell]['n'] and entry['repeats'] == old['cells'][cell]['repeats'], 'Canonical cell denominator differs')
        count(sum(entry['n'] for entry in row['cells'].values()), 1671, reader + '/unique questions')
        count(sum(entry['n'] * entry['repeats'] for entry in row['cells'].values()), 1813, reader + '/raw slots')
        calculated = recompute(row['cells'])
        require(set(row['metrics']) == set(AGGREGATES), 'Aggregate columns differ: ' + reader)
        for metric in AGGREGATES:
            if not mixed or metric in ('MCC', 'TTL', 'LRU', 'SF'):
                require(number(row['metrics'][metric]) == number(old['native'][metric]), 'Canonical aggregate differs: ' + reader + '/' + metric)
            require(abs(number(row['metrics'][metric]) - calculated[metric]) <= Decimal('1e-10'), 'Cell aggregate differs: ' + reader + '/' + metric)
        require(row['original_implementation'] == old['original'], 'Historical original metadata differs: ' + reader)
        require(row['previous_sdk_0_1_2_per_store_auto'] == previous.get(reader), 'Historical per-store Auto metadata differs: ' + reader)
    history = data['historical_scope']
    require(history['source'] == CANONICAL_PATH.as_posix(), 'Historical source path differs')
    require(history['source_provenance'] == canonical['provenance'], 'Historical score provenance was changed')
    count(history['SDK_paper_readers_preserved'], 5, 'historical paper readers')
    count(history['SDK_previous_per_store_auto_readers_preserved'], 3, 'historical per-store readers')
    validation = data['input_verification']
    require(validation['receipt'] == (REPLAY_PATH if mixed else PROOF_PATH).as_posix(), 'Wrong input proof reference')
    count(validation['actual_nonFC_search_replays'], 1471, 'data actual non-FC replays')
    count(validation['exact_complete_reader_messages'], 1450 if mixed else 1471, 'data same complete messages')
    if mixed:
        count(validation['changed_complete_reader_messages'], 21, 'data changed complete messages')
    count(validation['FC_frozen_trace_render_replays'], 200, 'data FC frozen-trace replays')
    require(validation['FC_search_recomputed'] is False and validation['FC_invariance_method'] == FC_METHOD, 'FC trace reuse mislabeled as new search')
    if not mixed:
        for key in ('new_encoder_calls', 'new_reranker_calls', 'new_reader_calls', 'new_judge_calls'):
            count(validation[key], 0, 'data ' + key)
    else:
        require(validation['new_encoder_calls'] is None and validation['new_reranker_calls'] is None,
                'Public input metadata must preserve unknown CPU warming counters')
        for key in ('new_reader_calls', 'new_judge_calls'):
            count(validation[key], 0, 'input replay data ' + key)
    validate_proof(proof, root)
    if mixed:
        require(proof['schema'] == 'muster.collection_auto.input_replay.v3', 'Mixed scores require the measured input inventory')
        replay_path = root / REPLAY_PATH
        replay_hash = hashlib.sha256(replay_path.read_bytes()).hexdigest()
        digest(validation['receipt_sha256'], 'input replay file')
        require(validation['receipt_sha256'] == replay_hash, 'Input replay file differs from its recorded hash')
        recorded = load(replay_path)
        if proof != recorded:
            for cell in CELLS[:-2]:
                require(proof['cells'][cell] == recorded['cells'][cell],
                        cell + '/supplied replay differs from its recorded source file')
            require(False, 'Supplied replay differs from its recorded source file')
        validate_lme_refresh(data, canonical, proof, root)
    return data


INPUT_SCOPE = (
    'Pass one store to `muster.build(source)` and several independent source stores to '
    '`muster.build_collection(source_texts)`. Collection members stay separate; a list of strings passed to '
    'ordinary `build` remains one store of records. The SDK derives the shared retrieval and rendering plan '
    'from the first up to three supplied stores and checks every member; strategies are not selected by the user.'
)
REUSE_SCOPE = (
    'These rows reuse archived SDK 0.1.2 paper-settings responses and scores after input-identity verification '
    'through `muster.build_collection(source_texts, embed=False)`, with all strategy and reference-chunk arguments '
    'at their defaults and no explicit LongMemEval adapter. The replay matched all 1,471 non-FC complete reader '
    'messages and selection IDs/pools. For FC, 200 frozen selection traces reproduced context bytes under the '
    'unchanged numbered cut/search/function/data path; neural FC search was not rerun. No new encoder, reranker, '
    'reader or judge calls were made. Reuse retains the archived model/provider/request and scorer contracts.'
)
LIMIT_SCOPE = (
    'Each reader covers 1,671 questions and 1,813 raw response slots across 14 cells; DetectiveQA retains three reads. '
    'Encoder and reranker inputs are capped at 2,048 tokens. This input replay is not a new answer-quality '
    'measurement and does not establish results for other benchmarks or every single-store input.'
)


def evaluation_scope(data):
    if data['schema'] != 'muster.collection_auto.mab.v2':
        return REUSE_SCOPE, LIMIT_SCOPE
    reuse = (
        'The default `muster.build_collection(source_texts, embed=False)` replay compared 1,471 non-FC inputs: '
        '1,450 complete reader messages matched the SDK 0.1.2 paper-settings archive; 21 LongMemEval messages '
        'changed after unpaired source turns were preserved. For each of the five readers, the LongMemEval cell '
        'merges 279 archived judgments with evaluation of those 21 changed inputs under the original requested '
        'model/provider/request and scorer contracts. The other 13 task cells retain the canonical archived values. '
        'For FC, 200 frozen selection traces reproduced context bytes under the unchanged numbered '
        'cut/search/function/data path; neural FC search was not rerun. Local cache warming and fresh reader/judge '
        'work are recorded separately in the receipts; the final input replay used caches only.'
    )
    limits = (
        'Each reader covers 1,671 questions and 1,813 raw response slots across 14 cells; DetectiveQA retains three '
        'reads. Encoder and reranker inputs are capped at 2,048 tokens. Input replay alone does not measure answer '
        'quality. These rows combine archived identical-input results with the separately evaluated changed '
        'LongMemEval inputs; they do not establish results for other benchmarks or every single-store input. '
        'The archived actual judge snapshot was not stored; a matching requested alias does not establish the same snapshot.'
    )
    return reuse, limits


def source_links(data):
    proof = REPLAY_PATH if data['schema'] == 'muster.collection_auto.mab.v2' else PROOF_PATH
    links = [('All task cells and historical provenance', DATA_PATH), ('API input replay and scope', proof)]
    if data['schema'] == 'muster.collection_auto.mab.v2':
        links.append(('LongMemEval response and judgment coverage', QUALITY_PATH))
    return links


def markdown(data):
    reuse, limits = evaluation_scope(data)
    lines = ['<a id="collection-auto"></a>', '### SDK 0.1.3 collection Auto on MemoryAgentBench', '', INPUT_SCOPE, '',
             '| Reader | ' + ' | '.join(METRICS) + ' |', '|---|' + '---:|' * len(METRICS)]
    for row in data['readers']:
        lines.append('| ' + ' | '.join([row['label']] + [display(row['metrics'][name]) for name in METRICS]) + ' |')
    lines += ['', 'Original benchmark metrics, 0–100 scale. Overall averages AR, TTL, LRU and SF.', '',
              reuse, '', limits, '',
              ' · '.join('[' + label + '](' + GITHUB + path.as_posix() + ')' for label, path in source_links(data))]
    return '\n'.join(lines)


def prose_html(text):
    return ''.join('<code>' + html.escape(part[1:-1]) + '</code>' if part.startswith('`') and part.endswith('`')
                   else html.escape(part) for part in re.split(r'(`[^`]+`)', text))


def landing(data):
    reuse, limits = evaluation_scope(data)
    caption = ('Original benchmark metrics, 0–100 scale. Archived results with changed LongMemEval inputs re-evaluated.'
               if data['schema'] == 'muster.collection_auto.mab.v2' else
               'Original benchmark metrics, 0–100 scale. Archived responses reused after input verification.')
    lines = ['<section class="sec" id="collection-auto">', '<div class="wrap">',
             '<div class="eyebrow">SDK 0.1.3 · collection Auto · MemoryAgentBench</div>',
             '<h2>Source-derived collection configuration.</h2>', '<p class="lead">' + prose_html(INPUT_SCOPE) + '</p>',
             '<div class="table-wrap"><table class="t" data-collection-auto="0.1.3" style="min-width:640px">',
             '<caption class="small">' + caption + '</caption>',
             '<thead><tr><th scope="col">Reader</th>' + ''.join('<th scope="col" data-collection-column="' + name + '">' + name + '</th>' for name in METRICS) + '</tr></thead><tbody>']
    for row in data['readers']:
        lines.append('<tr data-collection-reader="' + row['id'] + '"><th scope="row">' + html.escape(row['label']) + '</th>'
                     + ''.join('<td class="n" data-collection-metric="' + name + '">' + display(row['metrics'][name]) + '</td>' for name in METRICS) + '</tr>')
    lines += ['</tbody></table></div>', '<p class="note">' + prose_html(reuse) + '</p>',
              '<p class="note">' + prose_html(limits) + '</p>',
              '<p class="note">' + ' · '.join('<a href="' + GITHUB + path.as_posix() + '">' + html.escape(label) + '</a>'
                                             for label, path in source_links(data)) + '</p>', '</div>', '</section>']
    return '\n'.join(lines)


def wrapped(content):
    return BEGIN + '\n' + content + '\n' + END


def replace_block(text, content):
    require(text.count(BEGIN) == text.count(END) == 1, 'Need exactly one collection-auto marker pair')
    start, end = text.index(BEGIN), text.index(END)
    require(start < end, 'Collection-auto markers reversed')
    return text[:start] + wrapped(content) + text[end + len(END):]


def planned_updates(root, data):
    updates = []
    for relative in ('README.md', 'docs/index.html', 'docs/library/README.md'):
        path = root / relative
        before = path.read_bytes()
        text = before.decode('utf-8')
        content = landing(data) if relative.endswith('.html') else markdown(data)
        if BEGIN in text or END in text:
            after = replace_block(text, content)
        elif relative == 'docs/index.html':
            require(text.count(NATIVE_BEGIN) == 1, 'Need native marker for collection block insertion')
            after = text.replace(NATIVE_BEGIN, wrapped(content) + '\n' + NATIVE_BEGIN, 1)
        elif relative == 'docs/library/README.md':
            pattern = re.compile(r'^The separate \[0\.1\.3 collection Auto draft\].*?(?=\n\n|\Z)', re.M | re.S)
            require(len(pattern.findall(text)) == 1, 'Need the library evaluation draft or collection marker')
            after = pattern.sub(lambda _: wrapped(content), text, count=1)
        else:
            raise ContractError('Missing collection-auto marker: ' + relative)
        if after.encode('utf-8') != before:
            updates.append(dict(path=path, before=before, after=after.encode('utf-8')))
    return updates


def atomic_write(updates):
    """Stage every destination before replacing any; restore completed writes on failure."""
    staged = []
    replaced = []
    try:
        for row in updates:
            path = row['path']
            fd, name = tempfile.mkstemp(prefix=path.name + '.collection-auto-', dir=path.parent)
            staged.append((row, Path(name)))
            with os.fdopen(fd, 'wb') as stream:
                stream.write(row['after'])
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(name, stat.S_IMODE(path.stat().st_mode))
        require(all(row['path'].read_bytes() == row['before'] for row, _ in staged), 'A destination changed during generation')
        for row, temporary in staged:
            os.replace(temporary, row['path'])
            replaced.append(row)
    except Exception:
        for row in reversed(replaced):
            fd, name = tempfile.mkstemp(prefix=row['path'].name + '.collection-rollback-', dir=row['path'].parent)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(row['before'])
            os.chmod(name, stat.S_IMODE(row['path'].stat().st_mode))
            os.replace(name, row['path'])
        raise
    finally:
        for _, temporary in staged:
            if temporary.exists():
                temporary.unlink()


def generate(root, check=False):
    root = Path(root)
    data, canonical = load(root / DATA_PATH), load(root / CANONICAL_PATH)
    mixed = data.get('schema') == 'muster.collection_auto.mab.v2'
    proof = load(root / (REPLAY_PATH if mixed else PROOF_PATH))
    validate_data(data, canonical, proof, root)
    updates = planned_updates(root, data)
    if not check:
        atomic_write(updates)
    report = dict(status='stale' if check and updates else 'updated' if updates else 'current',
                  readers=5, reader_cells=70, questions_per_reader=1671, raw_slots_per_reader=1813,
                  proof_sdk_files=37, exact_nonFC_complete_messages=1450 if mixed else 1471,
                  changed_nonFC_complete_messages=21 if mixed else 0, FC_frozen_trace_replays=200,
                  changed=[row['path'].relative_to(root).as_posix() for row in updates])
    if mixed:
        report.update(LME_refresh_readers=5, changed_LME_question_reader_score_units=105,
                      final_cache_only_model_calls=0, CPU_warming_counters='unknown; bounded, never zero')
    else:
        report['new_model_calls'] = 0
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        report = generate(args.root, args.check)
    except (ContractError, KeyError, OSError, ValueError) as error:
        print('Collection Auto contract: ' + str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 1 if report['status'] == 'stale' else 0


if __name__ == '__main__':
    sys.exit(main())
