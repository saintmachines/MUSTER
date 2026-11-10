"""Exact, source-pinned five-reader MAB document contract. No benchmark imports."""
from decimal import Decimal, ROUND_HALF_UP
from pathlib import PurePosixPath
import re

SCHEMA = 'muster.native_mab.v2'
PAPER_REVISION = 'v9'
PAPER_REVISION = 'v9'
SOURCE_RECEIPT_SCHEMA = 'muster.paper.representative_mab.v9'
PUBLIC_B_SHA = '37e5eaf1346c222ce4807a705d9e14527058368d8cc2f2ca240298b99f488003'
DIRECT_V6_SHA = '926e14bc0af1353ca4b121a00b5ef377a6e886c9425feaec216939aef417227c'
SDK_MARKER = 'Current implementation: SDK (paper settings), library 0.1.2'
AUTO_MARKER = 'SDK: per-store automatic configuration'
PP_MARKER = 'Current SDK, paper settings'
READ_CAPTION_BEFORE = 'one query embedding, one rerank and one reader call over units embedded once per store.'
READ_CAPTION_AFTER = "one query embedding, an initial rerank and one reader call over units embedded once per store. The SDK's numbered-fact walk can add rerank rounds."
CELLS = ('shqa','mhqa','lme','evq','b77','trcC','clin','nlu','trcF','rec','summ','det','fcsh','fcmh')
COUNTS = dict(zip(CELLS, (100,100,300,100,100,100,100,100,100,200,100,71,100,100)))
READERS = {
    '4o-mini': ('GPT-4o-mini', 'gpt-4o-mini-2024-07-18', 'tab:expanded4o-mini'),
    '4.1-mini': ('GPT-4.1-mini', 'gpt-4.1-mini-2025-04-14', 'tab:expanded4.1-mini'),
    '5-mini': ('GPT-5-mini', 'gpt-5-mini', 'tab:expanded5-mini'),
    'glmh': ('GLM-5.3-Flash', 'z-ai/glm-5.3-flash', 'tab:expandedfrontier'),
    'luna': ('GPT-6 Luna', 'openai/gpt-6-luna', 'tab:expandedfrontier'),
}
AGGREGATES = ('AR','MCC','TTL','LRU','SF','Overall')
SOURCE_SHA = {
    'primary': 'b59a231b51a115449da1cbc6c144ffdffa5b07be023bc3266d41f6183395f204',
    'auto': '77b3d497e8d2dd61842bcf159e9c81ab1722d141df8ddffd9ac94f9612e144f6',
    'glmh': '563ed3fd409ae2f5a7dad5e7a8389ab44096051b8f3513455396dae0e192d566',
    'luna': '8d9cf8f24539077799ad70484b961fa629cd836704009a432d9f0baac1301dd8',
    'pp': '4a3a3cde9962a6e70e8a9473163647ea45ba73c95e0cd23f389198228b25c0fc',
}
PP_FC_BINDING_SHA = '8401cb8718d73737021d274214d3964bc4faf66b4098bd4f0c86093aaf66539b'
# Each position is tied to CELLS, never to the existence of a number elsewhere.
CELL_LOCKS = {
 '4o-mini': {
  'original': (92,65,56.333333333333336,82,96,70,90,88,65,13.25,22.237952357760168,66.19718309859155,92,37),
  'native': (92,65,56.333333333333336,83,96,70,90,88,65,13.25,21.85733719648448,66.19718309859155,93,60)},
 '4.1-mini': {
  'original': (90,76,61.66666666666667,82,97,73,89,85,49,15.666666666666664,32.32982824165706,70.89201877934272,93,42),
  'native': (90,76,61.66666666666667,81,97,73,89,85,49,15.666666666666664,32.62280037426967,70.89201877934272,94,51)},
 '5-mini': {
  'original': (93,78,71,82,95,84,94,89,59,11.666666666666666,44.46786325624772,79.34272300469483,94,42),
  'native': (93,78,71,82,95,84,94,89,59,12.166666666666666,44.47562760034487,79.34272300469483,95,59)},
 'glmh': {
  'original': (97,77,72,93,97,92,96,90,80,16.166666666666668,42.92399836143513,84.03755868544602,93,45),
  'native': (97,77,72,92,97,92,96,90,80,15.666666666666664,42.622937058347254,84.03755868544602,94,61)},
 'luna': {
  'original': (94,77,74.66666666666667,80,96,92,98,89,69,15.5,37.450325647289944,78.4037558685446,92,41),
  'native': (94,77,74.66666666666667,81,96,92,98,89,69,15.5,38.18560061870745,78.4037558685446,94,57.99999999999999)},
}
AUTO_LOCK = (89,66,56.333333333333336,77,94,66,89,87,60,12.083333333333336,13.578176715428778,66.19718309859155,93,60)
PP_LOCK = (89,69,54,70,94,66,91,86,68,14.333333333333334,20.054907438800903,58.685446009389665,93,60)
OLD_PAPER_HASHES = {
 'cabe57f083623fd6abcc93ea7fbe3724139fb4be7fcfc8e0220861233b406e7e',
 '0f4ccc3c793e4c3e2a18558e7e92e224d0649f350b40ffc2a0b3405a5dce3279',
 '8c5bb0c5178dc68022f307fafe47c03ffe152ff772fd9cf39a7647c5d645ca78',
 # v7 source and artifacts remain valid historical evidence, not current artifacts.
 '3b5030bf7af82f4616e86f0cffca6b09547463303f76def6278c022eee70b64b',
 '6896e3379fdcbadb9f62638c3a80874e00a9de2b95b2293fe557aad3f613b920',
 '6694014de38ef0678ed8883cf3b543b8486e08534d63ac4388f3e7a4a2173b84',
 '9896afaaa81b2cac4ded183db675eb4fff3ec8e2df46c9bed995a56b8b273f8e',
}
TABLE_SOURCES = {
 'tab:board': 'five SDK paper-settings rows and one SDK automatic-settings row',
 'tab:gate': 'original implementation rows plus source-bound SDK PP/final block',
 'tab:expanded4o-mini': '4o-mini original / SDK paper / SDK automatic / original Infini-style',
 'tab:expanded4.1-mini': '4.1-mini original / SDK paper / original Infini-style',
 'tab:expanded5-mini': '5-mini original / SDK paper / original Infini-style',
 'tab:expandedfrontier': 'GLM and Luna original / SDK paper cells',
}


class ContractError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def number(value):
    require(not isinstance(value, bool), 'Boolean metric')
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    require(result.is_finite() and 0 <= result <= 100, 'Metric outside0..100')
    return result


def display(value, places=2):
    return format(number(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP), f'.{places}f')


def recompute(cells):
    avg = lambda names: sum((number(cells[c]) for c in names), Decimal(0)) / len(names)
    mcc, ar = avg(('b77','clin','nlu','trcC','trcF')), avg(('shqa','mhqa','lme','evq'))
    ttl = (mcc + number(cells['rec'])) / 2
    lru, sf = avg(('summ','det')), avg(('fcsh','fcmh'))
    return dict(AR=ar, MCC=mcc, TTL=ttl, LRU=lru, SF=sf, Overall=(ar + ttl + lru + sf) / 4)


def validate_record(record, reader, automatic=False):
    require(reader in READERS and record['id'] == reader, 'Wrong reader identity')
    require(record['label'] == READERS[reader][0] and record['request_model'] == READERS[reader][1], 'Wrong reader label/model')
    require(set(record['cells']) == set(CELLS), 'Missing/extra task cell')
    for cell, count in COUNTS.items():
        entry = record['cells'][cell]
        require(type(entry['n']) is int and entry['n'] == count, f'{reader}/{cell}: n')
        require(type(entry['repeats']) is int and entry['repeats'] == (3 if cell == 'det' else 1), f'{reader}/{cell}: repeats')
    for view in ('original','native'):
        expected = CELL_LOCKS[reader][view]
        if automatic and view == 'native':
            require(reader == '4o-mini', 'Unmeasured automatic reader')
            expected = AUTO_LOCK
        values = {c: number(record['cells'][c][view + '_percent']) for c in CELLS}
        for c, locked in zip(CELLS, expected):
            require(values[c] == number(locked), f'{reader}/{view}/{c}: source-locked cell differs')
        computed = recompute(values)
        require(set(record[view]) == set(AGGREGATES), 'Wrong aggregate keys')
        for key in AGGREGATES:
            require(abs(number(record[view][key]) - computed[key]) <= Decimal('1e-10'), f'{reader}/{view}/{key}: aggregate')
    return record


def safe_path(value):
    require(isinstance(value, str), 'Artifact path must be a string')
    path = PurePosixPath(value)
    require(bool(re.fullmatch('[A-Za-z0-9_./-]+', value)), 'Unsafe paper path')
    require(not path.is_absolute() and '..' not in path.parts and path.parts[0] == 'paper', 'Non-public paper path')
    return value


def validate_paper(paper, require_final=False):
    require(paper['revision'] == PAPER_REVISION, 'Expected v9 paper metadata')
    if paper['status'] == 'awaiting_v9_final_receipt':
        require(not require_final, 'v9 final paper receipt is required')
        require(set(paper) == {'revision','status'}, 'Pending paper carries unverified metadata')
        return
    require(paper['status'] == 'verified_v9_final_receipt', 'Unknown paper receipt status')
    require(type(paper['pages']) is int and paper['pages'] > 0, 'Invalid v9 pages')
    for field, digest_field in (('tex','tex_sha256'),('pdf','clean_pdf_sha256'),('marked_pdf','marked_pdf_sha256'),('source_zip','source_zip_sha256')):
        safe_path(paper[field])
        digest = paper[digest_field]
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'Invalid artifact digest')
        require(digest not in OLD_PAPER_HASHES, 'Earlier paper artifact mislabeled as v9')
    safe_path(paper['baseline_tex'])
    require(paper['comparison_baselines']['public_B']['sha256'] == PUBLIC_B_SHA, 'Wrong public B baseline')
    require(paper['comparison_baselines']['direct_v6']['sha256'] == DIRECT_V6_SHA, 'Wrong direct v6 baseline')
    require(isinstance(paper['source_receipt_sha256'], str) and re.fullmatch('[0-9a-f]{64}', paper['source_receipt_sha256']), 'Invalid final v9 receipt digest')
    require(paper['allowed_modified_tables'] == list(TABLE_SOURCES), 'Unexpected paper change allowlist')
    require(paper['added_vs_public_B_tables'] == ['tab:expandedfrontier'], 'Only frontier may be added to public B')


def validate_data(data, require_final_paper=False):
    require(data['schema'] == SCHEMA and data['sdk_version'] == '0.1.2', 'Wrong data schema/version')
    b = data['benchmark']
    require(b['cells'] == 14 and b['questions_per_reader'] == 1671 and b['raw_slots_per_reader'] == 1813, 'Wrong benchmark denominators')
    require(b['cell_order'] == list(CELLS), 'Cell order differs')
    records = data['paper_configuration_readers']
    require([r['id'] for r in records] == list(READERS), 'Expected five distinct readers in fixed order')
    for r in records:
        validate_record(r, r['id'])
    validate_record(data['protocol_auto_4omini'], '4o-mini', automatic=True)
    provenance = data['provenance']
    require(provenance['primary_metric_sha256'] == SOURCE_SHA['primary'], 'Primary source pin differs')
    require(provenance['auto_metric_sha256'] == SOURCE_SHA['auto'], 'Auto source pin differs')
    require(provenance['frontier_merge_sha256'] == {k: SOURCE_SHA[k] for k in ('glmh','luna')}, 'Frontier merge pins differ')
    groups = provenance['response_accounting_groups']
    require(set(groups) == {'gpt_readers','frontier_readers'}, 'Accounting units must stay in separate groups')
    require(groups['gpt_readers']['readers'] == ['4o-mini','4.1-mini','5-mini'], 'GPT accounting readers')
    require(groups['frontier_readers']['readers'] == ['glmh','luna'], 'Frontier accounting readers')
    require(groups['gpt_readers']['fresh_validated_reader_response_rows'] == 678, 'GPT fresh count')
    require(groups['gpt_readers']['full_message_reused_question_reader_score_units'] == 4335, 'GPT reused score units')
    require(groups['frontier_readers']['fresh_reader_response_slots'] == 448, 'Frontier fresh slots')
    require(groups['frontier_readers']['reused_raw_response_slots'] == 3178, 'Frontier reused raw slots')
    require(data['paper_source_labels'] == TABLE_SOURCES, 'Unmapped or reassigned source table')
    pp = data['sdk_parent_priority']
    require(pp['status'] == 'complete_native_PP_14cell_quality' and pp['reader'] == '4o-mini', 'Incomplete PP source')
    require(set(pp['cells']) == set(CELLS), 'Incomplete PP cell vector')
    for cell, expected in zip(CELLS, PP_LOCK):
        require(number(pp['cells'][cell]) == number(expected), 'PP source-locked cell differs: ' + cell)
    comp = recompute(pp['cells'])
    require(set(pp['competencies']) == set(AGGREGATES), 'Wrong PP aggregate keys')
    for name in AGGREGATES:
        require(abs(comp[name] - number(pp['competencies'][name])) <= Decimal('1e-10'), f'PP/{name}: aggregate')
    require(pp['source_sha256'] == SOURCE_SHA['pp'], 'PP source pin differs')
    require(pp['FC_identity_sha256'] == PP_FC_BINDING_SHA, 'PP FC control-equivalence pin differs')
    validate_paper(data['paper'], require_final_paper)
    return data
