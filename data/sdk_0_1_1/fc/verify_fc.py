#!/usr/bin/env python3
"""Offline stdlib verifier for the frozen FC answer receipts. Does not generate answers."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
import string
import sys

def norm(t):
    t = str(t).lower()
    t = "".join(c for c in t if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", t).split())

def subem(pred, golds):
    p = norm(pred)
    if isinstance(golds, (str, int, float)) or golds is None:
        golds = [golds]
    return float(any(norm(g) in p for g in golds))


def main():
    base = Path(__file__).resolve().parent
    manifest = json.loads((base / 'MANIFEST.json').read_text())
    for name, item in manifest['files'].items():
        raw = (base / name).read_bytes()
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], ('file mismatch', name)
    rows = [json.loads(line) for line in (base / 'fc_answers.jsonl').read_bytes().splitlines() if line.strip()]
    assert len(rows) == 4800
    allowed = {'qid', 'dataset', 'hop', 'reader', 'arm', 'gold_aliases', 'visible_answer', 'first10_o200k', 'paper_cut_correct', 'input_sha256', 'context_sha256', 'answer_scope', 'raw_answer_sha256'}
    groups = defaultdict(dict)
    identities = set()
    scopes = Counter()
    aliases_by_qid = {}
    tokenizer = None
    tokenized = 0
    assert sys.argv[1:] in ([], ['--check-tokenization']), 'usage: verify_fc.py [--check-tokenization]'
    if sys.argv[1:]:
        import socket
        def no_network(*args, **kwargs):
            raise RuntimeError('Optional token check requires a preinstalled tiktoken and local o200k cache; network is disabled')
        socket.socket.connect = no_network
        socket.create_connection = no_network
        import tiktoken
        tokenizer = tiktoken.get_encoding('o200k_base')
    for row in rows:
        assert set(row) == allowed
        key = row['dataset'], row['reader'], row['arm'], row['qid']
        assert key not in identities
        identities.add(key)
        assert row['dataset'] in ('fc', 'k262') and row['reader'] in ('official', 'glm', 'luna') and row['arm'] in ('frozen010', 'v15')
        assert row['hop'] == ('mh' if '_mh_' in row['qid'] else 'sh')
        assert row['gold_aliases'] and all(norm(g) for g in row['gold_aliases'])
        qkey = row['dataset'], row['qid']
        if qkey in aliases_by_qid:
            assert aliases_by_qid[qkey] == row['gold_aliases']
        aliases_by_qid[qkey] = row['gold_aliases']
        for field in ('input_sha256', 'context_sha256', 'raw_answer_sha256'):
            assert re.fullmatch('[0-9a-f]{64}', row[field])
        scope = row['answer_scope']
        assert scope in ('full_visible', 'first10_only_long', 'first10_only_sensitive')
        assert (row['visible_answer'] is None) == (scope != 'full_visible')
        if tokenizer is not None and row['visible_answer'] is not None:
            assert tokenizer.decode(tokenizer.encode(row['visible_answer'])[:10]).strip() == row['first10_o200k'], ('token cut mismatch', key)
            tokenized += 1
        correct = bool(subem(row['first10_o200k'], row['gold_aliases']))
        assert correct == row['paper_cut_correct'], ('score mismatch', key)
        groups[(row['dataset'], row['reader'])][(row['arm'], row['qid'])] = row
        scopes[scope] += 1
    results = []
    for (dataset, reader), g in sorted(groups.items()):
        old = {q: r for (a, q), r in g.items() if a == 'frozen010'}
        new = {q: r for (a, q), r in g.items() if a == 'v15'}
        n = 600 if dataset == 'fc' else 200
        assert set(old) == set(new) and len(old) == n
        assert all(old[q]['gold_aliases'] == new[q]['gold_aliases'] for q in old)
        assert sum(r['hop'] == 'mh' for r in old.values()) == n // 2
        gain = sum(new[q]['paper_cut_correct'] and not old[q]['paper_cut_correct'] for q in old)
        loss = sum(old[q]['paper_cut_correct'] and not new[q]['paper_cut_correct'] for q in old)
        d = gain + loss
        p = min(1., 2 * sum(math.comb(d, i) for i in range(min(gain, loss) + 1)) / 2 ** d) if d else 1.
        results.append(dict(dataset=dataset, reader=reader, n=n, baseline_correct=sum(r['paper_cut_correct'] for r in old.values()),
                            candidate_correct=sum(r['paper_cut_correct'] for r in new.values()), only_candidate=gain, only_baseline=loss, paired_exact_p=p))
    expected = json.loads((base / 'expected_summary.json').read_text())
    assert len(results) == len(expected) == 6
    for result in results:
        target = next(x for x in expected if (x['dataset'], x['reader']) == (result['dataset'], result['reader']))
        assert all(result[k] == target[k] for k in result if k != 'paired_exact_p')
        assert abs(result['paired_exact_p'] - target['paired_exact_p']) < 1e-15
    print(json.dumps(dict(status='passed', rows=len(rows), answer_scopes=dict(scopes), tokenization_checked_rows=tokenized, results=results), indent=2))


if __name__ == '__main__':
    main()
