"""The harness's reranker score cache, read-only (harness/lme_pf29.py:1774-1848).

A file is named `<corpus signature>__<knob digest>.json`. The corpus signature hashes the exact unit list, so a
cut that names an existing file has produced, byte for byte, the units the published run scored.
"""
import hashlib
import json
import os

RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
# The knob string of every final leg except FactConsolidation (len_legs.json env: CEML=2048 CEBS=16 CE_SUBMAX=0).
KNOBS_FINAL = "CEML=2048|CEBS=16|CE_SUBMAX=0|CE_SUB_TOK=|CE_SLIDE=|name=" + RERANKER_NAME


def ce_sig(units, name=RERANKER_NAME):
    h = hashlib.sha1()
    h.update(("ce1|%s|%d\n" % (name, len(units))).encode())
    for t in units:
        h.update(hashlib.sha1(t.encode("utf-8", "replace")).digest())
    return h.hexdigest()[:16]


def knob_digest(knobs=KNOBS_FINAL):
    return hashlib.sha1(knobs.encode()).hexdigest()[:8]


def qkey(query):
    return hashlib.sha1(query.encode("utf-8", "replace")).hexdigest()[:16]


class LegacyCECache:
    """{question key: {unit index: score}} for one corpus, from a directory of harness cache files."""

    def __init__(self, directory, *, knobs=KNOBS_FINAL):
        self.directory, self.knobs, self.suffix = directory, knobs, knob_digest(knobs)

    def path_for(self, units):
        return os.path.join(self.directory, "%s__%s.json" % (ce_sig(units), self.suffix))

    def has(self, units):
        return os.path.exists(self.path_for(units))

    def load(self, units):
        p = self.path_for(units)
        if not os.path.exists(p):
            return {}
        return json.load(open(p))
