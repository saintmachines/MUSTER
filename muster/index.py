"""The model-free index of a store: entities, the entity graph, IDF, slots, positions and BM25.

Every structure is built with the harness's own expressions (src/real_e1_conflict_killer.py:26-54,
harness/lme_pf29.py:2466-2525, 2592-2596; core2.py:40-61, 429-438). In exact mode the sets are kept as sets,
because their iteration order under the paper's hash seed is part of the published output; native mode reads them
in a canonical order wherever order can matter (muster/search.py).
"""
import collections
import math
import re

_CAP = re.compile(r"\b([A-Z][\w.'-]+(?:\s+(?:of|the|and|de|van|von|di)?\s*[A-Z][\w.'-]+)*)\b")
HUB_FRAC = 0.05
ORDER_DISTINCT = 0.99


def entities(text):
    """Capitalised spans of more than two characters, as a set."""
    return {m.strip() for m in _CAP.findall(text or "") if len(m.strip()) > 2}


def build_entity_index(fact_ents, n_facts, hub_frac=HUB_FRAC):
    """({entity: set of units}, hubs). An entity in more than max(3, 5% of units) units is a hub and leaves the index."""
    ent_facts = {}
    for i, es in enumerate(fact_ents):
        for e in es:
            ent_facts.setdefault(e, set()).add(i)
    hub_cut = max(3, int(hub_frac * n_facts))
    hubs = {e for e, fs in ent_facts.items() if len(fs) > hub_cut}
    return {e: fs for e, fs in ent_facts.items() if e not in hubs}, hubs


def slotkey(a):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(a).lower()).split())


def btok(t):
    """BM25 tokens: lower case, punctuation to spaces, split on white space (BM25LC=1)."""
    t = t.lower()
    t = "".join(c if (c.isalnum() or c.isspace()) else " " for c in t)
    return t.split()


def newest_per_parent(pool, parent_of, order_of):
    newest = {}
    for i in pool:
        k = parent_of[i]
        if k is not None and (k not in newest or order_of[i] > order_of[newest[k]]):
            newest[k] = i
    return newest


class Index:
    """Built once per store version; a pure function of its units, keys, parents and order."""

    def __init__(self, units, keys, parent, order, *, has_order=False):
        N = self.N = len(units)
        self.units, self.keys, self.parent = list(units), list(keys), list(parent)
        self.r_ents = [entities(t) for t in self.keys]
        self.ef, self.hubs = build_entity_index(self.r_ents, N)
        self.idf = {e: math.log(N / max(len(v), 1)) for e, v in self.ef.items()}
        self.r_slot = [slotkey(t) for t in self.keys]
        sm = collections.defaultdict(list)
        for i, s in enumerate(self.r_slot):
            if s:
                sm[s].append(i)
        self.newest_of_slot = {k: max(v) for k, v in sm.items()}
        # the order a structural resolver may trust: declared (numbered logs), and separating almost every unit
        self.order = list(order) if has_order else list(range(N))
        order_ok = has_order and len(set(self.order)) >= ORDER_DISTINCT * N
        self.structural = bool(order_ok and len(set(self.parent)) < N)
        self.children_of = {}
        for i, p in enumerate(self.parent):
            if p is not None:
                self.children_of.setdefault(p, []).append(i)
        self.pos_of, self.at = {}, {}
        cnt = collections.Counter()
        for i, si in enumerate(self.parent):
            self.pos_of[i] = cnt[si]
            self.at[(si, cnt[si])] = i
            cnt[si] += 1
        self.newest_row = newest_per_parent(range(N), self.parent, self.order) if self.structural else {}
        self.adj_slot = {}
        if self.structural:
            ent_slots = collections.defaultdict(set)
            for s2, i2 in self.newest_row.items():
                for e2 in (self.r_ents[i2] - self.hubs):
                    ent_slots[e2].add(s2)
            cut = max(3, int(HUB_FRAC * max(len(self.newest_row), 1)))
            ent_slots = {e2: v2 for e2, v2 in ent_slots.items() if len(v2) <= cut}
            self.adj_slot = collections.defaultdict(set)
            for e2, ss2 in ent_slots.items():
                for s2 in ss2:
                    self.adj_slot[s2] |= (ss2 - {s2})
        self._bm25 = None

    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([btok(t) for t in self.units])
        return self._bm25
