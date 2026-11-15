"""Search: the paper's three-channel selection (meaning, keywords, the entity graph over parents), then the read
budget U split between them by reserved floors.

Exact mode runs the harness's statements in the harness's order (harness/lme_pf29.py:2601-2770, 2862-2893,
3179-3370, 3400, 3553-3584, 3660-3664; core2.py:73-100, 103-198, 201-262, 294-295), so iteration over sets, float
sums and tie orders are the published ones. Native mode differs only at the points listed in docs/library/exact.md:
canonical iteration where the harness iterates a set (N1) and a stable dense sort (N2).
"""
import collections
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .errors import StoreRefused
from .index import btok, entities, plain_words

K1, H, DECAY, RCAP, RDEG, M_ANCH, SLOT_CE_CUT = 200, 4, 0.5, 100, 5, 2, 400


@dataclass(frozen=True)
class Selection:
    query: str
    U: int
    policy: str
    ids: Tuple[int, ...]               # what render receives (numbered logs: the newest-per-slot walk)
    pool: Tuple[int, ...]
    heads: dict = field(default_factory=dict)
    floors: Tuple[int, int, int] = (0, 0, 0)
    n_reach: int = 0
    n_reach_sess: int = 0
    ce_hits: int = 0
    ce_misses: int = 0
    memory_version: Optional[int] = None
    fingerprint: Optional[str] = None
    mode: str = "native"


class Scorer:
    """One reranker score per (query, unit), computed once. Scores come from `preset` first (a published cache),
    then from `store` (a text-keyed cache), then from the reranker; misses are counted."""

    def __init__(self, query, units, reranker=None, preset=None, store=None):
        self.query, self.units, self.reranker, self.store = query, units, reranker, store
        self._s = {}
        self.hits = self.misses = 0
        for i, v in (preset or {}).items():
            self._s[int(i)] = v

    def score(self, items):
        todo = [i for i in items if i not in self._s]
        self.hits += len(items) - len(todo)
        if todo:
            got = {}
            if self.store is not None:
                got = self.store.get_many(self.query, [self.units[i] for i in todo])
            rest = [i for i in todo if self.units[i] not in got]
            if rest:
                if self.reranker is None:
                    raise StoreRefused("%d reranker scores are missing and no reranker was given" % len(rest))
                vals = self.reranker.score(self.query, [self.units[i] for i in rest])
                for i, v in zip(rest, vals):
                    got[self.units[i]] = float(v)
                if self.store is not None:
                    self.store.put_many(self.query, {self.units[i]: got[self.units[i]] for i in rest})
            self.misses += len(rest)
            for i in todo:
                self._s[i] = got[self.units[i]]
        return {i: self._s[i] for i in items}

    def rank(self, items):
        s = self.score(items)
        return sorted(items, key=lambda i: -s[i])

    def of(self, i):
        return self._s[i]


def _canon(xs):
    try:
        return sorted(xs)
    except TypeError:
        return sorted(xs, key=repr)


def reach_ranked(qents, node_ents, ent_nodes, hubs, idf, H=H, decay=DECAY, cap=None, tie=None, seeds=None, deg=None,
                 stop_rounds_at=None, canonical=False):
    """Entity-graph reach to depth H: hop first, then accumulated IDF, then `tie` (core2.py:103-198)."""
    hop, w, frontier = {}, collections.defaultdict(float), set()
    if seeds is not None:
        for n, sw in (sorted(seeds.items(), key=lambda kv: _canon([kv[0]])[0]) if canonical else seeds.items()):
            hop[n] = 1
            w[n] += sw
            frontier.add(n)
    else:
        for e in (_canon(qents) if canonical else qents):
            for n in (_canon(ent_nodes.get(e, ())) if canonical else ent_nodes.get(e, ())):
                hop.setdefault(n, 1)
                w[n] += idf.get(e, 0.0)
                frontier.add(n)
    for h in range(2, H + 1):
        if not frontier:
            break
        nxt = set()
        for i_ in (_canon(frontier) if canonical else frontier):
            es = node_ents[i_] - hubs
            if deg is not None and len(es) > deg:
                es = (sorted(es, key=lambda e: (-idf.get(e, 0.0), e)) if canonical
                      else sorted(es, key=lambda e: -idf.get(e, 0.0)))[:deg]
            elif canonical:
                es = sorted(es)
            for e in es:
                for n in (_canon(ent_nodes.get(e, ())) if canonical else ent_nodes.get(e, ())):
                    if n in hop:
                        continue
                    w[n] += idf.get(e, 0.0) * (decay ** (h - 1))
                    nxt.add(n)
        for n in nxt:
            hop[n] = h
        frontier = nxt
        if stop_rounds_at is not None and len(hop) >= stop_rounds_at:
            break
    _t = tie if tie is not None else (lambda n: n)
    ranked = sorted(hop, key=lambda n: (hop[n], -w[n], _t(n)))
    return (ranked if cap is None else ranked[:cap]), ranked, hop, w


def floor_fill(branches, floors, score_of, U=10):
    """Greedy score-descending selection under per-channel reserved floors (core2.py:201-262)."""
    member = {}
    for b, lst in branches.items():
        for i in lst:
            member.setdefault(i, set()).add(b)
    cand = sorted(member, key=lambda i: (-score_of(i), i))
    sel = list(cand[:U])
    have = {b: len(set(lst)) for b, lst in branches.items()}

    def cnt(b, chosen):
        return sum(1 for i in chosen if b in member[i])

    for b in sorted(floors, key=lambda x: (-floors[x], x)):
        want = min(floors[b], have.get(b, 0))
        guard = 0
        while cnt(b, sel) < want and guard <= U:
            inn = next((i for i in branches[b] if i not in sel), None)
            if inn is None:
                break
            out = None
            for j in sorted(sel, key=lambda x: (score_of(x), x)):
                if b in member[j]:
                    continue
                trial = [i for i in sel if i != j] + [inn]
                ok = True
                for b2, f2 in floors.items():
                    if b2 == b:
                        continue
                    need2 = min(f2, have.get(b2, 0))
                    if cnt(b2, trial) < min(need2, cnt(b2, sel)):
                        ok = False
                        break
                if ok:
                    out = j
                    break
            if out is None:
                break
            sel = [i for i in sel if i != out] + [inn]
            guard += 1
    return sorted(sel, key=lambda i: (-score_of(i), i))


def floors_lt10(U, fd, fs, fg):
    """Channel floors below ten units (lme_pf29.py:1144-1160): two dense anchors, one slot for each other channel,
    the rest by the (fd, fs, fg) ratio with largest remainders."""
    w = (fd, fs, fg)
    f = [0, 0, 0]
    if fd > 0:
        f[0] = min(2, U)
    for i in (2, 1):
        if w[i] > 0 and sum(f) < U:
            f[i] = 1
    R = U - sum(f)
    if R > 0 and sum(w) > 0:
        q = [R * x / float(sum(w)) for x in w]
        a = [int(x) for x in q]
        left = R - sum(a)
        for i in sorted((0, 2, 1), key=lambda i: -(q[i] - a[i]))[:left]:
            a[i] += 1
        f = [f[i] + a[i] for i in range(3)]
    return tuple(f)


def floors_for(U, rule_below_10=True):
    """(dense, keyword, graph) floors for a read of U units: 4/3/3 per ten, scaled by U // 10."""
    if rule_below_10 and U < 10:
        return floors_lt10(U, 4, 3, 3)
    q = max(1, U // 10)
    return (4 * q, 3 * q, 3 * q)


def _slot_reach(idx, qents, hops, cap):
    """Structural stores: reach over the slot graph, from the slots of units holding a query entity
    (lme_pf29.py:2501-2524). Sorted on (hop, unit) before anyone cuts it."""
    seeds = {idx.parent[j] for e in qents if e in idx.ef for j in idx.ef[e] if idx.parent[j] is not None}
    hopd = {s: 0 for s in seeds}
    front = set(seeds)
    for h in range(1, hops + 1):
        if len(hopd) >= cap or not front:
            break
        nxt = set()
        for s3 in front:
            for v3 in idx.adj_slot.get(s3, ()):
                if v3 not in hopd:
                    nxt.add(v3)
        for v3 in nxt:
            hopd[v3] = h
        front = nxt
    return [u for _, u in sorted((h3, idx.newest_row[s3]) for s3, h3 in hopd.items() if s3 in idx.newest_row)]


def value_walk(idx, query, scorer, U):
    """Native numbered logs: the lines a question's chain can run through. From the slots whose subject the question
    names, hop by hop along the subjects each line names, H lines deep; the reranker orders each hop, and the read takes
    the best line of every hop, then the second best of every hop, and so on. At most SLOT_CE_CUT lines are scored.
    Returns those lines and the hop of every line the walk reached."""
    links, row = idx.links, idx.newest_row
    hops, seen, left, hop_of = [], set(), SLOT_CE_CUT, {}
    keys = links.subjects_in(plain_words(query))
    for h in range(H):
        rows = []
        for key in keys:
            if key not in seen:
                seen.add(key)
                rows.extend(row[s] for s in links.slots_of[key])
        if not rows or left <= 0:
            break
        rows = sorted(rows)[:left]
        left -= len(rows)
        for i in rows:
            hop_of.setdefault(i, h)
        hops.append(scorer.rank(rows))
        keys = [key for i in rows for key in links.named[idx.parent[i]]]
    deepest = max((len(hop) for hop in hops), default=0)
    return [hop[r] for r in range(min(U, deepest)) for hop in hops if r < len(hop)], hop_of


def select(idx, vecs, qv, query, scorer, *, U, policy, exact, floors_rule_below_10=True, walk=True):
    """The selection one query receives: (ids, pool, heads, floors, n_reach, n_reach_sess). With `walk` False a numbered
    log keeps the paper's reading, no value walk and lines in serial order; Memory passes it for a log under its own
    serial spelling."""
    import numpy as np
    N = idx.N
    sess_id, r_ents, ef, hubs, idf = idx.parent, idx.r_ents, idx.ef, idx.hubs, idx.idf
    canonical = not exact
    sims = vecs @ qv
    order = np.argsort(-sims) if exact else np.argsort(-sims, kind="stable")
    dense = [int(i) for i in order[:K1]]
    ce_rank = scorer.rank(dense)
    qe = entities(query)
    _off_u, _all_u, _, _ = reach_ranked(qe, r_ents, ef, hubs, idf, H, DECAY, cap=None, deg=RDEG, stop_rounds_at=U,
                                        canonical=canonical)
    gph = [int(i) for i in _off_u]
    if gph:
        scorer.score(gph)
    gset = set(gph)
    if idx.structural:
        gph_s = _slot_reach(idx, qe, H, U)
        gseen_s = set(gph_s)
        if gph_s:
            scorer.score(gph_s[:SLOT_CE_CUT])
        grank_s = scorer.rank(gph_s[:SLOT_CE_CUT])
        drest_s = [i for i in ce_rank if i not in gseen_s]
    else:
        # one node per parent, represented by its best candidate, then advanced to the newest unit of its slot
        cand = set(ce_rank) | gset
        rep = {}
        for i in range(N):
            k = sess_id[i]
            if k not in rep:
                rep[k] = i
            elif i in cand and (rep[k] not in cand or scorer.of(i) > scorer.of(rep[k])):
                rep[k] = i
        for k, i in list(rep.items()):
            s_ = idx.r_slot[i]
            if s_:
                rep[k] = idx.newest_of_slot.get(s_, i)
        ent_s = collections.defaultdict(set)
        for k, i in rep.items():
            for e in ((sorted(r_ents[i] - hubs)) if canonical else (r_ents[i] - hubs)):
                ent_s[e].add(k)
        cut = max(3, int(0.05 * max(len(rep), 1)))
        ent_s = {e: v for e, v in ent_s.items() if len(v) <= cut}
        doc_ents = {k: (r_ents[i] - hubs) for k, i in rep.items()}
        seed_w = collections.defaultdict(float)
        for e in (_canon(qe) if canonical else qe):
            for j in (_canon(ef.get(e, ())) if canonical else ef.get(e, ())):
                k = sess_id[j]
                if k in rep:
                    seed_w[k] += idf.get(e, 0.0)
        _off_s, _all_s, _, _ = reach_ranked([], doc_ents, ent_s, hubs, idf, H, DECAY, cap=RCAP,
                                            tie=lambda k: rep.get(k, 0), seeds=seed_w, deg=RDEG, canonical=canonical)
        gph_s = [rep[k] for k in _off_s if k in rep]
        if gph_s:
            scorer.score(gph_s)
        grank_s = scorer.rank(gph_s)
        drest_s = [i for i in ce_rank if i not in set(gph_s)]
    bs = idx.bm25.get_scores(btok(query))
    sparse = sorted(range(N), key=lambda i: -bs[i])[:K1]
    scorer.score(sparse)
    fq = floors_for(U, floors_rule_below_10)
    fb = {"d": list(ce_rank)[:fq[0]], "s": list(sparse)[:fq[1]], "g": list(grank_s)[:fq[2]], "x": list(ce_rank)}
    quota = {"d": fq[0], "s": fq[1], "g": fq[2], "x": 0}
    flr = floor_fill(fb, quota, scorer.of, U)
    heads = {"flr": flr}
    if policy == "flrs":
        cr0 = set(ce_rank)
        exp = []
        for i0 in flr:
            for j0 in idx.children_of.get(sess_id[i0], ()):
                if abs(idx.pos_of.get(j0, 0) - idx.pos_of.get(i0, 0)) <= 1 and j0 not in exp:
                    exp.append(j0)
        newu = [j for j in exp if j not in cr0]
        scorer.score(newu)
        newu = sorted(set(newu), key=lambda j: (-scorer.of(j), j))
        fb2 = dict(fb, x=list(ce_rank) + newu)
        heads["flrs"] = floor_fill(fb2, quota, scorer.of, U)
    if exact:
        # the transplant asserts of lme_pf29.py:3360-3370, on the heads this mode builds
        lt10 = floors_rule_below_10 and U < 10
        static = fq if lt10 else (4, 3, 3)
        for name, sel in heads.items():
            pp = set(sel)
            for bl, bf in ((fb["d"], static[0]), (fb["s"], static[1]), (fb["g"], static[2])):
                if len(pp & set(bl)) < min(len(set(bl)), bf):
                    raise StoreRefused("floor violated on %s (lme_pf29.py:3369)" % name)
            if not set(ce_rank[:2]) <= pp:
                raise StoreRefused("anchors lost on %s (lme_pf29.py:3370)" % name)
    walked, hop_of = [], {}
    if policy == "attprio_sess":
        if canonical and walk and idx.structural:
            walked, hop_of = value_walk(idx, query, scorer, U)
        head = list(ce_rank[:M_ANCH]) + walked + list(grank_s) + list(drest_s)
        if canonical:
            heads["walk"] = [int(i) for i in walked]
    else:
        head = heads[policy]
    pool = list(dict.fromkeys(head))[:U]
    ids = pool
    if idx.structural:
        # numbered logs: each unit is replaced by the newest record of its slot, deduplicated on the substitute,
        # refilled from the dense ranking to U distinct slots (lme_pf29.py:3566-3584)
        seen, walk = set(), []
        for i in list(pool) + list(pool) + list(pool) + list(ce_rank):
            k = idx.newest_row.get(sess_id[i], i)
            if k in seen:
                continue
            seen.add(k)
            walk.append(k)
            if len(walk) >= U:
                break
        ids = walk
        if canonical and hop_of:
            # the reading order: what the walk reached, in the order it reached it; render keeps the rest by serial
            at = {i: n for n, i in enumerate(walked)}
            heads["first"] = sorted((int(i) for i in ids if i in hop_of), key=lambda i: (hop_of[i], at.get(i, len(at)), idx.order[i]))
    heads.update(ce100=[int(i) for i in ce_rank[:100]], sparse30=[int(i) for i in sparse[:30]],
                 grank30=[int(i) for i in grank_s[:30]], ce_rank=[int(i) for i in ce_rank])
    return ids, pool, heads, fq, len(gph), len(gph_s)
