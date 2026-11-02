#!/usr/bin/env python3
"""Retrieval, merge and resolution for every benchmark - corrected.

Six defects in the previous version made its comparisons uninterpretable, and all six came from the same
place: two retrieval branches that were never put on a common footing.

1. The graph branch bypassed the reranker. Dense candidates were ordered by the cross-encoder while graph
   reach was ordered by bi-encoder cosine, and the two lists were concatenated. Ranks were then read off
   positions, so a fact ranked first because it arrived from the graph, not because it scored well - and
   any parent holding a graph fact automatically outranked a parent whose best child was a top
   cross-encoder hit. "Select the top parents by reranked child" was only true of half the pool.
2. Because of (1), the earlier finding that score-merging is worse than reach-priority compared cosine
   against list position rather than one merge policy against another. It has to be re-measured.
3. `cap` meant two different things - the graph's search budget and the number of parents - so a change
   intended for one silently moved the other. They are now `bfs_budget` and `n_parents`.
4. The graph search was run once at the largest budget and sliced. Its budget is a stop condition, so
   `search(30)[:10]` explores further than `search(10)` and returns a different set; each budget now runs
   its own search.
5. Hierarchical configurations mixed graph reach into their children, so they measured the hierarchical baseline plus a graph,
   not the hierarchical baseline. `use_graph` is now an explicit factor and can be turned off.
6. Questions were consumed in file order while the original harness shuffled. FactConsolidation is ordered
   by difficulty - the first 36 questions score 0.312 against 0.227 for the rest - so any partial run was
   biased upward. Callers pass their own order; this module never assumes one.

Every candidate, from whichever branch, is scored by the same cross-encoder before anything is selected.
Merge policy then becomes a real choice rather than an artefact of concatenation order.
"""
from __future__ import annotations

import numpy as np

try:
    import rerankers
except Exception:                      # pragma: no cover
    rerankers = None

KEEP_ALL, PRUNE = "KEEP_ALL", "PRUNE"


class Corpus:
    """Units plus the hierarchy over them. Adapters build this; the pipeline never touches raw data."""

    def __init__(self, texts, vecs, parent_of=None, order_of=None, pos_of=None):
        n = len(texts)
        self.texts = list(texts)
        self.vecs = np.asarray(vecs, dtype=np.float32)
        self.parent_of = list(parent_of) if parent_of is not None else [None] * n
        self.order_of = list(order_of) if order_of is not None else list(range(n))
        if pos_of is not None:
            self.pos_of = list(pos_of)
        else:
            seen, pos = {}, [0] * n
            for i in sorted(range(n), key=lambda j: self.order_of[j]):
                p = self.parent_of[i]
                pos[i] = seen.get(p, 0)
                seen[p] = pos[i] + 1
            self.pos_of = pos
        self.children_of = {}
        for i, p in enumerate(self.parent_of):
            if p is not None:
                self.children_of.setdefault(p, []).append(i)

    def __len__(self):
        return len(self.texts)


def build_reranker(name="BAAI/bge-reranker-v2-m3", device="mps"):
    if rerankers is None:
        return None
    return rerankers.build(name, device=device)


class Scorer:
    """One cross-encoder score per (question, unit), cached, shared by every configuration.

    Both retrieval branches must be scored here before selection. Anything selected on a score this class
    did not produce is comparing across scales.
    """

    def __init__(self, ce, corpus, query, qvec=None):
        self.ce, self.corpus, self.query, self.qvec = ce, corpus, query, qvec
        self._s = {}

    def score(self, items):
        todo = [i for i in items if i not in self._s]
        if todo:
            if self.ce is not None:
                for i, v in zip(todo, self.ce.score(self.query, [self.corpus.texts[i] for i in todo])):
                    self._s[i] = float(v)
            else:                       # no reranker: fall back to bi-encoder, and say so upstream
                for i in todo:
                    self._s[i] = float(self.corpus.vecs[i] @ self.qvec)
        return {i: self._s[i] for i in items}

    def rank(self, items):
        s = self.score(items)
        return sorted(items, key=lambda i: -s[i])

    def of(self, i):
        return self._s[i]


def reach_ranked(qents, node_ents, ent_nodes, hubs, idf, H=4, decay=0.5, cap=None, tie=None,
                 seeds=None, deg=None, stop_rounds_at=None):
    """The canonical graph reach: traverse to depth H, rank, and cut inside this contract.

    Returns (offered, reached, hop, w).
      offered  the ranking cut at `cap` - what the branch proposes AND what it has scored
      reached  every node the traversal touched - for "have I already seen this" predicates

    The two are different sets and conflating them was a live defect. `gset` was used both as the
    branch's membership (so `drest` = dense minus gset fills the leftover slots) and as an exclusion
    set for the sparse tail (`srest` = sparse minus gset minus dense). Setting it to the whole reach
    silently deleted dense candidates that fell past the cut; setting it to the cut alone re-admits,
    through the sparse door, exactly the weak graph candidates the ranking just demoted. Returning
    both lets each call site say which one it means.

    Three separable decisions, each settled here rather than per call site:

    1. A budget bounds SELECTION, never TRAVERSAL. Both predecessors broke out of the expansion on a
       node count, which makes the retained subset a function of set-iteration order - and that order
       is `hash(i) mod table_capacity`, so it moves with corpus size, silently corrupting any
       corpus-size scaling experiment. Depth is handled by the weight instead: a node first reached
       at hop h carries decay**(h-1), 0.125 at h=4, and sinks on its own.
    2. The result is ORDERED - hop first, then accumulated IDF within a hop - so a cut is a cut
       on a ranking, never on enumeration order.
    3. The cut lives HERE, so no call site can hand an unbounded list to the cross-encoder. Scoring is
       the expensive step; traversal is set arithmetic.

    `tie` breaks equal weights. It defaults to the node's own id, but the document-level call passes
    the representative unit's index, because that index is the corpus's own recency order and it was
    the only temporal signal in the document list that the previous implementation carried. Losing it
    on a temporal benchmark is not a neutral change.

    `node_ents` and `ent_nodes` are the two directions of one index over WHATEVER the node is - units
    on one call, documents on the other - so this is level-agnostic and dataset-agnostic.
    """
    import collections as _c
    hop, w, frontier = {}, _c.defaultdict(float), set()
    if seeds is not None:
        # Caller-supplied seeding, for the case where the SEED index and the EXPANSION index must
        # differ. The document level needs exactly that: seeding must be broad - any unit holding a
        # query entity seeds its document - while expansion stays on the narrow document index
        # (representative-based, rarity-filtered). Seeding through the expansion index collapsed the
        # LongMemEval document reach from a median of 10 to 1 and silently degraded every
        # document-level arm to near-dense; the bench telemetry caught it, this parameter fixes it.
        for n, sw in seeds.items():
            hop[n] = 1
            w[n] += sw
            frontier.add(n)
    else:
        for e in qents:
            for n in ent_nodes.get(e, ()):
                hop.setdefault(n, 1)
                w[n] += idf.get(e, 0.0)
                frontier.add(n)
    for h in range(2, H + 1):
        if not frontier:
            break
        nxt = set()
        for i_ in frontier:
            es = node_ents[i_] - hubs
            if deg is not None and len(es) > deg:
                # Expansion degree cap: only this node's `deg` highest-IDF entities expand. The device
                # that actually binds where units are long - RULER chunks carry 26-40 entities against
                # 1.6-1.8 on FactConsolidation and LongMemEval, so one expansion step from seven seeds
                # touched 80-86% of that corpus. Measured at deg=5: RULER reach shrinks 17-44x with
                # gold presence at 10 unchanged, and the cap is an exact identity on the short-unit
                # corpora - the structure decides where it binds, not the dataset name. Seeding is
                # untouched, so a query still reaches everything it names directly.
                es = sorted(es, key=lambda e: -idf.get(e, 0.0))[:deg]
            for e in es:
                for n in ent_nodes.get(e, ()):
                    if n in hop:
                        continue
                    w[n] += idf.get(e, 0.0) * (decay ** (h - 1))
                    nxt.add(n)
        for n in nxt:
            hop[n] = h
        frontier = nxt
        if stop_rounds_at is not None and len(hop) >= stop_rounds_at:
            # Round-boundary count stop - deterministic, unlike the mid-enumeration truncations this
            # function replaced. The whole frontier expands before the check, so the collected set
            # never depends on iteration order. The correct invariant is therefore NOT "a budget must
            # never stop traversal" but "no truncation mid-enumeration; round-boundary stops are
            # fine". FactConsolidation's legacy unit reach stopped exactly this way, and reproducing
            # it is what keeps the unit-level contrast arms at their historical behaviour; the no-stop
            # mode stays for callers whose loss point is the post-ranking cut (LME/ruler).
            break
    _t = tie if tie is not None else (lambda n: n)
    # Hop first, weight within a hop. Measured on the ordering itself (RULER qa1, gold presence at 10):
    # set-iteration order 0.020, weight-only 0.120, hop-only 0.480, hop-then-weight 0.520. Weight-only
    # shipped briefly and was a measured regression (prio_doc 0.540 -> 0.450; qa2 graph recall
    # 0.196 -> 0.005): the weight accumulates once per linking frontier node, so it carries a
    # popularity term that lets a densely linked hop-2 node outrank the hop-1 node holding the one
    # rare query entity. Proximity is the primary key; weight only breaks ties inside a hop.
    ranked = sorted(hop, key=lambda n: (hop[n], -w[n], _t(n)))
    return (ranked if cap is None else ranked[:cap]), ranked, hop, w


def floor_fill(branches, floors, score_of, U=10):
    """Greedy score-descending selection under per-branch reserved floors.

    branches: {name: ordered, already-scored candidate list}. floors: {name: int}, sum <= U.
    A unit found by several branches counts toward every one of them, so a floor can be satisfied by
    overlap without spending a slot - a branch whose candidates already sit in the score top-U costs
    nothing. While a floor is short, the branch's next candidate IN ITS OWN RANKING - BM25 order
    for a lexical branch, hop order for a graph branch - replaces the worst selected unit
    OUTSIDE the branch. Filling by the shared score instead was measured to defeat the operator
    (RULER n=200: shared-score admissions carried gold 19.5% vs the branch's own top-3 54.0%):
    the floor exists precisely because the shared scorer is blind to this branch's signal, so
    the shared score keeps only the base selection and the eviction choice whose removal does not push any other floor below the count it
    has already achieved (capped at its target) - evicting a same-branch unit would cancel the gain,
    and demanding that not-yet-processed floors already be satisfied would deadlock. The dense loss is
    bounded by the sum of the other floors BY CONSTRUCTION, which is the same worst-case argument the
    graph's 2-of-10 reservation already carries; this operator is that reservation generalised to any
    number of branches. Measured on RULER before this existed: a reserved sparse quota reached
    0.850-0.858 pooled gold presence where RRF reached 0.784 and the plain score union 0.659 - the
    quota beats fusion, so the floors are the mechanism, not a tuning surface.

    Deterministic: all orderings break ties on the unit id. Floors larger than a branch's candidate
    list are clamped to it. Returns the selection in descending score order.
    """
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


def slices_of(sel, sess_of, pos_of, unit_at, gap=2, pad=1):
    """Adaptive level decision for a selected unit set: within one parent, hits whose positions are
    within `gap` of each other form one group; groups of >=2 render as a single padded contiguous
    span [min-pad, max+pad]; singleton hits stay single units (no padding). Deterministic: parents
    and positions sorted. Returns {unit: (span unit ids...)} for grouped units only."""
    by_par = {}
    for i in sel:
        by_par.setdefault(sess_of[i], []).append(i)
    out = {}
    for par in sorted(by_par, key=lambda x: str(x)):
        hits = sorted(by_par[par], key=lambda i: (pos_of.get(i, 0), i))
        groups, group = [], [hits[0]]
        for i in hits[1:]:
            if pos_of.get(i, 0) - pos_of.get(group[-1], 0) <= gap:
                group.append(i)
            else:
                groups.append(group)
                group = [i]
        groups.append(group)
        for g in groups:
            if len(g) >= 2:
                lo = pos_of.get(g[0], 0) - pad
                hi = pos_of.get(g[-1], 0) + pad
                span = tuple(unit_at[(par, q)] for q in range(lo, hi + 1) if (par, q) in unit_at)
                for i in g:
                    out[i] = span
    return out


def dense_candidates(corpus, qvec, k1):
    return [int(i) for i in np.argsort(-(corpus.vecs @ qvec))[:k1]]


def fuse(dense, graph, scorer, policy="score", rrf_k=60, budget=10):
    """Combine the two branches into one ranking. Both are already cross-encoder scored.

    score     one ranking by cross-encoder score; the branches compete on equal terms
    priority  graph first, then dense - the old behaviour, kept so the difference is measurable
    rrf       reciprocal rank fusion of the two branch rankings
    rrf_n     RRF with rank rescaled by list length, so a branch cannot win by returning more candidates
    rrf_nc    rrf_n weighted by each view's own top-1-minus-median score gap, a label-free confidence
    dense     dense only (graph ablated away)
    graph     graph only
    """
    d = list(dict.fromkeys(dense))
    g = list(dict.fromkeys(graph))
    if policy == "dense":
        return scorer.rank(d)
    if policy == "graph":
        return scorer.rank(g)
    both = list(dict.fromkeys(d + g))
    scorer.score(both)
    if policy == "score":
        return scorer.rank(both)
    if policy == "priority":
        gr = scorer.rank(g); dr = scorer.rank([i for i in d if i not in set(g)])
        return gr + dr
    if policy in ("rrf", "rrf_n", "rrf_nc"):
        lists = [scorer.rank(d), scorer.rank(g)]
        big = 10 ** 6
        if policy == "rrf":
            ranks = [{i: r for r, i in enumerate(L)} for L in lists]
            w = [1.0, 1.0]
        else:
            # Length normalisation. Plain RRF gives an item in a 100-long list a worse reciprocal than the
            # same item in a 13-long list purely because the list is long, so the branch that happens to
            # return more candidates wins on count. Rescaling rank by list length removes that: what
            # matters is where an item sits WITHIN its own branch. This is the defect behind fair fusion
            # keeping only 4% graph items while branch priority keeps 40.5%.
            B = float(budget or 10)
            ranks = [{i: B * r / max(len(L), 1) for r, i in enumerate(L)} for L in lists]
            if policy == "rrf_n":
                w = [1.0, 1.0]
            else:
                # Per-query view confidence, label-free: a view whose top item stands well clear of its own
                # median is answering this question; a flat view is guessing. No fitted parameter - the
                # weights are a softmax of each view's own score gap.
                gaps = []
                for L in lists:
                    if len(L) < 2:
                        gaps.append(0.0); continue
                    sc_ = [scorer.of(i) for i in L]
                    med = sorted(sc_)[len(sc_) // 2]
                    sd = (np.std(sc_) or 1.0)
                    gaps.append((max(sc_) - med) / sd)
                m = max(gaps)
                ex = [np.exp(x - m) for x in gaps]
                tot = sum(ex) or 1.0
                w = [len(lists) * e / tot for e in ex]
        return sorted(both, key=lambda i: -sum(
            wv / (rrf_k + rk.get(i, big)) for wv, rk in zip(w, ranks)))
    raise ValueError("unknown fuse policy %r" % policy)


def merge(ranked, corpus, n_units, hierarchical=True, n_parents=None, alpha=2, hops=None, scorer=None):
    """Select the units the reader sees. `ranked` must already be one consistent ranking.

    Flat: the top `n_units`. Hierarchical: take `n_parents * alpha` children, group them by parent, keep
    the `n_parents` best parents by their best child's rank, expand inside each parent, then cut back to
    `n_units` so the budget is identical either way. Without that final cut a hierarchy is compared while
    feeding several times the context, which measures budget rather than selection.
    """
    info = {"hierarchical": hierarchical, "alpha": alpha, "hops": hops, "n_parents_req": n_parents}
    if not hierarchical:
        pool = list(ranked)[:n_units]
        info["n_parents"] = len({corpus.parent_of[i] for i in pool})
        info["n_units"] = len(pool)
        return pool, info
    npar = n_parents or n_units
    children = list(ranked)[:max(npar * alpha, npar)]
    rank_of = {i: r for r, i in enumerate(children)}
    par = {}
    for i in children:
        key = corpus.parent_of[i]
        par.setdefault(("__u__", i) if key is None else key, []).append(i)
    best = sorted(par, key=lambda k: min(rank_of[i] for i in par[k]))[:npar]
    pool = []
    for k in best:
        if isinstance(k, tuple):
            pool += par[k]
            continue
        sibs = corpus.children_of.get(k, par[k])
        if hops is None:
            pool += sibs
        else:
            keep = set()
            for c in par[k]:
                p = corpus.pos_of[c]
                keep |= set(range(p - hops, p + hops + 1))
            pool += [i for i in sibs if corpus.pos_of[i] in keep]
    pool = list(dict.fromkeys(pool))
    info["n_pool_before_cut"] = len(pool)
    if len(pool) > n_units:
        pool = (scorer.rank(pool) if scorer is not None else pool)[:n_units]
    info["n_parents"] = len(best)
    info["n_units"] = len(pool)
    return pool, info


def resolve_then_truncate(ranked, corpus, n_units, key_of=None, scorer=None):
    """Drop superseded versions BEFORE cutting to the budget, then backfill to the full budget.

    Cutting first and resolving afterwards spends slots on facts that are about to be deleted. Measured on
    FactConsolidation-MH: of ten retrieved units only 8.03 distinct slots survive, so the reader receives
    eight units against a benchmark whose retrieve_num is ten - a 20% budget handicap that no competitor
    pays. Resolving first and backfilling from further down the ranking fills all ten with facts the
    resolver has already endorsed.
    """
    kf = key_of or (lambda i: corpus.parent_of[i])
    newest = {}
    for i in ranked:
        k = kf(i)
        if k is not None and (k not in newest or corpus.order_of[i] > corpus.order_of[newest[k]]):
            newest[k] = i
    keep = set(newest.values())
    survivors = [i for i in ranked if kf(i) is None or i in keep]
    pool = survivors[:n_units]
    if len(pool) < n_units:             # nothing left to promote: fall back to superseded items in order
        rest = [i for i in ranked if i not in set(pool)]
        pool = (pool + rest)[:n_units]
    return pool, {"n_units": len(pool), "n_removed": len(ranked) - len(survivors),
                  "n_slots": len({kf(i) for i in pool if kf(i) is not None})}


def newest_per_parent(pool, corpus, key_of=None):
    """Index of the highest-order unit per resolution key. The key is the SLOT, which is not necessarily
    the retrieval parent - changing what we retrieve by must not silently change what we resolve by."""
    kf = key_of or (lambda i: corpus.parent_of[i])
    newest = {}
    for i in pool:
        k = kf(i)
        if k is not None and (k not in newest or corpus.order_of[i] > corpus.order_of[newest[k]]):
            newest[k] = i
    return newest
