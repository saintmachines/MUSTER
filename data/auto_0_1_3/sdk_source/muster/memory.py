"""Memory: one version of a store, cut by its plan, indexed, and embedded once. Immutable.

    mem = muster.build(records)            # plan from the text, cut, index, one embedding pass
    ctx = mem.context("Where did Caroline move?")
    print(ctx.text)

A preset name builds the paper's exact plan instead: muster.build(text, "beam.u20").
"""
import hashlib
import operator
from dataclasses import dataclass, field, replace
from typing import Optional

from . import presets
from .cut import Cut, cut
from .derive import derive, store_text
from .errors import EnvironmentRefused, PlanError, QueryError, StaleSelection
from .index import Index
from .plan import Plan
from .records import DEFAULT_TEMPLATE
from .render import Block, Context, flat, numbered, r4
from .search import Scorer, Selection, select


def _check_query(query):
    if not isinstance(query, str):
        raise QueryError("a query is a string, got %s" % type(query).__name__)
    if not query.strip():
        raise QueryError("empty query")
    try:
        query.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise QueryError("the query is not valid UTF-8 text: position %d holds half of a surrogate pair" % exc.start) from exc


def _shown(x):
    """repr for an error message; an int too long for Python to print is named, not printed."""
    try:
        s = repr(x)
    except ValueError:
        return "an integer too long to print"
    return s if len(s) <= 60 else s[:57] + "..."


def _read_budget(U, default):
    """The units one read takes: the plan's own for None or 0, else a positive whole number (an int or a numpy integer).
    True, 1.9, "2" and inf are refused, not read as 1, 1, 2 or an overflow."""
    n = 0 if U is None else U
    try:
        if isinstance(n, bool):
            raise TypeError("a bool is not a count")
        n = operator.index(n)
    except TypeError as exc:
        raise PlanError("U must be a positive whole number, got %s" % _shown(U)) from exc
    if n == 0:
        n = int(default)
    if n < 1:
        raise PlanError("U must be a positive whole number, got %s" % _shown(U))
    return n


def _fingerprint(units):
    from .compat.ce_cache import ce_sig
    return ce_sig(list(units))


@dataclass
class Memory:
    plan: Plan
    text: str
    cut: Cut
    records: Optional[list] = None
    record_spans: Optional[list] = None
    version: int = 0
    vecs: object = None
    models: object = None
    _index: Optional[Index] = field(default=None, repr=False)
    ordered_records: object = field(default=None, repr=False)
    _semantic_keys: Optional[tuple] = field(default=None, repr=False)
    _record_unit_indices: Optional[tuple] = field(default=None, repr=False)

    @property
    def units(self):
        return self.cut.units

    @property
    def parent(self):
        return self.cut.parent

    @property
    def index(self) -> Index:
        if self._index is None:
            keys = self.cut.keys if self._semantic_keys is None else self._semantic_keys
            ordered = self.plan.branch == "numbered" and not self.cut.diagnostics.get("unmarked_records")
            self._index = Index(self.cut.units, keys, self.cut.parent, self.cut.order, has_order=ordered)
        return self._index

    @property
    def fingerprint(self):
        return _fingerprint(self.cut.units)

    def unit_records(self, i):
        """The records unit `i` was cut from (empty for a memory built from a plain string)."""
        if self.records is None or self.record_spans is None:
            return []
        if self._record_unit_indices is not None:
            return [self.records[self._record_unit_indices[i]]]  # includes zero-length record spans
        if not hasattr(self, "_ur"):
            ur, j = [], 0
            spans = self.record_spans
            for sp in self.cut.spans:
                if sp is None:
                    ur.append(())
                    continue
                s, e = sp
                while j < len(spans) and spans[j][1] <= s:
                    j += 1
                k, got = j, []
                while k < len(spans) and spans[k][0] < e:
                    got.append(k)
                    k += 1
                ur.append(tuple(got))
            self._ur = ur
        return [self.records[k] for k in self._ur[i]]

    def __len__(self):
        return len(self.cut.units)

    def _models(self, models):
        m = models or self.models
        if m is None:
            from .models import Models
            m = self.models = Models(encoder_spec=self.plan.encoder, reranker_spec=self.plan.reranker)
        return m

    def embed(self, models=None):
        """Embed every key in one batched call (the units, or the user turns of a rounds store)."""
        import numpy as np
        m = self._models(models)
        keys = list(self.cut.keys)
        self.vecs = m.encoder.encode(keys, batch_size=self.plan.encoder.batch_size, normalize_embeddings=True) \
            if keys else np.zeros((0, 1024), dtype=np.float32)
        return self

    def search(self, query: str, *, U: Optional[int] = None, models=None, cache=None, scores=None) -> Selection:
        """Select the units a reader gets for `query`. `cache` is a published score file's contents for this
        store ({question key: {unit index: score}}); `scores` is a text-keyed ScoreCache."""
        import numpy as np
        _check_query(query)
        if len(self) == 0:
            return Selection(query, 0, self.plan.policy, (), (), memory_version=self.version, mode=self.plan.mode)
        m = self._models(models)
        if self.vecs is None:
            self.embed(m)
        U = _read_budget(U, self.plan.U)
        exact = self.plan.mode == "exact"
        if exact:
            from .compat.env import check_environment
            check_environment()
            _check_exact_models(m, self.plan)
        qv = np.asarray(m.encoder.encode([query], normalize_embeddings=True), dtype=np.float32)[0]
        pre = None
        if cache is not None:
            from .compat.ce_cache import qkey
            pre = cache.get(qkey(query), {})
        sc = Scorer(query, list(self.cut.units), reranker=_LazyReranker(m), preset=pre, store=scores)
        rule = (not exact) or (self.plan.compat or {}).get("floors_lt10") == "rule"
        ids, pool, heads, fq, nr, nrs = select(self.index, self.vecs, qv, query, sc, U=U, policy=self.plan.policy,
                                               exact=exact, floors_rule_below_10=rule, walk=self.plan.serial != "lead")
        return Selection(query, U, self.plan.policy, tuple(ids), tuple(pool), heads, tuple(fq), nr, nrs,
                         sc.hits, sc.misses, self.version, self.fingerprint, self.plan.mode)

    def render(self, selection, *, style: Optional[str] = None) -> Context:
        if hasattr(selection, "memory_version"):
            if selection.memory_version != self.version or (selection.fingerprint and selection.fingerprint != self.fingerprint):
                raise StaleSelection("selection was made on another version of this memory")
            ids = list(selection.ids)
        else:
            ids = list(selection)
        style = style or self.plan.render
        chain = (self.plan.compat or {}).get("chain", "") if self.plan.mode == "exact" else ""
        if style == "r4" and chain.split(" ")[0] in ("seg3", "bgroup") and hasattr(selection, "heads"):
            # exact mode: the published two-step chain, intermediate render -> recovered ids -> one block per parent
            from .compat.intermediates import rank_pos, render_bgroup, render_seg3
            from .compat.recover import recover_ids
            units, idx = list(self.cut.units), self.index
            cap = int((self.plan.compat.get("env") or {}).get("RGROUP_MAX") or selection.U)
            txt = lambda i: units[i]
            if chain.startswith("seg3"):
                inter = render_seg3(ids, txt, idx.parent, idx.r_ents, idx.hubs, cap)
            else:
                pos = rank_pos(selection.heads.get("ce_rank", ()))
                inter = render_bgroup(ids, txt, pos, len(pos) + 1, idx.parent, idx.r_ents, idx.hubs, cap)
            ids = recover_ids(inter, units, idx.parent)
            c = r4(ids, self.cut.units, self.cut.parent)
            c = replace(c, meta=dict(intermediate=inter))
        elif style == "r4":
            c = r4(ids, self.cut.units, self.cut.parent)
        elif style == "flat":
            order = self.cut.order if self.plan.mode == "native" and self.plan.branch == "rounds" else None
            c = flat(ids, self.cut.units, self.cut.parent, order)
        elif style == "numbered":
            order = self.cut.order
            if self.cut.diagnostics.get("unmarked_records"):
                order = self._record_unit_indices or tuple(range(len(self.cut.units)))  # source positions, not clocks
            first = selection.heads.get("first", ()) if hasattr(selection, "heads") and self.plan.mode != "exact" else ()
            c = numbered(ids, self.cut.units, order, self.cut.prefixes, first)
            if self._record_unit_indices is not None:
                blocks = tuple(Block(n + 1, self.cut.parent[i], (i,),
                                     numbered([i], self.cut.units, order, self.cut.prefixes).text)
                               for n, i in enumerate(c.ids))
                c = replace(c, blocks=blocks)  # typed records retain structured source IDs and metadata
        else:
            raise PlanError("unknown render style %r" % style)
        return replace(c, mode=self.plan.mode, memory_version=self.version)

    def context(self, query: str, **kw) -> Context:
        style = kw.pop("style", None)
        return self.render(self.search(query, **kw), style=style)


def _check_exact_models(models, plan):
    """Exact plans run the paper's models at the paper's lengths; refuse a model built at another length. Injected
    models without a spec (fakes, services) are not checked here; exact runs record them instead."""
    enc = getattr(models.encoder, "spec", None)
    rr = getattr(models._rr, "spec", None) if getattr(models, "_rr", None) is not None \
        else getattr(models, "reranker_spec", None)
    for have, want in ((enc, plan.encoder), (rr, plan.reranker)):
        if have is not None and (have.name, have.max_length) != (want.name, want.max_length):
            raise EnvironmentRefused("exact plan %s needs %s at max length %d, got %s at %d"
                                     % (plan.source, want.name, want.max_length, have.name, have.max_length))


class _LazyReranker:
    """Loads the reranker only if a score is actually missing."""

    def __init__(self, models):
        self.models = models

    def score(self, query, docs):
        return self.models.reranker.score(query, docs)


def resolve_plan(store_text_, plan):
    if plan is None:
        return derive(store_text_)
    if isinstance(plan, str):
        return presets.get(plan)
    if isinstance(plan, Plan):
        return plan
    raise PlanError("plan must be None, a preset name or a Plan")


def build(store, plan=None, *, models=None, embed=True, template: str = DEFAULT_TEMPLATE,
          normalize_newlines: bool = True, declared_order=None, record_format: str = "content",
          reference_chunk: int = 4096) -> Memory:
    """Build one store. reference_chunk configures Auto; an explicit plan is used unchanged."""
    exact_preset = isinstance(plan, str) or (isinstance(plan, Plan) and plan.mode == "exact")
    if declared_order is not None:
        if template != DEFAULT_TEMPLATE:
            raise PlanError("declared_order uses producer-rendered bodies; custom template is unsupported")
        if exact_preset:
            raise PlanError("declared_order is a native opt-in input mode and cannot change exact replay")
        from .source_order import build_ordered_memory
        return build_ordered_memory(store, declared_order, record_format=record_format,
                                    plan=plan, models=models, embed=embed)
    if record_format != "content":
        raise PlanError("record_format requires declared_order provenance")
    text, spans, recs = store_text(store, template=template, normalize=(normalize_newlines and not exact_preset))
    p = (derive(recs if recs is not None else text, reference_chunk=reference_chunk,
                template=template, normalize_newlines=normalize_newlines)
         if plan is None else resolve_plan(text, plan))
    semantic, record_indices = None, None
    if recs is not None and p.mode == "native" and p.branch == "numbered":
        from .record_serial import cut_record_serials
        c, semantic, record_indices = cut_record_serials(recs, text, spans, p, normalize=normalize_newlines)
    else:
        c = cut(text, p)
    mem = Memory(plan=p, text=text, cut=c, records=recs, record_spans=spans, models=models,
                 _semantic_keys=semantic, _record_unit_indices=record_indices)
    if embed and len(c):
        mem.embed(models)
    return mem


def context(memory: Memory, query: str, **kw) -> Context:
    return memory.context(query, **kw)


def search(memory: Memory, query: str, **kw) -> Selection:
    return memory.search(query, **kw)


def render(memory: Memory, selection, **kw) -> Context:
    return memory.render(selection, **kw)
