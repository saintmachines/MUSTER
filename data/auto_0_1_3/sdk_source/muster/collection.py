"""Independent memories sharing a source-derived plan; stores are never joined."""
from dataclasses import dataclass
from typing import Tuple

from .derive import _derive_prepared_collection, _prepare_collection, derive
from .memory import Memory, build
from .plan import Plan
from .records import DEFAULT_TEMPLATE


@dataclass(frozen=True)
class MemoryCollection:
    """Sequence of independent Memory objects; ``plan`` is shared by nonempty members."""

    plan: Plan
    memories: Tuple[Memory, ...]

    def __len__(self):
        return len(self.memories)

    def __iter__(self):
        return iter(self.memories)

    def __getitem__(self, index):
        return self.memories[index]


def build_collection(stores, *, reference_chunk=4096, sample=3, step8=True, models=None, embed=True,
                     template=DEFAULT_TEMPLATE, normalize_newlines=True) -> MemoryCollection:
    """Build independent stores under one native source-derived configuration.

    The first three stores supply statistics by default (the paper's sampling
    rule); sample=None uses all stores. Every member is checked for compatible
    source boundaries. Empty members remain empty memories with U=0 and never
    receive a numbered-log plan. A list of records passed to ordinary ``build``
    continues to mean one store; this explicit collection entry keeps sources
    independent. Supplied record iterators are materialized only once.
    """
    prepared = _prepare_collection(stores, template=template, normalize=normalize_newlines)
    plan = _derive_prepared_collection(
        prepared, reference_chunk=reference_chunk, sample=sample, step8=step8,
        template=template, normalize=normalize_newlines)
    shared_models = models
    if embed and shared_models is None and plan.branch != "empty":
        from .models import Models
        shared_models = Models(encoder_spec=plan.encoder, reranker_spec=plan.reranker)
    memories = []
    for source, _, _ in prepared:
        own = derive(source, reference_chunk=reference_chunk, template=template,
                     normalize_newlines=normalize_newlines)
        member_plan = own if own.branch == "empty" else plan
        memories.append(build(source, member_plan, models=shared_models, embed=embed,
                              template=template, normalize_newlines=normalize_newlines))
    return MemoryCollection(plan, tuple(memories))
