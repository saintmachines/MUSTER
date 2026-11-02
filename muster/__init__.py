"""MUSTER: long-term memory for agents that keeps every word, sets its own strategy from the stored text, and makes no
model call to build or grow the memory.

    import muster
    plan = muster.derive(records)        # how this store is cut, grouped and read, decided from its own text
    print(muster.explain(plan))

Heavy dependencies (numpy, torch, nltk, rank_bm25) are imported only when a step needs them.
"""
from ._version import __version__
from .derive import derive, derive_collection, explain
from .errors import (AssetMissing, EnvironmentRefused, MusterError, PlanError, PlanViolation, RecordError,
                     StaleSelection, StoreLocked, StoreRefused)
from .plan import ENCODER, RERANKER, ModelSpec, Plan
from .records import DEFAULT_TEMPLATE, Record, numbered, serialize
from . import presets
from .memory import Memory, build, context, render, search
from .models import Models
from .render import Context, compose
from .store import AddResult, Store
from .cache import ScoreCache

__all__ = ["__version__", "build", "search", "render", "context", "compose", "Memory", "Models", "Context", "Store", "AddResult", "ScoreCache", "derive", "derive_collection", "explain", "Plan", "ModelSpec", "ENCODER", "RERANKER",
           "Record", "DEFAULT_TEMPLATE", "numbered", "serialize", "presets", "MusterError", "PlanError", "RecordError",
           "StoreRefused", "EnvironmentRefused", "AssetMissing", "StaleSelection", "PlanViolation", "StoreLocked"]
