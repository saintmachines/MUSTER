"""Plan: how one store is cut, grouped, searched and rendered. Frozen, JSON round-trip, identified by its hash."""
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Optional, Tuple

BRANCHES = ("numbered", "record", "prose", "rounds", "empty")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    revision: Optional[str]
    max_length: int
    batch_size: int
    dtype: str = "float32"


ENCODER = ModelSpec("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", 2048, 32)
RERANKER = ModelSpec("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", 2048, 16)


@dataclass(frozen=True)
class Plan:
    branch: str                      # numbered | record | prose | rounds | empty
    atom: Optional[str]              # blank | sentence | None (numbered, rounds)
    chunk: Optional[int]             # o200k packing budget per unit
    mscale: int                      # o200k size of a parent; 0 = one parent per unit
    U: int                           # units read per query by default
    policy: str                      # flr | flrs | attprio_sess
    render: str                      # r4 | flat | numbered
    mode: str = "native"             # native | exact
    frozen: bool = False             # presets and explicit plans never re-derive on add
    supersede: str = "off"           # off | numbered | declared
    reference_chunk: int = 4096
    encoder: ModelSpec = ENCODER
    reranker: ModelSpec = RERANKER
    compat: Optional[dict] = None    # exact presets only: the harness settings this plan stands for
    stats: Optional[dict] = None
    source: str = "derived"          # derived | preset:<name> | custom
    trace: Tuple[str, ...] = field(default=())
    serial: str = "dot"              # numbered logs: dot ("12. body") | lead (the log's own style, "Step 12: body")

    def __post_init__(self):
        if self.branch not in BRANCHES:
            raise ValueError("unknown branch %r" % self.branch)
        if self.mode not in ("native", "exact"):
            raise ValueError("mode must be native or exact")
        if self.serial not in ("dot", "lead"):
            raise ValueError("serial must be dot or lead")

    def to_dict(self):
        d = asdict(self)
        d["trace"] = list(self.trace)
        if d["serial"] == "dot":         # plans from before the field keep their JSON and plan_id
            del d["serial"]
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["encoder"] = ModelSpec(**d["encoder"]) if isinstance(d.get("encoder"), dict) else d.get("encoder", ENCODER)
        d["reranker"] = ModelSpec(**d["reranker"]) if isinstance(d.get("reranker"), dict) else d.get("reranker", RERANKER)
        d["trace"] = tuple(d.get("trace") or ())
        return cls(**d)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def plan_id(self):
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def with_(self, **kw):
        return replace(self, **kw)
