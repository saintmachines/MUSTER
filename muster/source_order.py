"""Opt-in record boundaries and source order, without inferred revision semantics.

The producer declares the source sequence. Prefix formats are explicit codecs,
not detectors; dates, gaps and repeated serials never trigger replacement.
"""
import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

from .errors import RecordError
from .stats import LEAD


@dataclass(frozen=True)
class SourceOrder:
    source_sha256: str
    source_locator: str
    kind: str
    declaration: str
    namespace: str

    def __post_init__(self):
        if not isinstance(self.source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise RecordError("source_sha256 must identify the original source version")
        if self.kind not in ("structured_array", "explicit_append_sequence"):
            raise RecordError("declared source order kind must be structured_array or explicit_append_sequence")
        if any(not isinstance(v, str) or not v.strip()
               for v in (self.source_locator, self.declaration, self.namespace)):
            raise RecordError("source locator, order declaration and namespace are required")
        try:
            for value in (self.source_locator, self.declaration, self.namespace):
                value.encode("utf-8")
        except UnicodeError as exc:
            raise RecordError("source provenance must be valid UTF-8 text") from exc


@dataclass(frozen=True)
class OrderedRecord:
    id: str
    occurrence_id: str
    source_position: int
    body: str
    raw_text: str
    prefix: str
    marker: Optional[int]


@dataclass(frozen=True)
class OrderedRecords:
    records: Tuple[OrderedRecord, ...]
    order: SourceOrder
    record_format: str
    _original_json: str

    def roundtrip(self):
        """Original JSON values, including original prefixes and all metadata."""
        return json.loads(self._original_json)

    def diagnostics(self):
        markers = [r.marker for r in self.records]
        present = bool(markers) and all(m is not None for m in markers)
        return dict(feature="declared_order_records_v1", experimental=True,
                    source_order=asdict(self.order), record_format=self.record_format,
                    input_records_sha256=hashlib.sha256(self._original_json.encode("utf-8")).hexdigest(),
                    record_ids=[r.id for r in self.records],
                    occurrence_ids=[r.occurrence_id for r in self.records],
                    marker_values=markers,
                    contiguous_markers=bool(present and all(b == a + 1 for a, b in zip(markers, markers[1:]))),
                    source_order_preserved=True, marker_is_revision_clock=False,
                    supersession_eligible=False, deleted_record_ids=[], policy="keep_all")


_PREFIX = {"step": re.compile(r"^(\[step (\d+)\] )(.*)$", re.DOTALL),
           "integer_dot": re.compile(r"^((\d+)\. )(.*)$", re.DOTALL)}


def normalize_ordered_records(records, order, *, record_format="content"):
    """Preserve each declared record as one body, with reversible prefix parsing.

    `content` accepts content/text mappings or strings. `step` and `integer_dot`
    accept the same explicit record sequence with a producer-declared prefix.
    `serial` reads the shared leading-serial lexer without detector thresholds;
    its marker is diagnostic and the declared array remains the source order.
    A single plain-text blob supplies neither trustworthy boundaries nor order.
    The source hash is a caller-supplied provenance assertion; this function
    also records a canonical hash of the received JSON records.
    """
    if isinstance(order, dict):
        try:
            order = SourceOrder(**order)
        except TypeError as exc:
            raise RecordError("invalid source order fields") from exc
    if not isinstance(order, SourceOrder):
        raise RecordError("explicit SourceOrder provenance is required")
    if record_format not in ("content", "step", "integer_dot", "serial"):
        raise RecordError("record_format must be content, step, integer_dot or serial")
    if not isinstance(records, (list, tuple)) or not records:
        raise RecordError("explicit nonempty record array required; plain text has no declared boundaries")
    try:
        original = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        original.encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RecordError("declared records must be valid UTF-8 JSON values") from exc
    out, seen = [], set()
    for i, row in enumerate(copy.deepcopy(records)):
        occurrence = "%s@%s#record%d" % (order.namespace, order.source_sha256, i)
        if isinstance(row, dict):
            raw = row.get("text", row.get("content"))
            if "text" in row and "content" in row and row["text"] != row["content"]:
                raise RecordError("text and content disagree in record %d" % i)
            rid = row.get("id", occurrence)
        else:
            raw, rid = row, occurrence
        if not isinstance(raw, str):
            raise RecordError("record %d needs a content/text string" % i)
        if not isinstance(rid, (str, int)) or isinstance(rid, bool) or not str(rid).strip():
            raise RecordError("record %d has an invalid ID" % i)
        rid = str(rid)
        if rid in seen:
            raise RecordError("duplicate record ID %r" % rid)
        seen.add(rid)
        prefix, body, marker = "", raw, None
        if record_format == "serial":
            m = LEAD.match(raw)
            if m is None or m.end(2) > 32:
                raise RecordError("record %d does not match declared serial prefix" % i)
            try:
                marker = int(m.group(2))
            except ValueError as exc:
                raise RecordError("record %d has an unsupported serial integer" % i) from exc
            prefix, body = raw[:m.end()], raw[m.end():]
        elif record_format != "content":
            m = _PREFIX[record_format].fullmatch(raw)
            if m is None:
                raise RecordError("record %d does not match declared %s prefix" % (i, record_format))
            try:
                marker = int(m.group(2))
            except ValueError as exc:
                raise RecordError("record %d has an unsupported serial integer" % i) from exc
            prefix, body = m.group(1), m.group(3)
        if not body.strip():
            raise RecordError("record %d has an empty body" % i)
        if prefix + body != raw:
            raise AssertionError("record prefix roundtrip failed")
        out.append(OrderedRecord(rid, occurrence, i, body, raw, prefix, marker))
    return OrderedRecords(tuple(out), order, record_format, original)


def build_declared(envelope, *, models=None, embed=True):
    """Build the optional record-preserving view from one JSON envelope."""
    if not isinstance(envelope, dict) or set(envelope) - {"records", "order", "format"}:
        raise RecordError("declared envelope has records, order and optional format only")
    try:
        records, order = envelope["records"], envelope["order"]
    except KeyError as exc:
        raise RecordError("declared envelope needs records and order") from exc
    from .memory import build
    return build(records, models=models, embed=embed, declared_order=order,
                 record_format=envelope.get("format", "content"))


def build_ordered_memory(records, order, *, record_format, plan, models, embed):
    from . import bpe
    from .cut import Cut
    from .memory import Memory, resolve_plan
    from .records import Record
    normalized = normalize_ordered_records(records, order, record_format=record_format)
    bodies = tuple(r.body for r in normalized.records)
    text, spans, pos = "\n\n".join(bodies), [], 0
    for body in bodies:
        spans.append((pos, pos + len(body)))
        pos += len(body) + 2
    p = resolve_plan(text, plan)
    if p.mode != "native":
        raise RecordError("declared-order records are a native opt-in mode; exact replay is unchanged")
    trace = ("Caller declares record boundaries and source-array/append order; no date sorting.",
             "One body per unit, one parent per occurrence; all versions kept without a revision contract.",
             "The base plan supplies U=%d; declared record granularity replaces automatic token packing." % p.U)
    p = p.with_(branch="record", atom=None, chunk=None, mscale=0, policy="flrs", render="r4", supersede="off",
                compat=None, source="declared-order:record-boundaries-v1",
                trace=trace)
    diagnostics = normalized.diagnostics()
    c = Cut(bodies, bodies, tuple("u:" + r.occurrence_id for r in normalized.records),
            tuple(range(len(bodies))), tuple(spans),
            tuple(len(bpe.o200k().encode(b, disallowed_special=())) for b in bodies), None, diagnostics)
    recs = [Record(text=r.body, id=r.id, meta={"occurrence_id": r.occurrence_id,
                                            "source_position": str(r.source_position)}) for r in normalized.records]
    mem = Memory(plan=p, text=text, cut=c, records=recs, record_spans=spans, models=models,
                 ordered_records=normalized)
    if mem.index.structural or mem.index.newest_row:
        raise AssertionError("source order unexpectedly enabled revision pruning")
    if embed:
        mem.embed(models)
    return mem
