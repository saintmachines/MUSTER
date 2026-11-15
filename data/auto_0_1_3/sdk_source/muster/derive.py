"""derive: a store's plan from its own text, with no model, no query and nothing to tune.

The rule (paper, Algorithm 1, Part I; thresholds fixed before any experiment, harness/derive_config.py):

  numbered log  >= 80% of lines open with a serial "12. ..."  -> one unit per record, parent = the slot it updates,
                the newest serial of a slot wins
  record store  >= 90% of the text sits in blank-line records of at most the reference chunk P
                -> units of max(P/8, 512) tokens with records kept whole
  prose         otherwise -> units of P/4 tokens (step 8: the P-token unit searched at a quarter of its size with four
                times as many units, so the token budget is unchanged)

  Parents are 8 units, or 4 or 2 if fewer than ten parents would remain; else there is no parent layer.
  Native recipes: independent records use flrs/flat, dated speaker records flr/r4, prose flr/r4.
  An alternating date/turn-list literal uses user/answer rounds under its source sessions, flrs/flat.

Added after the paper (2026-11, thresholds set on public benchmark logs): a store the serial fraction misses is still
a numbered log when >= 95% of its lines after the first carry a
leading serial one above the previous line's, written in the log's own style ("Step 3: ..."), and its median line is
at most 20 tokens. The serial is read from that prefix, and the reader sees each line as written. Native collections
require all nonempty members to have the same serial format; historical exact statistics and presets stay unchanged.
"""
import ast
import operator
import re
from collections.abc import Mapping
from typing import Optional, Sequence
from dataclasses import replace

from .errors import PlanError, RecordError
from .plan import Plan
from .records import DEFAULT_TEMPLATE, Record, as_record, normalize_newlines, record_list, require_text, serialize
from .record_serial import body_serial_stats
from .stats import LEAD_LINE_TOKENS, LEAD_TAU, Stats, collection_stats, lead_serial, store_stats

NUMBERED_TAU, RECORD_TAU, MIN_PARENTS = 0.80, 0.90, 10
RECORD_STEP, MIN_UNIT, PROSE_STEP = 8, 512, 4
U_DEFAULT = {"numbered": 10, "record": 10, "prose": 40}
MAX_REVISIONS = PROSE_STEP  # value_slots in cut.py: a changed subject of a relation holds at most this many lines on average

_DATED_SPEAKER = re.compile(
    r"^\s*(?:[^\s\[\]]{1,64}\s+)?\[(?P<time>[^\]\n]{1,128})\]\s+"
    r"(?P<speaker>[^:\n]{1,64}):[ \t]+\S"
)
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?"
          r"|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_TIME = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b"
                   r"|\b" + _MONTH + r"[- ]\d{1,2}(?:st|nd|rd|th)?,?[- ]\d{4}\b"
                   r"|\b\d{1,2}(?:st|nd|rd|th)?[- ]" + _MONTH + r",?[- ]\d{4}\b", re.IGNORECASE)
_TURN_HEAD = re.compile(
    r"^[ \t]*(?:[^\s\[\]]{1,64}[ \t]+)?(?:\[(?P<time>[^\]\n]{1,128})\][ \t]+)?"
    r"(?P<speaker>[A-Za-z][\w .'-]{0,31}):[ \t]+\S"
)
TURN_BRACKET_SHARE = 0.10


def _reference_chunk(value):
    try:
        if isinstance(value, bool):
            raise TypeError
        result = operator.index(value)
    except TypeError as exc:
        raise PlanError("reference_chunk must be a positive whole number") from exc
    if result < 1:
        raise PlanError("reference_chunk must be a positive whole number")
    return result


def _sample_size(sample):
    if sample is None:
        return None
    try:
        if isinstance(sample, bool):
            raise TypeError
        result = operator.index(sample)
    except TypeError as exc:
        raise PlanError("sample must be a positive whole number or None for all stores") from exc
    if result < 1:
        raise PlanError("sample must be a positive whole number or None for all stores")
    return result


def _conversation_share(text, records=None):
    """Dated speaker boundaries, not names or words inside the message body."""
    if records is not None:
        bodies = [r for r in records if r.text.strip()]
        return sum(bool(r.time and _TIME.search(r.time) and r.speaker and r.speaker.strip())
                   for r in bodies) / float(max(len(bodies), 1))
    blocks = [part for part in text.split("\n\n") if part.strip()]
    share = 0.0
    if len(blocks) >= 2:
        count = 0
        for block in blocks:
            first = next(line for line in block.split("\n") if line.strip())
            match = _DATED_SPEAKER.match(first)
            count += bool(match and _TIME.search(match.group("time")))
        share = count / float(len(blocks))
    if share >= NUMBERED_TAU:
        return share
    return max(share, _turn_share(text))


def _turn_share(text):
    """Dated source turns, including message bodies with line and paragraph breaks.

    A speaker must open a bracketed source header before its unbracketed
    headers can start turns. A date in an earlier header carries forward.
    The result is the share of nonblank characters inside dated turns.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    heads = []
    for i, line in enumerate(lines):
        match = _TURN_HEAD.match(line)
        if match:
            heads.append((i, match))
    speakers = {match.group("speaker").strip() for _, match in heads if match.group("time")}
    starts = {i: match for i, match in heads if match.group("speaker").strip() in speakers}
    if len(starts) < 2 or sum(1 for match in starts.values() if match.group("time")) < TURN_BRACKET_SHARE * len(starts):
        return 0.0
    total = inside = 0
    dated = in_turn = False
    for i, line in enumerate(lines):
        size = len(line.strip())
        total += size
        match = starts.get(i)
        if match is not None:
            in_turn = True
            if match.group("time") and _TIME.search(match.group("time")):
                dated = True
        if in_turn and dated:
            inside += size
    return inside / float(max(total, 1))


def _is_rounds_literal(text):
    """Only a complete alternating date/turn-list source literal is a rounds store."""
    if not text.lstrip().startswith("["):
        return False
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return False
    if not isinstance(value, list) or not value or len(value) % 2:
        return False
    user = False
    for date, turns in zip(value[::2], value[1::2]):
        if not isinstance(date, str) or not date.strip() or not isinstance(turns, list) or not turns:
            return False
        try:
            require_text(date, "the session date")
        except RecordError:
            return False
        for turn in turns:
            if not isinstance(turn, dict) or not {"role", "content"} <= set(turn):
                return False
            if turn["role"] not in ("user", "assistant", "system") or not isinstance(turn["content"], str):
                return False
            try:
                require_text(turn["content"], "the session turn content")
            except RecordError:
                return False
            user |= turn["role"] == "user"
    return user


def _empty_plan(st, P, trace):
    trace.append("empty store")
    return Plan(branch="empty", atom=None, chunk=None, mscale=0, U=0, policy="flr", render="flat",
                reference_chunk=P, stats=st.to_dict(), trace=tuple(trace))


def _rounds_plan(st, P, trace):
    trace.append("alternating date/turn-list literal: user/answer rounds and unpaired turns, parent = source session")
    return Plan(branch="rounds", atom=None, chunk=None, mscale=0, U=10, policy="flrs", render="flat",
                reference_chunk=P, stats=st.to_dict(), trace=tuple(trace))


def store_text(store, *, template=DEFAULT_TEMPLATE, normalize=True):
    """(text, record spans or None, records or None) for a string or an iterable of records."""
    if isinstance(store, str):
        require_text(store, "the store")
        return (normalize_newlines(store) if normalize else store), None, None
    recs = [as_record(x) for x in record_list(store)]
    text, spans = serialize(recs, template=template, normalize=normalize)
    return text, spans, recs


def _parents(units_est, chunk):
    for m in (8, 4, 2):
        if units_est // m >= MIN_PARENTS:
            return chunk * m, units_est // m, m
    return 0, 0, None


def _plan(branch, chunk, units_est, st: Stats, P, trace, serial="dot", *, conversation=False, legacy=False):
    if branch == "numbered":
        if serial == "dot":
            trace.append("serial fraction %.4f >= %.2f: numbered log; one unit per record, parent = slot, newest "
                         "serial wins" % (st.serial, NUMBERED_TAU))
        return Plan(branch="numbered", atom=None, chunk=None, mscale=0, U=U_DEFAULT["numbered"],
                    policy="attprio_sess", render="numbered", supersede="numbered", reference_chunk=P,
                    stats=st.to_dict(), trace=tuple(trace), serial=serial)
    atom = "blank" if branch == "record" else "sentence"
    ms, npar, m = _parents(units_est, chunk)
    trace.append("%d units estimated at %d tokens -> %s" % (
        units_est, chunk, ("parents of %d units (%d tokens), %d of them" % (m, ms, npar)) if ms
        else "no parent layer (fewer than %d parents at 2 units each)" % MIN_PARENTS))
    policy, render = ("flrs", "r4") if legacy else (
        ("flr", "r4") if branch == "prose" or conversation else ("flrs", "flat"))
    if not legacy:
        trace.append("%s: policy %s, render %s" % (
            "dated speaker records" if conversation and branch == "record" else branch, policy, render))
    return Plan(branch=branch, atom=atom, chunk=chunk, mscale=ms, U=U_DEFAULT[branch], policy=policy, render=render,
                reference_chunk=P, stats=st.to_dict(), trace=tuple(trace))


def derive(store, *, reference_chunk: int = 4096, template: str = DEFAULT_TEMPLATE,
           normalize_newlines: bool = True) -> Plan:
    """The plan of one store, with the documented native serial-format extension."""
    P = _reference_chunk(reference_chunk)
    text, _, recs = store_text(store, template=template, normalize=normalize_newlines)
    st = store_stats(text, P)
    body_stats = body_serial_stats(recs, normalize=normalize_newlines) if recs is not None else None
    if body_stats is not None:
        st = replace(st, serial=body_stats[0])  # metadata never supplies a record serial
    trace = ["serial fraction %.4f, record share %.4f, %d tokens (cl100k_base)" % (st.serial, st.record_share, st.tokens)]
    if st.tokens == 0:
        return _empty_plan(st, P, trace)
    if st.serial >= NUMBERED_TAU:
        if body_stats is not None:
            trace.append("Serials are read only at the start of original record bodies; metadata is retrieval/display data.")
        return _plan("numbered", None, 0, st, P, trace)
    share, med = lead_serial(text) if body_stats is None else body_stats[1:]
    if med is not None and med <= LEAD_LINE_TOKENS:
        if body_stats is None:
            trace.append("leading serial one above the previous line's on %.4f >= %.2f of lines, median line %g <= %d "
                         "tokens: numbered log under its own spelling; one unit per record, parent = slot, newest serial "
                         "wins" % (share, LEAD_TAU, med, LEAD_LINE_TOKENS))
        else:
            trace.append("leading body serial one above the previous record's on %.4f >= %.2f of original record "
                         "pairs, median entire record body %g <= %d tokens: numbered records; metadata kept for "
                         "retrieval/display, not slot/order" % (share, LEAD_TAU, med, LEAD_LINE_TOKENS))
        return _plan("numbered", None, 0, st, P, trace, serial="lead")
    if recs is None and _is_rounds_literal(text):
        return _rounds_plan(st, P, trace)
    conversation = _conversation_share(text, recs)
    if conversation >= NUMBERED_TAU:
        trace.append("dated speaker record share %.4f >= %.2f" % (conversation, NUMBERED_TAU))
    return _unserial_plan(st, P, trace, conversation=conversation >= NUMBERED_TAU)


def _unserial_plan(st, P, trace, *, conversation=False):
    if st.record_share >= RECORD_TAU:
        chunk = max(P // RECORD_STEP, MIN_UNIT)
        trace.append("record share %.4f >= %.2f: record store; units of %d tokens, records kept whole"
                     % (st.record_share, RECORD_TAU, chunk))
        return _plan("record", chunk, st.tokens // chunk, st, P, trace, conversation=conversation)
    chunk = P // PROSE_STEP if P > MIN_UNIT else P
    trace.append("record share %.4f < %.2f: prose; units of %d tokens (P/%d, step 8)"
                 % (st.record_share, RECORD_TAU, chunk, PROSE_STEP))
    return _plan("prose", chunk, st.tokens // chunk, st, P, trace)


def _prepare_collection(stores, *, template=DEFAULT_TEMPLATE, normalize=True):
    if isinstance(stores, (str, bytes, Record, Mapping, set, frozenset)):
        raise PlanError("stores must be an ordered collection of independent stores")
    try:
        items = list(stores)
    except TypeError as exc:
        raise PlanError("stores must be an ordered collection of independent stores") from exc
    prepared = []
    for item in items:
        text, _, recs = store_text(item, template=template, normalize=normalize)
        prepared.append((text if recs is None else recs, text, recs))
    return prepared


def _derive_prepared_collection(prepared, *, reference_chunk=4096, sample=3, step8=True,
                                template=DEFAULT_TEMPLATE, normalize=True):
    P, sample = _reference_chunk(reference_chunk), _sample_size(sample)
    texts = [text for _, text, _ in prepared]
    st = collection_stats(texts, P, sample=sample, mode="native")
    plans = [derive(source, reference_chunk=P, template=template, normalize_newlines=normalize)
             for source, _, _ in prepared]
    sampled = plans if sample is None else plans[:sample]
    serials = [p.stats["serial"] for p in sampled if p.stats["tokens"]]
    st = replace(st, serial=sum(serials) / len(serials) if serials else 0.0)
    trace = [
        "collection: %s; %d of %d stores supply statistics; all stores checked for compatible source boundaries"
        % ("all stores (sample=None)" if sample is None else "first %d stores (sample=%d)" % (sample, sample),
           st.rows, len(prepared)),
        "%d stores: serial fraction %.4f (mean), record share %.4f (pooled), %d tokens per store"
        % (st.rows, st.serial, st.record_share, st.tokens),
        "shared parent estimate uses mean sampled whole-store tokens; stores are cut separately, never concatenated",
    ]
    active = [p for p in plans if p.branch != "empty"]
    if not active:
        return _empty_plan(st, P, trace).with_(source="derived:collection")
    if not any(p.branch != "empty" for p in sampled):
        raise PlanError("sample contains only empty stores; sample=None uses all stores")
    numbered = [p for p in active if p.branch == "numbered"]
    if numbered:
        if len(numbered) != len(active) or len({p.serial for p in numbered}) != 1:
            raise PlanError("collection mixes numbered serial formats or numbered and unnumbered stores")
        trace.append("all nonempty stores have compatible body serials; each source keeps its own clock")
        return _plan("numbered", None, 0, st, P, trace, serial=numbered[0].serial).with_(source="derived:collection")
    rounds = [p for p in active if p.branch == "rounds"]
    if rounds:
        if len(rounds) != len(active):
            raise PlanError("collection mixes session literals and other source structures")
        return _rounds_plan(st, P, trace).with_(source="derived:collection")
    conversation = [_conversation_share(text, recs) >= NUMBERED_TAU
                    for (_, text, recs), p in zip(prepared, plans) if p.branch != "empty"]
    if any(conversation) and not all(conversation):
        trace.append("mixed dated-speaker and independent stores: use the common branch's recipe without a conversation override")
    record = st.record_share >= RECORD_TAU
    chunk = max(P // RECORD_STEP, MIN_UNIT) if record else P
    if not record and step8 and P > MIN_UNIT:
        chunk = P // PROSE_STEP
    trace.append("%s; units of %d tokens" % ("record store" if record else "prose", chunk))
    return _plan("record" if record else "prose", chunk, max(1, st.tokens // chunk), st, P, trace,
                 conversation=all(conversation)).with_(source="derived:collection")


def derive_collection(stores: Sequence, *, reference_chunk: int = 4096, sample: Optional[int] = 3,
                      step8: bool = True, mode: str = "native", template: str = DEFAULT_TEMPLATE,
                      normalize_newlines: bool = True) -> Plan:
    """A shared plan from the first three independent stores, or all with sample=None.

    Native mode validates source boundaries across the supplied collection;
    only the selected prefix supplies pooled statistics and the mean size.
    ``mode="exact"`` retains the historical collection-statistics path.
    """
    if mode == "exact":
        P = int(reference_chunk)
        if P not in (512, 4096):
            raise PlanError("reference_chunk must be 512 or 4096, MemoryAgentBench's protocol sizes; got %d" % P)
        st = collection_stats(stores, P, sample=sample, mode=mode)
        trace = ["%d stores: serial fraction %.4f (mean), record share %.4f (pooled), %d tokens per store"
                 % (st.rows, st.serial, st.record_share, st.tokens)]
        if st.serial >= NUMBERED_TAU:
            return _plan("numbered", None, 0, st, P, trace, legacy=True)
        record = st.record_share >= RECORD_TAU
        chunk = P // RECORD_STEP if (record and P // RECORD_STEP >= MIN_UNIT) else P
        if not record and step8 and P > MIN_UNIT:
            chunk = P // PROSE_STEP
        trace.append("%s; units of %d tokens" % ("record store" if record else "prose", chunk))
        return _plan("record" if record else "prose", chunk, max(1, st.tokens // chunk), st, P, trace, legacy=True)
    if mode != "native":
        raise PlanError("mode must be native or exact")
    prepared = _prepare_collection(stores, template=template, normalize=normalize_newlines)
    return _derive_prepared_collection(prepared, reference_chunk=reference_chunk, sample=sample, step8=step8,
                                       template=template, normalize=normalize_newlines)


def explain(plan: Plan) -> str:
    """A plain description of a plan, then the decision step by step."""
    rows = [("store", {"numbered": "numbered log", "record": "record store", "prose": "prose", "rounds": "conversation rounds",
                       "empty": "empty"}[plan.branch])]
    if plan.source == "declared-order:record-boundaries-v1":
        rows += [("units", "one per declared record"), ("parents", "one per source occurrence"),
                 ("order", "declared source sequence; all versions kept")]
    elif plan.branch == "numbered":
        rows += [("units", "one per record"), ("parents", "one per slot"), ("order", "newest per slot wins")]
        if plan.serial == "lead":
            rows.append(("serials", "each line's own prefix, shown as written"))
    elif plan.branch in ("record", "prose"):
        rows += [("units", "%s tokens%s" % (format(plan.chunk, ","), ", records whole" if plan.atom == "blank" else "")),
                 ("parents", ("%s tokens" % format(plan.mscale, ",")) if plan.mscale else "none")]
    elif plan.branch == "rounds":
        rows += [("units", "user/answer rounds and unpaired turns"), ("parents", "one per session")]
    rows += [("read", "%d units per query" % plan.U), ("mode", plan.mode), ("source", plan.source)]
    out = ["  %-10s %s" % r for r in rows]
    if plan.trace:
        out.append("  how:")
        out += ["    - " + t for t in plan.trace]
    return "\n".join(out)
