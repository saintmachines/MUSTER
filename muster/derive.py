"""derive: a store's plan from its own text, with no model, no query and nothing to tune.

The rule (paper, Algorithm 1, Part I; thresholds fixed before any experiment, harness/derive_config.py):

  numbered log  >= 80% of lines open with a serial "12. ..."  -> one unit per record, parent = the slot it updates,
                the newest serial of a slot wins
  record store  >= 90% of the text sits in blank-line records of at most the reference chunk P
                -> units of max(P/8, 512) tokens with records kept whole
  prose         otherwise -> units of P/4 tokens (step 8: the P-token unit searched at a quarter of its size with four
                times as many units, so the token budget is unchanged)

  Parents are 8 units, or 4 or 2 if fewer than ten parents would remain; else there is no parent layer.

Added after the paper (2026-11, thresholds set on public benchmark logs; `derive` only, not `derive_collection` or the
presets): a store the serial fraction misses is still a numbered log when >= 95% of its lines after the first carry a
leading serial one above the previous line's, written in the log's own style ("Step 3: ..."), and its median line is
at most 20 tokens. The serial is read from that prefix, and the reader sees each line as written.
"""
from typing import Iterable, Sequence, Union
from dataclasses import replace

from .errors import PlanError
from .plan import Plan
from .records import DEFAULT_TEMPLATE, as_record, normalize_newlines, serialize
from .record_serial import body_serial_stats
from .stats import LEAD_LINE_TOKENS, LEAD_TAU, Stats, collection_stats, lead_serial, store_stats

NUMBERED_TAU, RECORD_TAU, MIN_PARENTS = 0.80, 0.90, 10
RECORD_STEP, MIN_UNIT, PROSE_STEP = 8, 512, 4
U_DEFAULT = {"numbered": 10, "record": 10, "prose": 40}


def store_text(store, *, template=DEFAULT_TEMPLATE, normalize=True):
    """(text, record spans or None, records or None) for a string or an iterable of records."""
    if isinstance(store, str):
        return (normalize_newlines(store) if normalize else store), None, None
    recs = [as_record(x) for x in store]
    text, spans = serialize(recs, template=template, normalize=normalize)
    return text, spans, recs


def _parents(units_est, chunk):
    for m in (8, 4, 2):
        if units_est // m >= MIN_PARENTS:
            return chunk * m, units_est // m, m
    return 0, 0, None


def _plan(branch, chunk, units_est, st: Stats, P, trace, serial="dot"):
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
    return Plan(branch=branch, atom=atom, chunk=chunk, mscale=ms, U=U_DEFAULT[branch], policy="flrs", render="r4",
                reference_chunk=P, stats=st.to_dict(), trace=tuple(trace))


def derive(store, *, reference_chunk: int = 4096, template: str = DEFAULT_TEMPLATE,
           normalize_newlines: bool = True) -> Plan:
    """The plan of one store, with the documented native serial-format extension."""
    P = int(reference_chunk)
    text, _, recs = store_text(store, template=template, normalize=normalize_newlines)
    st = store_stats(text, P)
    body_stats = body_serial_stats(recs, normalize=normalize_newlines) if recs is not None else None
    if body_stats is not None:
        st = replace(st, serial=body_stats[0])  # metadata never supplies a record serial
    trace = ["serial fraction %.4f, record share %.4f, %d tokens (cl100k_base)" % (st.serial, st.record_share, st.tokens)]
    if st.tokens == 0:
        trace.append("empty store")
        return Plan(branch="empty", atom=None, chunk=None, mscale=0, U=0, policy="flr", render="flat",
                    reference_chunk=P, stats=st.to_dict(), trace=tuple(trace))
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
    if st.record_share >= RECORD_TAU:
        chunk = max(P // RECORD_STEP, MIN_UNIT)
        trace.append("record share %.4f >= %.2f: record store; units of %d tokens, records kept whole"
                     % (st.record_share, RECORD_TAU, chunk))
        return _plan("record", chunk, st.tokens // chunk, st, P, trace)
    chunk = P // PROSE_STEP if P > MIN_UNIT else P
    trace.append("record share %.4f < %.2f: prose; units of %d tokens (P/%d, step 8)"
                 % (st.record_share, RECORD_TAU, chunk, PROSE_STEP))
    return _plan("prose", chunk, st.tokens // chunk, st, P, trace)


def derive_collection(stores: Sequence[str], *, reference_chunk: int = 4096, sample: int = 3, step8: bool = True,
                      mode: str = "native") -> Plan:
    """One plan for a benchmark cell from its first `sample` stores (harness/derive_config.py, then step 8)."""
    P = int(reference_chunk)
    if P not in (512, 4096):
        raise PlanError("reference_chunk must be 512 or 4096, MemoryAgentBench's protocol sizes; got %d" % P)
    st = collection_stats(stores, P, sample=sample, mode=mode)
    trace = ["%d stores: serial fraction %.4f (mean), record share %.4f (pooled), %d tokens per store"
             % (st.rows, st.serial, st.record_share, st.tokens)]
    if st.serial >= NUMBERED_TAU:
        return _plan("numbered", None, 0, st, P, trace)
    record = st.record_share >= RECORD_TAU
    chunk = P // RECORD_STEP if (record and P // RECORD_STEP >= MIN_UNIT) else P
    if not record and step8 and P > MIN_UNIT:
        chunk = P // PROSE_STEP
    trace.append("%s; units of %d tokens" % ("record store" if record else "prose", chunk))
    return _plan("record" if record else "prose", chunk, max(1, st.tokens // chunk), st, P, trace)


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
        rows += [("units", "one user turn with its answer"), ("parents", "one per session")]
    rows += [("read", "%d units per query" % plan.U), ("mode", plan.mode), ("source", plan.source)]
    out = ["  %-10s %s" % r for r in rows]
    if plan.trace:
        out.append("  how:")
        out += ["    - " + t for t in plan.trace]
    return "\n".join(out)
