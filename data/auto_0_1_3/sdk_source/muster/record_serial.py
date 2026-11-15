"""Native record-body serials; metadata remains retrieval/display data, never a clock."""
import re
import statistics

from . import bpe
from .records import normalize_newlines
from .stats import LEAD, LEAD_TAU, MAX_LINES, NUM


def _bodies(records, normalize):
    return [normalize_newlines(r.text) if normalize else r.text for r in records]


def body_serial_stats(records, *, normalize=True):
    """Dot share and lead share/median over original record bodies, not metadata or inner lines."""
    bodies = _bodies(records, normalize)[:MAX_LINES]
    dot = sum(bool(NUM.match(t)) for t in bodies) / float(max(len(bodies), 1))
    previous, steps = None, 0
    for text in bodies:
        m = LEAD.match(text)
        value = int(m.group(2)) if m and m.end(2) <= 32 else None
        if value is not None and previous is not None and value == previous + 1:
            steps += 1
        previous = value
    share = steps / float(max(len(bodies) - 1, 1))
    median = None
    if bodies and share >= LEAD_TAU:
        enc = bpe.cl100k()
        median = statistics.median([bpe.count(enc, t) for t in bodies])
    return dot, share, median


def cut_record_serials(records, text, spans, plan, *, normalize=True):
    """Cut known record boundaries; keep serialized metadata for retrieval and reader text.

    Return (Cut, body semantic keys, original record indices). Unmarked entries
    receive a sentinel, not an inferred revision clock, and their own parent.
    """
    from .cut import AS_MINREC, Cut, _SERIAL, serial_text, serial_value, slot_of, value_slots
    from .errors import PlanError
    bodies = _bodies(records, normalize)
    lead = plan.serial == "lead"
    dot = re.compile(_SERIAL.pattern, _SERIAL.flags | re.DOTALL)
    matches = []
    for raw in bodies:
        m = (LEAD if lead else dot).match(raw)
        matches.append(m if m and (not lead or m.end(2) <= 32) else None)
    share = sum(m is not None for m in matches) / float(max(len(bodies), 1))
    if share < AS_MINREC:
        raise PlanError("not a numbered record array: %.3f of record bodies carry the declared serial" % share)
    units, keys, semantic, parents, order, prefixes, unmarked = [], [], [], [], [], [], []
    display_needed = False
    for i, (raw, m) in enumerate(zip(bodies, matches)):
        shown = text[spans[i][0]:spans[i][1]]
        if m is not None:
            marker = serial_value(m.group(2) if lead else m.group(1))
            body = (raw[m.end():] if lead else m.group(2)).strip()
            prefix = raw[:m.end()] if lead else None
            parent = "s:" + slot_of(body)
        else:
            marker, body, prefix, parent = -1, raw, "", "u:record%d" % i
            unmarked.append(i)
        # Preserve the existing spelling for metadata-free, single-line records.
        # Otherwise the serialized record is both the retrieval and reader view.
        own_spelling = m is not None and shown == raw and "\n" not in raw and "\r" not in raw
        if own_spelling:
            unit, pre = body, prefix
        else:
            unit, pre = shown, ""
            display_needed = True
        units.append(unit)
        keys.append(unit)
        semantic.append(body)
        parents.append(parent)
        order.append(marker)
        prefixes.append(pre)
    if not lead:  # N8 reads dot-serial records; records under their own spelling keep the paper's slot (0.1.0)
        marked = [i for i, p in enumerate(parents) if p.startswith("s:")]
        for i, k in zip(marked, value_slots([semantic[i] for i in marked])):
            parents[i] = "s:" + k
    # If a dot array mixes decorated and undecorated records, spell the latter
    # like the existing dot renderer without dropping the decorated metadata.
    if not lead and display_needed:
        prefixes = ["[%s] " % serial_text(order[i]) if p is None else p for i, p in enumerate(prefixes)]
    actual = [o for o, m in zip(order, matches) if m is not None]
    diagnostics = dict(numbered_share=round(share, 4),
        distinct_serial_share=round(len(set(actual)) / float(max(len(actual), 1)), 4),
        n_parents=len(set(parents)), serial_source="record_bodies", unmarked_records=unmarked,
        metadata_preserved_for_retrieval_and_display=True)
    enc = bpe.o200k()
    cut = Cut(tuple(units), tuple(keys), tuple(parents), tuple(order), tuple(spans),
              tuple(bpe.count(enc, u) for u in units), "\n", diagnostics,
              tuple(prefixes) if lead or display_needed else None)
    return cut, tuple(semantic), tuple(range(len(records)))
