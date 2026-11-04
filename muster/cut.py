"""Cutting: store text -> units, the parent of each unit, and its order.

Three shapes, chosen by the plan:
  record / prose  atoms (blank-line records, or sentences) packed greedily up to `chunk` o200k tokens; an atom is
                  split only if it alone exceeds `chunk`. Parents label each unit with the `mscale`-token block its
                  first token falls in ("m<k>"), or one parent per unit ("u<i>") when there is no parent layer.
  numbered        one unit per "<serial>. <body>" line, serial stripped; order = the serial; parent = the slot the
                  record updates ("s:" + entity | relation skeleton). A plan with serial "lead" reads each line's
                  own prefix instead ("Step 3: <body>") and keeps it for the render.
  rounds          a LongMemEval-style literal [date, turns, date, turns, ...]: one unit per user turn with the
                  assistant reply, parent = the session.

In exact mode every step is the harness's (harness/lme_pf29.py:1224-1291, 2089-2146, 2232-2247, 1059-1101), and the
harness's refusals become StoreRefused. Native mode departs only where the harness would crash or split a character.
"""
import ast
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple

from . import bpe
from .errors import AssetMissing, PlanError, StoreRefused
from .stats import LEAD, lead_match

_BLANK = re.compile(r"\n\s*\n")
JOIN = {"blank": "\n\n", "newline": "\n", "sentence": " "}


@dataclass(frozen=True)
class Cut:
    units: Tuple[str, ...]
    keys: Tuple[str, ...]                 # what is embedded: the unit, except rounds (the user turn)
    parent: tuple                         # "m%d", "u%d", "s:<slot>" or an int session, as the harness labels them
    order: tuple                          # the serial on numbered logs, else 0..N-1
    spans: Tuple[Optional[Tuple[int, int]], ...]
    n_o200k: Tuple[int, ...]
    sep: Optional[str]
    diagnostics: dict = field(default_factory=dict)
    prefixes: Optional[Tuple[str, ...]] = None   # serial "lead": each line's own prefix, "" where it has none

    def __len__(self):
        return len(self.units)


# ---------------------------------------------------------------- atoms
def _ensure_punkt(exact):
    import nltk
    d = os.path.join(bpe.asset_dir(), "nltk_data")
    if os.path.isdir(d) and d not in nltk.data.path:
        nltk.data.path.insert(0, d)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        if exact:
            raise AssetMissing("sentence splitting needs nltk's punkt_tab; run `muster fetch`")
        os.makedirs(d, exist_ok=True)
        nltk.download("punkt_tab", download_dir=d, quiet=True)
        if d not in nltk.data.path:
            nltk.data.path.insert(0, d)
    return nltk


def _with_spans(text, pieces):
    out, cur = [], 0
    for p in pieces:
        j = text.find(p, cur)
        if j < 0:
            out.append((p, None))
        else:
            out.append((p, (j, j + len(p))))
            cur = j + len(p)
    return out


def atoms(text: str, kind: str, *, exact: bool = False):
    """[(atom, (start, end) or None)] for kind blank, newline or sentence (lme_pf29.py:1224-1236)."""
    if kind == "blank":
        out, pos = [], 0
        for m in _BLANK.finditer(text):
            piece = text[pos:m.start()]
            if piece.strip():
                out.append((piece, (pos, m.start())))
            pos = m.end()
        piece = text[pos:]
        if piece.strip():
            out.append((piece, (pos, len(text))))
        return out
    if kind == "newline":
        out, pos = [], 0
        for piece in text.split("\n"):
            if piece.strip():
                out.append((piece, (pos, pos + len(piece))))
            pos += len(piece) + 1
        return out
    if kind == "sentence":
        nltk = _ensure_punkt(exact)
        return _with_spans(text, nltk.sent_tokenize(text))
    raise PlanError("unknown atom kind %r" % kind)


# ---------------------------------------------------------------- packing
def _special_strings(enc, text):
    return sorted(s for s in enc.special_tokens_set if s in text)


def pack(atom_list, chunk: int, joiner: str, *, exact: bool = False):
    """Greedy packing (lme_pf29.py:1255-1291). Returns [(unit, [atom indices])] and the oversize-atom count."""
    enc = bpe.o200k()
    out, cur, curi, n, over = [], [], [], 0, 0
    for i, a in enumerate(atom_list):
        if exact:
            t = bpe.encode_pack(enc, a)
        else:
            t = enc.encode(a, disallowed_special=())
        if len(t) > chunk:
            over += 1
            if cur:
                out.append((joiner.join(cur), curi))
                cur, curi, n = [], [], 0
            if exact:
                for o in range(0, len(t), chunk):
                    out.append((enc.decode(t[o:o + chunk]), [i]))
            else:                           # N5: slice on character boundaries, so the slices rebuild the atom
                _, offs = enc.decode_with_offsets(t)
                cuts = [offs[o] for o in range(0, len(t), chunk)] + [len(a)]
                for s, e in zip(cuts, cuts[1:]):
                    if e > s:
                        out.append((a[s:e], [i]))
            continue
        if n + len(t) > chunk and cur:
            out.append((joiner.join(cur), curi))
            cur, curi, n = [], [], 0
        cur.append(a)
        curi.append(i)
        n += len(t)
    if cur:
        out.append((joiner.join(cur), curi))
    return out, over


# ---------------------------------------------------------------- numbered logs (AUTOSTRUCT, AS_SLOT=5)
_SERIAL = re.compile(r"^\s*(\d+)\.\s+(.+)$", re.DOTALL)
_CAPRE = re.compile(r"\b([A-Z][\w.'-]+(?:\s+(?:of|the|and|de|van|von|di)?\s*[A-Z][\w.'-]+)*)\b")
_CAPSTOP = {"The", "A", "An", "In", "On", "At", "Of", "This", "That", "These", "Those", "It", "Its", "Which",
            "What", "Who", "Where", "When", "There", "He", "She", "They", "His", "Her", "Their"}
AS_MINREC = 0.8


def entities(t):
    """Capitalised spans, minus sentence-initial function words (lme_pf29.py:2107-2110)."""
    return [m.group(1) for m in _CAPRE.finditer(t or "") if m.group(1).split()[0] not in _CAPSTOP]


def slot_of(body):
    """The slot a numbered record updates: its first entity, then the sentence with every entity masked."""
    e = entities(body)
    sub = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", e[0].lower())).strip() if e else ""
    sk = re.sub(r"\s+", " ", _CAPRE.sub("_", body or "")).strip().rstrip(".").strip()
    return sub + "|" + sk


def _numbered(text, exact, serial="dot"):
    lines = [ln for ln in text.split("\n") if ln.strip()]
    lead = serial == "lead"
    ms = [(lead_match(ln) if not exact else LEAD.match(ln)) if lead else _SERIAL.match(ln) for ln in lines]
    share = sum(1 for x in ms if x) / float(max(len(lines), 1))
    if share < AS_MINREC:
        if exact:
            raise StoreRefused("numbered log: only %.3f of lines are `<serial>. <body>` (needs %.1f; lme_pf29.py:2126)"
                               % (share, AS_MINREC))
        raise PlanError("not a numbered log: %.3f of all lines carry a %sserial" % (share, "leading " if lead else ""))
    units, order, parent, prefixes = [], [], [], []
    unmarked = []
    for x, ln in zip(ms, lines):
        if x and lead:
            ser, body, pre = int(x.group(2)), ln[x.end():].strip(), ln[:x.end()]
        elif x:
            ser, body, pre = int(x.group(1)), x.group(2).strip(), None
        else:
            ser, body, pre = len(units), ln.strip(), ""
            if lead and not exact:
                ser, body = -1, ln  # preserve the literal line; no invented revision clock
                unmarked.append(len(units))
        units.append(body)
        order.append(ser)
        parent.append("u:rawline%d" % (len(units) - 1) if lead and not exact and x is None else "s:" + slot_of(body))
        prefixes.append(pre)
    spans = tuple(sp for _, sp in _with_spans(text, lines))
    enc = bpe.o200k()
    serials = [o for o, x in zip(order, ms) if x]
    diag = dict(numbered_share=round(share, 4), distinct_serial_share=round(len(set(serials)) / float(max(len(serials), 1)), 4),
                n_parents=len(set(parent)))
    if unmarked:
        diag["unmarked_records"] = unmarked
    return Cut(tuple(units), tuple(units), tuple(parent), tuple(order), spans,
               tuple(bpe.count(enc, u) for u in units), "\n", diag, tuple(prefixes) if lead else None)


# ---------------------------------------------------------------- rounds (LongMemEval literal)
def sessions_from_literal(literal: str):
    """[(date, turns)] from "[date, turns, date, turns, ...]" (lme_pf29.py:1059-1072)."""
    lit = ast.literal_eval(literal)
    out, i = [], 0
    while i < len(lit):
        if isinstance(lit[i], str) and i + 1 < len(lit) and isinstance(lit[i + 1], list):
            out.append((lit[i], lit[i + 1]))
            i += 2
        else:
            i += 1
    return out


def rounds_of(sessions):
    """One unit per user turn with the assistant turn that answers it (lme_pf29.py:1075-1101)."""
    rtext, rkey, sess_id, rdate = [], [], [], []
    for si, (date, turns) in enumerate(sessions):
        for ti, t in enumerate(turns):
            if not isinstance(t, dict) or t.get("role") != "user":
                continue
            nxt = ""
            if ti + 1 < len(turns) and isinstance(turns[ti + 1], dict) and turns[ti + 1].get("role") != "user":
                nxt = turns[ti + 1].get("content", "")
            rtext.append(("[" + date + "] " + t.get("content", "") + "\n" + nxt).strip())
            rkey.append(t.get("content", "").strip())
            sess_id.append(si)
            rdate.append(date)
    return rtext, rkey, sess_id, rdate


def _rounds(literal):
    sess = sessions_from_literal(literal)
    rtext, rkey, sid, rdate = rounds_of(sess)
    enc = bpe.o200k()
    return Cut(tuple(rtext), tuple(rkey), tuple(sid), tuple(range(len(rtext))), tuple([None] * len(rtext)),
               tuple(bpe.count(enc, u) for u in rtext), None, dict(n_sessions=len(sess), dates=tuple(rdate)))


# ---------------------------------------------------------------- entry point
def cut(text: str, plan, *, exact: Optional[bool] = None) -> Cut:
    """Cut `text` under `plan`. `exact` defaults to plan.mode == "exact"."""
    exact = (plan.mode == "exact") if exact is None else exact
    if plan.branch == "empty":
        return Cut((), (), (), (), (), (), None, {})
    if plan.branch == "numbered":
        return _numbered(text, exact, plan.serial)
    if plan.branch == "rounds":
        return _rounds(text)
    enc = bpe.o200k()
    if exact:
        bad = [s for s in _special_strings(enc, text) if s != "<|endoftext|>"]
        if bad:
            raise StoreRefused("the store contains special-token text %r, which the harness's packer cannot encode "
                               "(lme_pf29.py:1273)" % bad[0])
    at = atoms(text, plan.atom, exact=exact)
    packed, over = pack([a for a, _ in at], plan.chunk, JOIN[plan.atom], exact=exact)
    units = [u for u, _ in packed]
    spans = []
    for _, idx in packed:
        s, e = at[idx[0]][1], at[idx[-1]][1]
        spans.append((s[0], e[1]) if (s and e) else None)
    diag = dict(atoms=len(at), oversize_atoms=over)
    if plan.mscale:
        if exact and not units:
            raise StoreRefused("MSCALE produced no units (lme_pf29.py:2239)")
        parent, cum = [], 0
        for u in units:
            parent.append("m%d" % (cum // plan.mscale))
            if exact:
                try:
                    cum += len(bpe.encode_label(enc, u))
                except ValueError as e:
                    raise StoreRefused("a unit holds special-token text the harness's parent labels cannot encode "
                                       "(lme_pf29.py:2237): %s" % e)
            else:
                cum += bpe.count(enc, u)
        if len(set(parent)) >= len(units):
            if exact:
                raise StoreRefused("MSCALE produced a flat layer (%d units, %d parents) (lme_pf29.py:2240-2243)"
                                   % (len(units), len(set(parent))))
            parent = ["u%d" % i for i in range(len(units))]
            diag["flat_parents"] = True
        if exact:
            ns = lambda t: len(re.sub(r"\s+", "", t))
            lost = ns(text) - ns("".join(units))
            if abs(lost) > 0.01 * max(1, ns(text)):
                raise StoreRefused("MSCALE lost or duplicated text: %d non-space characters (lme_pf29.py:2244-2247)" % lost)
    else:
        parent = ["u%d" % i for i in range(len(units))]
    return Cut(tuple(units), tuple(units), tuple(parent), tuple(range(len(units))), tuple(spans),
               tuple(bpe.count(enc, u) for u in units), "\n\n" if plan.atom != "newline" else "\n", diag)
