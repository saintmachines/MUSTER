"""Records: what an application puts into memory, and how they become the stored text.

A store is text. A list of records is written as one line per record, records separated by a blank line, which is
the shape of the paper's conversation stores (LoCoMo's "<id> [<time>] <speaker>: <text>").
"""
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

from .errors import RecordError

DEFAULT_TEMPLATE = "{id} [{time}] {speaker}: {text}"
RECORD_SEP = "\n\n"


@dataclass(frozen=True)
class Record:
    text: str
    id: Optional[str] = None
    time: Optional[str] = None
    speaker: Optional[str] = None
    session: Optional[str] = None
    slot: Optional[str] = None
    order: Optional[float] = None
    status: str = "accepted"
    ref: Optional[str] = None
    kind: str = "text"
    sort_key: Optional[tuple] = None
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise RecordError("Record.text must be a string, got %s" % type(self.text).__name__)
        t = self.time
        if isinstance(t, (datetime, date)):
            object.__setattr__(self, "time", t.isoformat())
        elif isinstance(t, (int, float)) and not isinstance(t, bool):
            raise RecordError("Record.time %r is a number: seconds or milliseconds is ambiguous; pass a datetime or a string" % t)
        elif t is not None and not isinstance(t, str):
            raise RecordError("Record.time must be a string or a datetime")
        for k in ("id", "speaker", "session"):
            v = getattr(self, k)
            if v is not None and not isinstance(v, str):
                object.__setattr__(self, k, str(v))
        if self.status not in ("accepted", "candidate", "rejected"):
            raise RecordError("Record.status must be accepted, candidate or rejected")


def as_record(x: Union[Record, Mapping[str, Any], str]) -> Record:
    """A Record from a Record, a mapping (text|content, id, time|timestamp, speaker|role, session) or a string."""
    if isinstance(x, Record):
        return x
    if isinstance(x, str):
        return Record(text=x)
    if isinstance(x, Mapping):
        text = x.get("text", x.get("content"))
        if text is None:
            raise RecordError("a record mapping needs 'text' or 'content'")
        known = {"text", "content", "id", "time", "timestamp", "speaker", "role", "session", "slot", "order",
                 "status", "ref", "kind", "sort_key", "meta"}
        meta = dict(x.get("meta") or {})
        meta.update({k: str(v) for k, v in x.items() if k not in known})
        return Record(text=str(text), id=x.get("id"), time=x.get("time", x.get("timestamp")),
                      speaker=x.get("speaker", x.get("role")), session=x.get("session"), slot=x.get("slot"),
                      order=x.get("order"), status=x.get("status", "accepted"), ref=x.get("ref"),
                      kind=x.get("kind", "text"), sort_key=x.get("sort_key"), meta=meta)
    raise RecordError("cannot read a record from %s" % type(x).__name__)


def line(rec: Record, template: str = DEFAULT_TEMPLATE) -> str:
    """One record as one line. Fields that are absent drop together with their brackets and separators."""
    if template == DEFAULT_TEMPLATE:
        head = []
        if rec.id:
            head.append(rec.id)
        if rec.time:
            head.append("[%s]" % rec.time)
        h = " ".join(head)
        if rec.speaker:
            h = (h + " " if h else "") + rec.speaker + ":"
        return (h + " " if h else "") + rec.text
    vals = {"id": rec.id, "time": rec.time, "speaker": rec.speaker, "session": rec.session, "text": rec.text}
    out = template
    for k, v in vals.items():
        if v is None or v == "":
            out = re.sub(r"\[?\{%s\}\]?:?\s?" % k, "", out)
        else:
            out = out.replace("{%s}" % k, str(v))
    return out


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def serialize(records: Iterable[Record], *, template: str = DEFAULT_TEMPLATE,
              normalize: bool = True) -> Tuple[str, list]:
    """(store text, [(start, end) character span of each record])."""
    parts, spans, pos = [], [], 0
    for i, r in enumerate(records):
        s = line(r, template)
        if normalize:
            s = normalize_newlines(s)
        if i:
            pos += len(RECORD_SEP)
        spans.append((pos, pos + len(s)))
        parts.append(s)
        pos += len(s)
    return RECORD_SEP.join(parts), spans


def numbered(texts: Iterable[str], *, start: int = 1) -> list:
    """Records "1. <text>", "2. <text>", ...: a numbered log, where a later serial updates an earlier one."""
    out = []
    for n, t in enumerate(texts, start):
        if "\n" in t or "\r" in t:
            raise RecordError("a numbered record must be one line: %r" % t[:60])
        out.append(Record(text="%d. %s" % (n, t), order=float(n)))
    return out
