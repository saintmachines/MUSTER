"""Turn common conversation shapes into records."""
from .records import Record


def chat(messages, *, session=None, start=0):
    """[{"role": ..., "content": ..., "time"?: ...}] -> one Record per message, id "S<session>:<n>"."""
    out = []
    for n, m in enumerate(messages, start + 1):
        out.append(Record(text=str(m.get("content", "")), id=("S%s:%d" % (session, n)) if session is not None else None,
                          time=m.get("time") or m.get("timestamp"), speaker=m.get("role") or m.get("speaker"),
                          session=None if session is None else str(session)))
    return out


def numbered_lines(lines, start=1):
    """A numbered log: "1. <line>", "2. <line>", ... joined by newlines (one record per line)."""
    return "\n".join("%d. %s" % (n, ln) for n, ln in enumerate(lines, start))


def locomo(conversation):
    """LoCoMo's own record line, one per turn: "<dia_id> [<session date_time>] <speaker>: <text>", plus
    " [image: <caption>]" when the turn has an image. Sessions in numeric order."""
    import re
    keys = sorted((k for k in conversation if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
    out = []
    for k in keys:
        date = conversation[k + "_date_time"]
        for t in conversation[k]:
            text = t["text"]
            if t.get("blip_caption"):
                text += " [image: %s]" % t["blip_caption"]
            out.append(Record(text=text, id=t["dia_id"], time=date, speaker=t["speaker"], session=k.split("_")[1]))
    return out
