"""Rendering: the selected units as the text a reader receives.

  r4        one block per parent: units of a block in id order, blocks in order of their first unit,
            "Memory <n>:" before each block (harness/render_rule.py:18-21). The paper's render.
  flat      units in id order, one per line (harness/lme_pf29.py:764-765).
  numbered  "[<serial>] <unit>" in walk order, stable by serial (lme_pf29.py:3630-3632, without the tag); a log
            numbered under its own spelling is shown as written, "<prefix><unit>".
"""
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from . import bpe


@dataclass(frozen=True)
class Block:
    index: int
    parent: object
    ids: Tuple[int, ...]
    text: str


@dataclass(frozen=True)
class Context:
    text: str
    style: str
    ids: Tuple[int, ...]
    blocks: Tuple[Block, ...] = ()
    mode: str = "native"
    memory_version: Optional[int] = None
    meta: dict = field(default_factory=dict)

    @property
    def ctxhash(self):
        """sha1 of the UTF-8 text, first 12 hex digits, as every bundle records it."""
        return hashlib.sha1(self.text.encode()).hexdigest()[:12]

    @property
    def tok(self):
        return bpe.count(bpe.o200k(), self.text)

    def messages(self, question, *, system=None, template="{context}\n\nQuestion: {question}"):
        out = [{"role": "system", "content": system}] if system else []
        out.append({"role": "user", "content": template.format(context=self.text, question=question)})
        return out

    def __str__(self):
        return self.text


def per_parent_groups(ids: Sequence[int], parent: Sequence) -> list:
    g = {}
    for i in sorted(ids):
        g.setdefault(parent[i], []).append(i)
    return sorted(g.values(), key=lambda b: b[0])


def r4(ids, units, parent) -> Context:
    groups = per_parent_groups(ids, parent)
    blocks = tuple(Block(n + 1, parent[b[0]], tuple(b), "\n".join(units[i] for i in b)) for n, b in enumerate(groups))
    text = "\n".join("Memory %d:\n%s" % (b.index, b.text) for b in blocks)
    return Context(text, "r4", tuple(i for b in groups for i in b), blocks)


def flat(ids, units) -> Context:
    s = sorted(ids)
    return Context("\n".join(units[i] for i in s), "flat", tuple(s))


def numbered(walk_ids, units, order, prefixes=None) -> Context:
    w = sorted(walk_ids, key=lambda i: order[i])          # stable: ties keep walk order
    if prefixes is not None:
        return Context("\n".join(prefixes[i] + units[i] for i in w), "numbered", tuple(w))
    return Context("\n".join("[%s] %s" % (order[i], units[i]) for i in w), "numbered", tuple(w))


def compose(*contexts: Context, sep: str = "\n\n") -> Context:
    """Stores read separately, joined in order; block headers renumbered across them."""
    blocks, parts, n = [], [], 0
    for c in contexts:
        if c.style == "r4" and c.blocks:
            for b in c.blocks:
                n += 1
                blocks.append(Block(n, b.parent, b.ids, b.text))
            parts.append("\n".join("Memory %d:\n%s" % (b.index, b.text) for b in blocks[-len(c.blocks):]))
        else:
            parts.append(c.text)
    return Context(sep.join(parts), "composed", tuple(i for c in contexts for i in c.ids), tuple(blocks))
