"""Recover the selected unit ids from a rendered harness context (harness/render_parent.py:13-43,
harness/rerender_nomerge.py:53-55, harness/render_rule.py:28-29), so R4 can be rendered from what a published run fed.
"""
import collections
import re

HDR = re.compile(r"(?m)^Memory \d+:\n")


def index80(units):
    idx = {}
    for i, s in enumerate(units):
        idx.setdefault(s[:80], []).append(i)
    return idx


def blocks_of(context):
    """The block bodies of a "Memory n:" context, joiners stripped (render_rule.py:28-29)."""
    parts = HDR.split(context)[1:]
    return [b[:-1] if (n < len(parts) - 1 and b.endswith("\n")) else b for n, b in enumerate(parts)]


def decompose(block, units, idx):
    """Unit ids of one block; among same-text units, the first one after the block's previous unit."""
    ids, p = [], 0
    while p < len(block):
        cand = [i for i in idx.get(block[p:p + 80], []) if block.startswith(units[i], p)
                and (p + len(units[i]) == len(block) or block[p + len(units[i])] == "\n")]
        if not cand:
            return None
        L = max(len(units[j]) for j in cand)
        cand = [j for j in cand if len(units[j]) == L]
        after = [j for j in cand if not ids or j > ids[-1]]
        i = min(after) if after else min(cand)
        ids.append(i)
        p += len(units[i])
        if p < len(block):
            p += 1
    return ids


def resolve(dec, units, parent):
    """Where one id was given to two places (a repeated passage), move the extra one to an unused copy of the same
    text, preferably in the parent of its block-mates."""
    used = collections.Counter(i for d in dec for i in d)
    copies = collections.defaultdict(list)
    for j, s in enumerate(units):
        copies[s].append(j)
    for d in dec:
        for p, i in enumerate(d):
            if used[i] < 2:
                continue
            mates = {parent[x] for x in d if x != i}
            free = [j for j in copies[units[i]] if used[j] == 0]
            alt = [j for j in free if mates and parent[j] in mates] or free
            if alt:
                used[i] -= 1
                d[p] = alt[0]
                used[alt[0]] += 1
    return dec


def recover_ids(context, units, parent, idx=None):
    """The ids a harness context holds, or raise ValueError (a header-less context, a failed decomposition)."""
    if not HDR.search(context):
        raise ValueError("context has no 'Memory n:' header; nothing to recover")
    idx = idx if idx is not None else index80(units)
    dec = [decompose(b, units, idx) for b in blocks_of(context)]
    if any(d is None for d in dec):
        raise ValueError("unit decomposition failed")
    dec = resolve(dec, units, parent)
    ids = [i for d in dec for i in d]
    if len(set(ids)) != len(ids):
        raise ValueError("a unit id is used twice")
    return ids
