"""The harness's intermediate renders, from which the published R4 contexts were made (exact mode only).

The paper's R4 contexts were produced in two steps: the harness rendered the selected units with an intermediate
grouping (bgroup for prose stores, seg3 for BEAM and LoCoMo), and render_rule.py then recovered the unit ids from that
text and rendered one block per parent. Where a store holds several units with identical text, the recovery can name
another copy than the one selected, and the published context follows the copy. Reproducing a published context byte
for byte therefore needs both steps. Lifted from harness/lme_pf29.py:271-333 (bgroup) and 525-715 (seg3, HMARK off).
"""
import math


def rank_pos(order):
    """First position of each unit in a ranking."""
    pos = {}
    for n, i in enumerate(order):
        pos.setdefault(i, n)
    return pos


def render_bgroup(fed, txt, pos, far, parent, ents, hubs, cap):
    fed = sorted(fed)
    lim = max(1, cap if cap > 0 else 10)
    if len(fed) <= lim:
        return "\n".join("Memory %d:\n%s" % (n + 1, txt(i))
                         for n, i in enumerate(sorted(fed, key=lambda j: (pos.get(j, far), j))))
    E = {i: (ents[i] - hubs) if i < len(ents) else set() for i in fed}
    _ov = [len(E[fed[a]] & E[fed[b]]) for a in range(len(fed)) for b in range(a + 1, len(fed))]
    PB = (max(_ov) if _ov else 0) + 1
    groups = {i: [i] for i in fed}

    def sim(a, b):
        s = 0
        for x in groups[a]:
            for y in groups[b]:
                s += len(E[x] & E[y]) + (PB if parent[x] == parent[y] else 0)
        return s / float(len(groups[a]) * len(groups[b]))

    size_cap = int(math.ceil(len(fed) / float(lim)))
    while len(groups) > lim:
        ks = sorted(groups)
        pick, best = None, -1.0
        for ia in range(len(ks)):
            for ib in range(ia + 1, len(ks)):
                a, b = ks[ia], ks[ib]
                if len(groups[a]) + len(groups[b]) > size_cap:
                    continue
                s = sim(a, b)
                if s > best:
                    pick, best = (a, b), s
        if pick is None:
            size_cap += 1
            continue
        a, b = pick
        groups[a] = groups[a] + groups[b]
        del groups[b]
    out = sorted(groups.values(), key=lambda m: (min(pos.get(i, far) for i in m), m[0]))
    return "\n".join("Memory %d:\n%s" % (n + 1, "\n".join(txt(i) for i in sorted(m))) for n, m in enumerate(out))


def render_seg3(fed, txt, parent, ents, hubs, cap):
    """seg3 (doc_first) with HMARK off: an ordered segmentation of the sorted fed units, merged by priority levels."""
    fed = sorted(fed)
    P = len({parent[i] for i in fed})
    lim = min(len(fed), max(-cap, P)) if cap < 0 else max(1, cap if cap > 0 else 10)
    if len(fed) <= lim:
        return "\n".join("Memory %d:\n%s" % (n + 1, txt(i)) for n, i in enumerate(fed))
    E = {i: (ents[i] - hubs) if i < len(ents) else set() for i in fed}
    D = {}
    for i, p in enumerate(parent):
        if i < len(ents):
            D.setdefault(p, set()).update(ents[i])
    D = {p: (s - hubs) for p, s in D.items()}
    blocks = [[i] for i in fed]

    def pars(b):
        return {parent[i] for i in b}

    def jac(sa, sb):
        u = len(sa | sb)
        return (len(sa & sb) / float(u)) if u else 0.0

    def b1(a, b):
        s = 0
        for x in a:
            for y in b:
                s += len(E[x] & E[y])
        return s / float(len(a) * len(b))

    def b2(a, b):
        return jac(set().union(*[E[i] for i in a]), set().union(*[E[i] for i in b]))

    def b3(a, b):
        sa = set().union(*[D.get(p, set()) for p in pars(a)]) if a else set()
        sb = set().union(*[D.get(p, set()) for p in pars(b)]) if b else set()
        return jac(sa, sb)

    _sing = [(x, y) for ix, x in enumerate(fed) for y in fed[ix + 1:] if parent[x] != parent[y]]
    tau_i = (sum(b1([x], [y]) for x, y in _sing) / len(_sing)) if _sing else 0.0
    _pl = sorted({parent[i] for i in fed}, key=lambda p: min(i for i in fed if parent[i] == p))
    _grp = {p: [i for i in fed if parent[i] == p] for p in _pl}
    _pp = [(a, b) for ia, a in enumerate(_pl) for b in _pl[ia + 1:]]
    tau_b = (sum(b2(_grp[a], _grp[b]) for a, b in _pp) / len(_pp)) if _pp else 0.0
    tau_c = (sum(b3(_grp[a], _grp[b]) for a, b in _pp) / len(_pp)) if _pp else 0.0
    LEVELS = [(1, b1, tau_i, True), (1, b1, 0.0, True), (2, b2, tau_b, False), (2, b2, 0.0, False),
              (3, b3, tau_c, False), (3, b3, 0.0, False), (4, None, None, False)]
    size_cap = int(math.ceil(len(fed) / float(lim)))
    while len(blocks) > lim:
        got = False
        for lv, bond, thr, same_only in LEVELS:
            if lv == 4:
                pool = [j for j in range(len(blocks) - 1) if len(blocks[j]) + len(blocks[j + 1]) <= size_cap]
                if not pool:
                    continue
                k = min(pool, key=lambda j: len(blocks[j]) + len(blocks[j + 1]))
                blocks[k] = blocks[k] + blocks[k + 1]
                del blocks[k + 1]
                got = True
                break
            cand = []
            for j in range(len(blocks) - 1):
                A, B = blocks[j], blocks[j + 1]
                one = (len(pars(A)) == 1 and pars(A) == pars(B))
                if same_only != one:
                    continue
                if not one and len(A) + len(B) > size_cap:
                    continue
                s = bond(A, B)
                if s < thr or (s <= 0 and not (lv == 1 and thr == 0.0)):
                    continue
                cand.append((s, j))
            if not cand:
                continue
            s, j = max(cand, key=lambda t: (t[0], -t[1]))
            blocks[j] = blocks[j] + blocks[j + 1]
            del blocks[j + 1]
            got = True
            break
        if not got:
            size_cap += 1
    return "\n".join("Memory %d:\n%s" % (n + 1, "\n".join(txt(i) for i in m)) for n, m in enumerate(blocks))
