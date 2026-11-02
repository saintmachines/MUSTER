#!/usr/bin/env python3
"""LongMemEval as MemoryAgentBench actually ships it: the Accurate_Retrieval split, not the original file.

Everything measured on LongMemEval so far used lme_data/longmemeval_s, the ORIGINAL 500-question release.
MemoryAgentBench ships something else: five rows of the Accurate_Retrieval split tagged longmemeval_s*,
sixty questions each, three hundred in total, and the sixty questions of a row SHARE ONE 1.6M-character
context instead of carrying their own haystack. The give-away was already in hand and went unread - CAR's
cross-benchmark script reports n=45 for knowledge-update, and this split has 9 per row over 5 rows, while
the original file has 78. The two are not the same question set, so no number computed on the original is
comparable to CAR's or to the leaderboard.

Three consequences follow, and all three are why this file exists rather than a flag on the old one:
  the questions carry a "Current Date:" prefix, which is part of the input and must not be stripped;
  one context per row means five embeddings instead of five hundred, so this is far cheaper and faster;
  the six question types are present in fixed proportion per row (15 multi-session, 15 temporal-reasoning,
  9 knowledge-update, 9 single-session-user, 6 assistant, 6 preference), so per-type reporting is exact.

Retrieval, views and router are unchanged from the previous harness, which keeps the comparison to the
FactConsolidation results honest: units are rounds, the document is the session, the graph runs either
over rounds or over sessions, _full adds BM25 and the +-h session merge, and the router picks the view
from the suppression structure of the extracted facts.
"""
import ast, hashlib, json, math, os, random, re, string, sys, collections
from concurrent.futures import ThreadPoolExecutor

MUSTER_HOME = os.environ.get("MUSTER_HOME", os.path.dirname(os.path.abspath(__file__)))   # release: the harness directory; override with MUSTER_HOME
sys.path.insert(0, MUSTER_HOME)
sys.path.insert(0, os.path.join(MUSTER_HOME, "src"))
import numpy as np, pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


BM25LC = os.environ.get("BM25LC", "") not in ("", "0")
EXPAND = int(os.environ.get("EXPAND", "0"))
SPH = int(os.environ.get("SPH", "0"))
SPHW = int(os.environ.get("SPHW", "7"))
SPHS = int(os.environ.get("SPHS", "3"))
SPHP = int(os.environ.get("SPHP", "0"))
SPHM = os.environ.get("SPHM", "")
RUNL = int(os.environ.get("RUNL", "3"))


# --- HBM BEGIN (hierarchical block BM25: score coarse, select fine, feed coarse) ---
# Block size in units. 0 / unset => every helper below is bypassed and the file behaves exactly
# as it did before this patch.
HBM = int(os.environ.get("HBM", "0"))
# 1 => the reader receives the whole block of a sparse-face unit. RAISES the fed token count to
# the official 10 x 4096 ceiling; see the module docstring of patch_hbm.py.
HBM_FEED = int(os.environ.get("HBM_FEED", "0"))
# 1 => once the blocks are exhausted, backfill `sparse` with flat-BM25 order up to K1 so that
# |sparse| equals the baseline's and every downstream consumer of the sparse TAIL (srest,
# r100s, the flrm tails at [3:13], the gpos_s witness at [:50]) stays on the same scale. The
# FACE - sparse[:f_s], f_s <= 6 - is always inside the pure one-per-block prefix.
HBM_BF = int(os.environ.get("HBM_BF", "1"))


def _hbm_blocks(rtext, parent, B):
    # Partition [0, len(rtext)) into DISJOINT blocks of at most B consecutive units that never
    # cross a `parent` boundary.
    #
    # Returns (mem, blkid, btxt):
    #   mem[b]   = the ordered unit ids of block b
    #   blkid[i] = the block that owns unit i          (a total function on [0, N): a PARTITION)
    #   btxt[b]  = the block's text, joined the same way the reader's context is joined
    #
    # A partition is the whole point. SPH's overlapping windows let neighbouring windows share
    # members, so the head of the sparse list degenerated to one region; here two distinct blocks
    # share no unit by construction, so k blocks are k distinct regions.
    #
    N = len(rtext)
    mem, btxt = [], []
    b0 = 0
    while b0 < N:
        p = parent[b0]
        m = []
        j = b0
        while j < N and len(m) < B and parent[j] == p:
            m.append(j)
            j += 1
        mem.append(m)
        btxt.append("\n".join(rtext[x] for x in m))
        b0 = j
    blkid = {}
    for b, m in enumerate(mem):
        for i in m:
            blkid[i] = b
    # a partition: every unit owned exactly once, nothing invented, no boundary crossed
    assert len(blkid) == N and sum(len(m) for m in mem) == N
    for b, m in enumerate(mem):
        assert m == list(range(m[0], m[0] + len(m))) and len(m) <= B
        assert len({parent[i] for i in m}) == 1
    return mem, blkid, btxt


def _hbm_champions(mem, blkid, bs, bsw, K1, backfill, N):
    # Top blocks by block-BM25 (`bsw`), each contributing its own best-child by unit-BM25 (`bs`).
    #
    # Returns (sparse, npure). `npure` = how many leading entries came from blocks, i.e. the
    # prefix over which "at most one unit per block" is guaranteed and asserted.
    #
    # Ties: blocks break on block id, children on the SMALLEST unit id - the same direction the
    # baseline's stable `sorted(range(N), key=lambda i: -bs[i])` breaks them, so the two selections
    # agree wherever they can.
    #
    order = sorted(range(len(mem)), key=lambda b: (-bsw[b], b))
    sparse, seen = [], set()
    for b in order:
        ch = max(mem[b], key=lambda i: (bs[i], -i))
        if ch not in seen:
            sparse.append(ch)
            seen.add(ch)
        if len(sparse) >= K1:
            break
    npure = len(sparse)
    # THE DIVERSITY INVARIANT, checked rather than argued: no block contributes twice.
    assert len({blkid[i] for i in sparse}) == npure, "HBM: two units selected from one block"
    if backfill:
        for i in sorted(range(N), key=lambda x: -bs[x]):
            if len(sparse) >= K1:
                break
            if i not in seen:
                sparse.append(i)
                seen.add(i)
    return sparse[:K1], npure


def _hbm_expand(idxs, sface, of_map, on):
    # Replace each SPARSE-FACE unit by its whole block, order-preserving and deduped.
    #
    # `sface` is the arm's own reserved sparse-floor list - exactly the membership `floor_fill`
    # uses ("a unit found by several branches counts toward every one of them"), so which units
    # count as sparse is not re-decided here. Identity when `on` is false or `sface` is empty,
    # which is what makes non-flr arms and HBM_FEED=0 byte-identical.
    #
    if not on or not sface:
        return list(idxs)
    out = []
    for i in idxs:
        for j in (of_map.get(i, (i,)) if i in sface else (i,)):
            if j not in out:
                out.append(j)
    return out
# --- HBM END ---


QSTRIP = os.environ.get("QSTRIP", "") not in ("", "0")


def _qstrip(t):
    """Retrieval-side query cleaner (rule R1): remove a structurally-identified answer-candidate
    enumeration block. No-op when the marker is absent, and no-op unless QSTRIP=1. The reader
    prompt is never touched."""
    if not QSTRIP:
        return t
    return re.sub(r"Below is a list of possible[^\[]*\[.*?\]", " ", t, flags=re.S).strip()


# BM25STEM=1 (2026-10-14, swap item 5): Porter stemming (original algorithm, as SQLite FTS5 porter) on every BM25 token,
# documents and queries. Unset = the tokens below, unchanged.
BM25STEM = os.environ.get("BM25STEM", "") not in ("", "0")
_QPK = {"prompt_name": os.environ["EMB_QPROMPT"]} if os.environ.get("EMB_QPROMPT") else {}   # swap item 4: query encode only
if BM25STEM:
    import functools as _ft
    from nltk.stem.porter import PorterStemmer as _PS
    _PSTEM = _ft.lru_cache(maxsize=None)(_PS(mode=_PS.ORIGINAL_ALGORITHM).stem)


def _btok(t):
    """BM25 tokenizer. Default = the file's historical raw split (case-sensitive) so existing
    runs stay byte-comparable; BM25LC=1 switches to standard lowercase+punct-strip."""
    if not BM25LC:
        return [_PSTEM(w) for w in t.split()] if BM25STEM else t.split()
    t = t.lower()
    t = "".join(c if (c.isalnum() or c.isspace()) else " " for c in t)
    return [_PSTEM(w) for w in t.split()] if BM25STEM else t.split()


# --- DATEANN BEGIN (deterministic date-delta annotation; $0; reader-side only) ---
import datetime as _dt

DATEANN = os.environ.get("DATEANN", "") not in ("", "0")
# NULLCTX: the reader gets an empty document block. The measurement is how much of this
# cell's score survives with no retrieved text at all - on EventQA the gold is inside the
# question on 100/100 and inside the book on 2/100, so that residue may be most of it.
NULLCTX = os.environ.get("NULLCTX", "") not in ("", "0")
# T8.  Adds the two missing corners of the view 2x2 - prune_norefill (substitution without
# the refill) and slotdiv (the refill without the substitution).  slotdiv needs no store
# bit, no extractor and no profile, so it is defined on every store.  Off by default, so
# this file reproduces lme_pf12 byte for byte when NOREFILL is unset.
NOREFILL = os.environ.get("NOREFILL", "") not in ("", "0")
_DA_N = [0]      # lines annotated for the question currently being assembled

# the question's own clock, e.g. "Current Date: 2023/05/27 (Sat) 22:54,"
_DA_QRE = re.compile(r"Current Date:\s*(\d{4})/(\d{2})/(\d{2})")
# a fed unit's own stamp at line start, e.g. "[Chat Time: 2023/03/04 (Sat) 10:12] ..." possibly
# behind the view's "[CURRENT] " marker. The date must sit inside the FIRST bracket of the line.
_DA_URE = re.compile(r"^(\[CURRENT\]\s*)?\[([^\]\n]*?)(\d{4})/(\d{2})/(\d{2})([^\]\n]*)\]")


def _qdate(qtext):
    """The question's Current Date as a date, or None. None disables annotation for that question
    rather than guessing - an unparsable clock must never produce a wrong delta."""
    m = _DA_QRE.search(qtext or "")
    if not m:
        return None
    try:
        return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _da_phrase(n):
    """Fixed wording. n = (question date - unit date) in whole days."""
    if n == 0:
        return " | today"
    if n > 0:
        return " | %d days before now" % n
    return " | %d days after now" % (-n)


def _dateann(text, qdate):
    """Annotate every fed unit's own bracket stamp with its distance from the question's clock.

    Pure Python date arithmetic - no model call, no cost, no randomness. Identity when DATEANN is
    off, when the question has no parsable clock, on any line whose first bracket holds no parsable
    date, and on a line already carrying the marker (so re-application cannot compound)."""
    if not DATEANN or qdate is None or not text:
        return text
    out = []
    for ln in text.split("\n"):
        m = _DA_URE.match(ln)
        if not m:
            out.append(ln)
            continue
        head, pre, post = (m.group(1) or ""), m.group(2), m.group(6)
        if " | " in pre or " | " in post:
            out.append(ln)          # already annotated
            continue
        try:
            ud = _dt.date(int(m.group(3)), int(m.group(4)), int(m.group(5)))
        except ValueError:
            out.append(ln)          # e.g. 2023/02/30 - leave the line exactly as it was
            continue
        _DA_N[0] += 1
        out.append("%s[%s%s/%s/%s%s%s]%s"
                   % (head, pre, m.group(3), m.group(4), m.group(5), post,
                      _da_phrase((qdate - ud).days), ln[m.end():]))
    return "\n".join(out)
# --- DATEANN END ---
from real_e1_conflict_killer import multihop, entities, build_entity_index
from real_e1_lme import BGE
import core2 as MC
import unified_router as UR
from budget_guard import Guard
import tiktoken

ENC = tiktoken.encoding_for_model("gpt-4o-mini")
OFFICIAL_CAP = 10 * 4096

RENDER = os.environ.get("RENDER", "flat")
assert RENDER in ("flat", "mab", "group", "bgroup", "hier", "hseg", "seg3"), "RENDER=%s" % RENDER
HMARK = os.environ.get("HMARK", "off")          # sub-block markers inside a block
assert HMARK in ("off", "sub", "span", "both")
HBOND = os.environ.get("HBOND", "doc")          # stage-2 block identity: doc | block
assert HBOND in ("doc", "block", "adj")
RGROUP_MAX = int(os.environ.get("RGROUP_MAX", "0") or 0)     # 0 = U


def _rank_pos(order):
    """First position of each unit in a ranking; missing units sort after every ranked one."""
    pos = {}
    for n, i in enumerate(order):
        pos.setdefault(i, n)
    return pos


def _render_bgroup(fed, txt, pos, far, parent, ents, hubs, cap):
    """Average-linkage agglomeration under a size cap. Deterministic, no chaining."""
    import math
    fed = sorted(fed)
    lim = max(1, cap if cap > 0 else 10)
    if len(fed) <= lim:
        return "\n".join("Memory %d:\n%s" % (n + 1, txt(i))
                          for n, i in enumerate(sorted(fed, key=lambda j: (pos.get(j, far), j))))
    E = {i: (ents[i] - hubs) if i < len(ents) else set() for i in fed}
    # A shared parent outranks any lexical bond PRESENT IN THIS QUESTION -- an ordering, not a
    # magnitude, so the number is read from the graph rather than chosen.
    _ov = [len(E[fed[a]] & E[fed[b]]) for a in range(len(fed)) for b in range(a + 1, len(fed))]
    PB = (max(_ov) if _ov else 0) + 1
    groups = {i: [i] for i in fed}

    def sim(a, b):
        """Mean pairwise bond over the ORIGINAL members -- growth alone never raises this."""
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
        if pick is None:                    # nothing legal left: the split cannot be this even
            size_cap += 1
            continue
        a, b = pick
        groups[a] = groups[a] + groups[b]
        del groups[b]
    out = sorted(groups.values(), key=lambda m: (min(pos.get(i, far) for i in m), m[0]))
    bond = lambda x, y: len(E[x] & E[y]) + (PB if parent[x] == parent[y] else 0)
    win = [bond(m[a], m[b]) for m in out for a in range(len(m)) for b in range(a + 1, len(m))]
    btw = [bond(x, y) for gi in range(len(out)) for gj in range(gi + 1, len(out))
           for x in out[gi] for y in out[gj]]
    _RG.clear()
    _RG.update(sizes=sorted((len(m) for m in out), reverse=True), forced=0, ovl=[],
               nopos=sum(1 for i in fed if i not in pos), nfed=len(fed), ngroup=len(out),
               cap=size_cap, mode="bgroup",
               pure=sum(1 for m in out if len({parent[i] for i in m}) == 1),
               within=(sum(win) / len(win)) if win else 0.0,
               between=(sum(btw) / len(btw)) if btw else 0.0, pb=PB)
    return "\n".join("Memory %d:\n%s" % (n + 1, "\n".join(txt(i) for i in sorted(m)))
                      for n, m in enumerate(out))


def _render_hier(fed, txt, pos, far, parent, ents, hubs, cap):
    """Hierarchical blocks, source order preserved inside each document.

    Three priorities, in this order, each gated by ITS OWN background level:
      1. WITHIN a document, merge only blocks ADJACENT among that document's selected units,
         strongest UNIT-level content bond first. Weak boundaries dissolve, strong ones survive.
      2. ACROSS documents, merge whole blocks by DOCUMENT-level relatedness.
      3. A CHAIN MAY STOP: a merge below its phase's background level is refused while any
         better merge exists. The block COUNT is the hyperparameter and is always reached.

    2026-09-14, two defects fixed after the b40 null (F1 0.4125 -> 0.4106, p=0.92):

    (a) PHASE 2 WAS NEVER IMPLEMENTED. One bond() served both phases and it was always the
        UNIT-level entity intersection, so "document-level relatedness" was unit-level
        relatedness wearing its name. At chunk 4096 a unit is a large slice of its document and
        the unit set proxied the document set; at chunk 1024 the proxy dies -- the background
        fell 0.699 -> 0.130 and 48% of merges were made below it, i.e. arbitrarily. Phase 2 now
        scores the UNION of entities over ALL of a document's units in the store, normalised, and
        is gated by its own background TAU_C. ents and parent arrive as the FULL per-unit
        tables (:2065, :3445), so this costs one pass and needs no new plumbing.

    (b) THE BLOCK COUNT WAS PINNED AT 10 while the fed units span a MEDIAN OF 15 DOCUMENTS at
        the b40 configuration. Fifteen documents cannot occupy ten blocks without mixing, so
        phase 1 was impossible before it began: DRY measured 5.52/10 single-document blocks
        against 9.08/10 at 4096xU=20, the one setting where this renderer matched flat.
        A NEGATIVE cap is now adaptive: lim = min(len(fed), max(-cap, documents among fed)).
        DRY at the b40 cell: lim=10 -> 55% pure, lim=16 -> 85%, lim=20 -> 93%.
    """
    import math
    fed = sorted(fed)
    P = len({parent[i] for i in fed})
    if cap < 0:
        lim = min(len(fed), max(-cap, P))
    else:
        lim = max(1, cap if cap > 0 else 10)
    if len(fed) <= lim:
        return "\n".join("Memory %d:\n%s" % (n + 1, txt(i)) for n, i in enumerate(fed))

    E = {i: (ents[i] - hubs) if i < len(ents) else set() for i in fed}

    # Document-level signatures: every unit of the document, not only the fed ones. This is the
    # object phase 2 is supposed to compare and the thing the old code never built.
    D = {}
    for i, p in enumerate(parent):
        if i < len(ents):
            D.setdefault(p, set()).update(ents[i])
    D = {p: (s - hubs) for p, s in D.items()}

    blocks = [[i] for i in fed]                      # each block is a list of unit ids, sorted

    def pars(b):
        return {parent[i] for i in b}

    def bond_i(a, b):
        """Phase 1: mean pairwise unit-entity overlap. Adjacent slices of one document."""
        s = 0
        for x in a:
            for y in b:
                s += len(E[x] & E[y])
        return s / float(len(a) * len(b))

    def sig(b):
        u = set()
        for p in pars(b):
            u |= D.get(p, set())
        return u

    # Source rank of each parent. `parent` is built in source order (:2036-2039), so first
    # appearance IS the rank; this needs no numeric parse and works on every cell.
    PRANK = {}
    for _p in parent:
        if _p not in PRANK:
            PRANK[_p] = len(PRANK)

    def gap(a, b):
        """Nearest distance in source spans between two blocks. 1 = they touch."""
        ra = {PRANK.get(parent[i], 0) for i in a}
        rb = {PRANK.get(parent[i], 0) for i in b}
        return min(abs(x - y) for x in ra for y in rb) or 1

    def bond_c(a, b):
        """Phase 2: Jaccard between two blocks. HBOND picks what a block IS here.

        `doc` (default) uses the union of the DOCUMENTS' full signatures -- every unit of that
        document in the store. `block` uses only the units actually in the block, i.e. treats the
        block as one big unit, which is what the reader will see. Scale-free either way, so a long
        document is not rewarded for being long the way a raw intersection count would.
        """
        if HBOND == "block":
            sa = set().union(*[E[i] for i in a]) if a else set()
            sb = set().union(*[E[i] for i in b]) if b else set()
        else:
            sa, sb = sig(a), sig(b)
        u = len(sa | sb)
        v = (len(sa & sb) / float(u)) if u else 0.0
        return (v / float(gap(a, b))) if HBOND == "adj" else v

    # Each phase gets its own background, measured on this question's own singletons. One tau
    # cannot gate two different statistics -- that was the coupling that let phase priority leak.
    _ci = [bond_i([x], [y]) for ix, x in enumerate(fed) for y in fed[ix + 1:]
           if parent[x] != parent[y]]
    tau_i = (sum(_ci) / len(_ci)) if _ci else 0.0
    _pl = sorted({parent[i] for i in fed})
    _cc = [bond_c([i for i in fed if parent[i] == a], [i for i in fed if parent[i] == b])
           for ia, a in enumerate(_pl) for b in _pl[ia + 1:]]
    tau_c = (sum(_cc) / len(_cc)) if _cc else 0.0

    size_cap = int(math.ceil(len(fed) / float(lim)))
    n_intra = n_cross = n_relax = n_cgood = 0
    c_dist = []
    p_gap = []

    def candidates(intra, cap_on, tau_on):
        """Eligible (i, j, bond) pairs. intra=True restricts to same-parent ADJACENT blocks."""
        out = []
        thr = tau_i if intra else tau_c
        for a in range(len(blocks)):
            for b in range(a + 1, len(blocks)):
                A, B = blocks[a], blocks[b]
                if cap_on and len(A) + len(B) > size_cap:
                    continue
                pa, pb = pars(A), pars(B)
                same = (len(pa) == 1 and pa == pb)
                if intra != same:
                    continue
                if intra:
                    # adjacent among THIS document's selected units: nothing of the same parent
                    # may sit between the two blocks
                    lo, hi = min(max(A), max(B)), max(min(A), min(B))
                    if any(parent[k] in pa and lo < k < hi for k in fed):
                        continue
                s = bond_i(A, B) if intra else bond_c(A, B)
                if tau_on and s < thr:
                    continue
                out.append((s, a, b))
        return out

    while len(blocks) > lim:
        got = False
        # STRICT phase priority: a document is exhausted before any cross-document merge, and tau
        # is relaxed WITHIN a phase, never by falling through to the next one. Phase 1 carries NO
        # uniform size cap -- a document may become one block or be split where its content
        # changes; the block COUNT is the hyperparameter, the block SIZE is not.
        for intra, tau_on, cap_on, which in ((True, True, False, 'i'), (True, False, False, 'i'),
                                             (False, True, True, 'c'), (False, False, True, 'c'),
                                             (False, False, False, 'c')):
            cs = candidates(intra, cap_on, tau_on)
            if not cs:
                continue
            s, a, b = max(cs, key=lambda t: (t[0], -min(pos.get(x, far) for x in blocks[t[1]])))
            # distance BEFORE the merge: blocks[b] is deleted and min(blocks[a]) moves
            _d = abs(min(blocks[a]) - min(blocks[b]))
            _g = gap(blocks[a], blocks[b]) if which == 'c' else 0
            blocks[a] = sorted(blocks[a] + blocks[b])
            del blocks[b]
            if which == 'i':
                n_intra += 1
            else:
                n_cross += 1
                c_dist.append(_d)
                p_gap.append(_g)
                if s >= tau_c:
                    n_cgood += 1
            if not tau_on:
                n_relax += 1
            got = True
            break
        if not got:                                   # nothing legal at any relaxation
            size_cap += 1
            n_relax += 1

    out = sorted(blocks, key=lambda m: (min(m), min(pos.get(i, far) for i in m)))
    _RG.clear()
    # DRY audit: the emitted block membership, so the GLOBAL reading order can be
    # checked from outside. Stage-2 cross merges can interleave blocks, which the
    # min()-sort does not fix.
    _RG.update(memb=[list(m) for m in out],
               sizes=sorted((len(m) for m in out), reverse=True), forced=0, ovl=[],
               nopos=sum(1 for i in fed if i not in pos), nfed=len(fed), ngroup=len(out),
               cap=size_cap, mode="hier", tau=tau_i, tau_c=tau_c, npar=P, lim=lim,
               intra=n_intra, cross=n_cross, cgood=n_cgood, relax=n_relax, hbond=HBOND,
               cdist=(sum(c_dist) / float(len(c_dist)) / float(max(len(parent), 1))) if c_dist else 0.0,
               pgap=(sum(p_gap) / float(len(p_gap))) if p_gap else 0.0,
               pure=sum(1 for m in out if len({parent[i] for i in m}) == 1),
               within=0.0, between=0.0, pb=0)
    return "\n".join("Memory %d:\n%s" % (n + 1, "\n".join(txt(i) for i in m))
                      for n, m in enumerate(out))


def _render_hseg(fed, txt, pos, far, parent, ents, hubs, cap, doc_first=False):
    """The design as stated, with the global reading order GUARANTEED.

    TWO RENDERERS SHARE THIS BODY, and `doc_first` is the only difference between them.

      RENDER=hseg   doc_first=False   the object every hseg leg before 2026-09-17 measured.
      RENDER=seg3   doc_first=True    the design as specified. USE THIS ONE.

    `hseg` carries a defect found on 2026-09-17: the guard below read `s <= 0 or s < thr` and so
    refused a level-1 merge whose two blocks shared no entity -- even on the RELAXED rung, whose
    whole purpose is to accept exactly that. Two blocks of ONE document are related by being one
    document; a zero entity overlap means the extractor found no shared surface form, not that the
    halves are unrelated. The refused pair then fell through to levels 2 and 3, i.e. it was merged
    ACROSS documents instead -- the exact inversion of the stated priority "same document first".

    Measured (free DRY, n=100 each, 1024-token chunks):

                            within-doc merges available / taken       cross merges   pure blocks
      hier v2   U=40 adapt              23.32 / 23.21  (99.5%)            0.00       16.79/16.79
      hseg      U=40 adapt              23.32 /  9.08  (38.9%)           14.13        9.56/16.79
      seg3      U=40 adapt              23.32 / 23.21  (99.5%)            0.00       16.79/16.79
      hseg      U=80 adapt              59.04 / 27.74  (47.0%)           31.20        9.57/20.08
      seg3      U=80 adapt              59.04 / 58.94  (99.8%)            0.00       20.08/20.08
      hseg      lim=10                  23.32 /  9.04  (38.8%)           20.96        2.29/10
      seg3      lim=10                  23.32 / 23.21  (99.5%)            6.79        5.52/10

    So `hseg` ran at ~39-47% of the intended within-document merging at EVERY block count, and the
    legs h5 h9 c5 c9 a5 a9 ($83.16) all carry it. They are kept reproducible under RENDER=hseg
    rather than deleted, because a wrong number whose cause is known is worth more than a gap.

    WHERE THE LADDER ACTUALLY LIVES. When lim >= P the level-1 merges alone reach the block count
    (available = nfed - P >= nfed - lim = needed), so levels 2, 3 and 4 never fire and `seg3`
    collapses onto `hier` -- 90/100 identical partitions at U=40, 91/100 at U=80. Only lim < P
    exercises the ladder, and there `seg3` and `hier` agree on just 12/100: they spend the SAME
    6.79 cross merges, but `hier` picks any pair and inverts the reading order on 77% of questions
    while `seg3` merges only neighbours and inverts on 0%.

    The author's specification: at equal priority keep the in-document order, then block relevance, then
    document-level relevance, and a merge may stop midway; the number of groups is a hyperparameter, and
    grouping keeps the order and respects relevance under it.
    Three priority levels, order preserved at EVERY level, and the block count is a hyperparameter
    that is always reached.

    `_render_hier` implemented levels 1 and 3 and then offered level 2 as an ALTERNATIVE to level 3
    under HBOND, which is not what was asked: they are sequential priorities, not a menu. And its
    emission sorted blocks by min(), which only orders the output when blocks do not interleave --
    stage-2 cross merges interleave them, so 77 of 100 questions were read out of order, 69.6
    inverted pairs on average (measured independently and reproduced here exactly).

    The fix is structural rather than a better score. `blocks` is kept as an ORDERED SEGMENTATION of
    the sorted fed sequence and only CONSECUTIVE segments may merge. A segmentation that only ever
    merges neighbours stays a segmentation, so the concatenated output is the fed units in source
    order, always, by construction -- 0 inversions rather than fewer inversions.

    Nothing is lost at level 1 by that restriction. Parents are contiguous spans of the source
    (`:2036-2039` assigns `m%d % (_cum // MSCALE)` walking the text), verified on every cell used
    here: `par == sorted(par)`. So two blocks of one parent that are adjacent among that parent's
    selected units are already neighbours in the global list.

    The ladder, each level gated by its own background and each falling back only within itself:

      1  same parent          bond over UNIT entity sets           >= tau_i
      1' same parent          same                                 > 0          (never on 0 overlap)
      2  different parents    bond over the BLOCKS' OWN content    >= tau_b
      2' different parents    same                                 > 0
      3  different parents    bond over the DOCUMENTS' signatures  >= tau_c
      3' different parents    same                                 > 0
      4  anything consecutive                                      unconditional

    Level 4 exists because the block COUNT is the hyperparameter: when every bond is zero the count
    still has to be met, and that decision is made here, on the current blocks, rather than smuggled
    into level 1 by relaxing its threshold on a whole parent. `n_forced` reports how often it fired;
    that number is part of the result, not a footnote.
    """
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

    blocks = [[i] for i in fed]                 # ordered segmentation, blocks[k] < blocks[k+1]

    def pars(b):
        return {parent[i] for i in b}

    def jac(sa, sb):
        u = len(sa | sb)
        return (len(sa & sb) / float(u)) if u else 0.0

    def b1(a, b):                                # level 1: unit entities, mean pairwise overlap
        s = 0
        for x in a:
            for y in b:
                s += len(E[x] & E[y])
        return s / float(len(a) * len(b))

    def b2(a, b):                                # level 2: the block's own content
        return jac(set().union(*[E[i] for i in a]), set().union(*[E[i] for i in b]))

    def b3(a, b):                                # level 3: the documents' signatures
        sa = set().union(*[D.get(p, set()) for p in pars(a)]) if a else set()
        sb = set().union(*[D.get(p, set()) for p in pars(b)]) if b else set()
        return jac(sa, sb)

    # Backgrounds, each measured with its own statistic on this question's own material.
    _sing = [(x, y) for ix, x in enumerate(fed) for y in fed[ix + 1:] if parent[x] != parent[y]]
    tau_i = (sum(b1([x], [y]) for x, y in _sing) / len(_sing)) if _sing else 0.0
    _pl = sorted({parent[i] for i in fed}, key=lambda p: min(i for i in fed if parent[i] == p))
    _grp = {p: [i for i in fed if parent[i] == p] for p in _pl}
    _pp = [(a, b) for ia, a in enumerate(_pl) for b in _pl[ia + 1:]]
    tau_b = (sum(b2(_grp[a], _grp[b]) for a, b in _pp) / len(_pp)) if _pp else 0.0
    tau_c = (sum(b3(_grp[a], _grp[b]) for a, b in _pp) / len(_pp)) if _pp else 0.0

    LEVELS = [(1, b1, tau_i, True), (1, b1, 0.0, True),
              (2, b2, tau_b, False), (2, b2, 0.0, False),
              (3, b3, tau_c, False), (3, b3, 0.0, False),
              (4, None, None, False)]
    n_by = {1: 0, 2: 0, 3: 0, 4: 0}
    n_relax = 0
    stage1 = None                                   # partition when level 1 exhausts
    import math as _m
    size_cap = int(_m.ceil(len(fed) / float(lim)))   # cross levels only

    while len(blocks) > lim:
        got = False
        for lv, bond, thr, same_only in LEVELS:
            if lv == 4:                          # the count is the hyperparameter: meet it
                pool = [j for j in range(len(blocks) - 1)
                        if len(blocks[j]) + len(blocks[j + 1]) <= size_cap]
                if not pool:
                    continue
                k = min(pool, key=lambda j: len(blocks[j]) + len(blocks[j + 1]))
                if stage1 is None:            # BEFORE the merge: this one ends stage 1
                    stage1 = [list(b) for b in blocks]
                blocks[k] = blocks[k] + blocks[k + 1]
                del blocks[k + 1]
                n_by[4] += 1
                got = True
                break
            cand = []
            for j in range(len(blocks) - 1):
                A, B = blocks[j], blocks[j + 1]
                one = (len(pars(A)) == 1 and pars(A) == pars(B))
                if same_only != one:
                    continue
                # levels 2 and 3 join blocks of DIFFERENT documents; that is where one block can
                # run away, so that is where the cap belongs. Level 1 stays free.
                if not one and len(A) + len(B) > size_cap:
                    continue
                s = bond(A, B)
                if s < thr or (s <= 0 and not (doc_first and lv == 1 and thr == 0.0)):
                    continue                     # zero overlap across documents is never "related";
                                                 # WITHIN one document it is, so RENDER=seg3 admits it
                cand.append((s, j))
            if not cand:
                continue
            s, j = max(cand, key=lambda t: (t[0], -t[1]))
            if lv != 1 and stage1 is None:    # BEFORE the merge: this one ends stage 1
                stage1 = [list(b) for b in blocks]
            blocks[j] = blocks[j] + blocks[j + 1]
            del blocks[j + 1]
            n_by[lv] += 1
            if thr == 0.0:
                n_relax += 1
            got = True
            break
        if not got:                              # nothing legal anywhere: widen by one
            size_cap += 1
            n_relax += 1

    _RG.clear()
    _RG.update(memb=[list(m) for m in blocks],
               sizes=sorted((len(m) for m in blocks), reverse=True), forced=n_by[4], ovl=[],
               nopos=sum(1 for i in fed if i not in pos), nfed=len(fed), ngroup=len(blocks),
               cap=size_cap, mode=("seg3" if doc_first else "hseg"), npar=P, lim=lim,
               tau=tau_i, tau_b=tau_b, tau_c=tau_c,
               lv1=n_by[1], lv2=n_by[2], lv3=n_by[3], lv4=n_by[4], relax=n_relax,
               intra=n_by[1], cross=n_by[2] + n_by[3] + n_by[4], cgood=n_by[2] + n_by[3],
               cdist=0.0, pgap=0.0, hbond="hseg",
               pure=sum(1 for m in blocks if len({parent[i] for i in m}) == 1),
               within=0.0, between=0.0, pb=0)
    if HMARK == "off" or not stage1:
        return "\n".join("Memory %d:\n%s" % (n + 1, "\n".join(txt(i) for i in m))
                          for n, m in enumerate(blocks))
    # which stage-1 part each unit belonged to; both partitions are contiguous, so a final block
    # splits cleanly along them and no unit is moved, duplicated or dropped.
    s1of = {}
    for si, sb in enumerate(stage1):
        for u in sb:
            s1of[u] = si
    # source-order rank of every span in the STORE, and how many there are, so the marker locates
    # the piece in the novel instead of naming an internal id.
    _prank, _ptot = {}, 0
    for _p in parent:
        if _p not in _prank:
            _prank[_p] = len(_prank)
    _ptot = len(_prank)
    out = []
    for n, m in enumerate(blocks):
        parts, cur, key = [], [], None
        for u in m:
            k = s1of.get(u, -1)
            if cur and k != key:
                parts.append(cur); cur = []
            cur.append(u); key = k
        if cur:
            parts.append(cur)
        body = []
        for pi, pt in enumerate(parts):
            assert len({parent[i] for i in pt}) == 1, \
                ("a stage-1 part spans two spans", n + 1, pi + 1, sorted({parent[i] for i in pt}))
            _loc = "part %d of %d" % (_prank.get(parent[pt[0]], 0) + 1, max(_ptot, 1))
            if HMARK == "sub":
                tagtxt = "[%d.%d]" % (n + 1, pi + 1)
            elif HMARK == "span":
                tagtxt = "[%s]" % _loc
            else:
                tagtxt = "[%d.%d | %s]" % (n + 1, pi + 1, _loc)
            body.append(tagtxt + "\n" + "\n".join(txt(i) for i in pt))
        out.append("Memory %d:\n%s" % (n + 1, "\n".join(body)))
    _RG["nsub"] = sum(1 for m in blocks for _ in [0]) and sum(
        len(set(s1of.get(u, -1) for u in m)) for m in blocks)
    return "\n".join(out)


def _render_units(fed, rtext, cur, order, parent, ents, hubs, cap):
    """The context string for one arm's fed units, under RENDER.

    `flat` is byte-for-byte what this file did before RENDER existed, so it is the default and no
    published number moves. The other two exist to separate the marker, the order and the grouping.
    """
    txt = lambda i: (("[CURRENT] " + rtext[i]) if i in cur else rtext[i])
    if RENDER == "flat":
        return "\n".join(txt(i) for i in sorted(fed))
    pos = _rank_pos(order)
    far = len(pos) + 1
    if RENDER == "mab":
        return "\n".join("Memory %d:\n%s" % (n + 1, txt(i))
                          for n, i in enumerate(sorted(fed, key=lambda j: (pos.get(j, far), j))))
    if RENDER == "bgroup":
        return _render_bgroup(fed, txt, pos, far, parent, ents, hubs, cap)
    if RENDER == "hier":
        return _render_hier(fed, txt, pos, far, parent, ents, hubs, cap)
    if RENDER == "hseg":
        return _render_hseg(fed, txt, pos, far, parent, ents, hubs, cap, doc_first=False)
    if RENDER == "seg3":
        return _render_hseg(fed, txt, pos, far, parent, ents, hubs, cap, doc_first=True)
    # ---- group ---------------------------------------------------------------------------------
    fed = sorted(fed)
    root = {i: i for i in fed}

    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]; x = root[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            root[max(ra, rb)] = min(ra, rb)

    for a in range(len(fed)):                      # a shared parent is the strongest bond we have
        for b in range(a + 1, len(fed)):
            if parent[fed[a]] == parent[fed[b]]:
                union(fed[a], fed[b])
    groups = {}
    for i in fed:
        groups.setdefault(find(i), []).append(i)
    lim = cap if cap > 0 else 10
    _forced, _ovl = [0], []
    if len(groups) > lim:
        # merge the pair sharing the most non-hub entities, ties by the better rank, until it fits
        key = {g: (set().union(*[ents[i] for i in m]) - hubs) for g, m in groups.items()}
        best = {g: min(pos.get(i, far) for i in m) for g, m in groups.items()}
        while len(groups) > lim:
            gs = sorted(groups, key=lambda g: best[g])
            pick, sc = None, -1
            for a in range(len(gs)):
                for b in range(a + 1, len(gs)):
                    s = len(key[gs[a]] & key[gs[b]])
                    if s > sc or (s == sc and pick and best[gs[a]] < best[pick[0]]):
                        pick, sc = (gs[a], gs[b]), s
            if pick is None:
                break
            a, b = pick                            # a merge with zero overlap is still forced: the
            _ovl.append(sc)                        # item count is a ceiling, not a preference
            _forced[0] += (1 if sc <= 0 else 0)
            groups[a] += groups[b]
            key[a] |= key[b]; best[a] = min(best[a], best[b])
            del groups[b], key[b], best[b]
    out = sorted(groups.values(), key=lambda m: (min(pos.get(i, far) for i in m), m[0]))
    _RG.clear()
    _RG.update(sizes=sorted((len(m) for m in out), reverse=True),
               forced=_forced[0], ovl=_ovl[:], nopos=sum(1 for i in fed if i not in pos),
               nfed=len(fed), ngroup=len(out))
    return "\n".join("Memory %d:\n%s" % (n + 1, "\n".join(txt(i) for i in sorted(m)))
                      for n, m in enumerate(out))


RENDER_SLACK = int(os.environ.get("RENDER_SLACK", "64"))   # framing only; see the guard below      # MemoryAgentBench: retrieve_num 10 x chunk_size 4096 tokens
CHUNK = int(os.environ.get("CHUNK", "4096"))
# MSCALE: the COARSE chunk size; 0 disables.  Units are cut at CHUNK and their parents are
# the MSCALE-sized chunks containing them - a two-scale hierarchy the existing ladder
# traverses with no new retrieval code.  This is a hierarchical-retrieval mechanism from prior work.
MSCALE = int(os.environ.get("MSCALE", "0") or 0)

# MRGATE. The operator may ADD into budget the protocol leaves unused; it may never DISPLACE one of
# the U units the protocol already grants. Measured 2026-09-07: at CHUNK=4096 the ten units already
# weigh 98-99% of ROW_CAP, and turning the operator on there re-partitions that full budget over
# more documents and loses 8,143 / 8,145 / 10,450 tokens on det / summ / evq -- one _mr_cap trim at
# a unit boundary costs a whole 4,096-token unit, so the loss is exactly two units. At CHUNK=512 the
# same ten units weigh 10-13% and the operator fills the room instead.
# The test is the benchmark's own ceiling, not a constant of ours: ROW_CAP = retrieve_num x
# chunk_size. Off by default so every result already on disk is reproduced byte for byte.
MRGATE = int(os.environ.get("MRGATE", "0") or 0)
_MR_CAPNOW = [0]
_MR_TOKNOW = [None]
HYBRID = int(os.environ.get("HYBRID", "0") or 0)
AUTOSTRUCT = int(os.environ.get("AUTOSTRUCT", "0") or 0)
# FORCE_ORDER (2026-10-07, counterfactual only): treat the stored position as if it were a declared serial on a store
# that has none, so the structural (newest-per-parent) resolver runs where the serial gate would not fire. Off by default.
FORCE_ORDER = int(os.environ.get("FORCE_ORDER", "0") or 0)
assert not MSCALE or MSCALE > CHUNK, ("MSCALE must be COARSER than CHUNK", MSCALE, CHUNK)
# ATOM: what the chunker packs.  sentence (default, reproduces every existing number),
# blank, newline, or auto (the coarsest separator with >= FIT_Q of atoms inside CHUNK).
# Replaces the HARDSPLIT / RECSPLIT / join-patch pile with one knob.  See patch_atom.py.
ATOM = os.environ.get("ATOM", "sentence")
FIT_Q = float(os.environ.get("FIT_Q", "0.98"))
# ATOM1: the unit IS the atom -- no packing, no splitting. The only way to express
# "one record per unit" for a store whose records vary 7..30 tokens, where no single
# chunk_size can do it (>= 30 and < 14 at once). Default OFF: every existing number
# is produced with ATOM1 unset and is unchanged by this file.
ATOM1 = int(os.environ.get("ATOM1", "0") or 0)
# SUPMARK: render a superseded unit as [SUPERSEDED] instead of leaving it bare. Adds the
# view `comp_sup` alongside composite; identical set, order and budget, so the only
# difference is the label. Default OFF -- every number produced so far had it unset.
SUPMARK = int(os.environ.get("SUPMARK", "0") or 0)
assert ATOM in ("sentence", "blank", "newline", "auto"), ("bad ATOM", ATOM)
OFFICIAL_PROMPT = os.environ.get("OFFICIAL_PROMPT", "") not in ("", "0")

PARQ = os.environ.get("PARQ", os.path.expanduser("~/.cache/huggingface/hub/datasets--ai-hyz--MemoryAgentBench/"
                              "snapshots/7ea066982b140a19337e17e60d45d4076e042faf/data/"
                              "Accurate_Retrieval-00000-of-00001.parquet"))
ROWS = [int(x) for x in os.environ.get("ROWS", "17,18,19,20,21").split(",")]
NQ = int(os.environ.get("NQ", "0"))
K1, U, H, SEEDN = int(os.environ.get("K1", "100")), int(os.environ.get("U", "10")), 4, 5
# how many of the top dense units are guaranteed, and how many neighbours each anchor
# attaches. Measured worth of the guarantee: 0.130 on LongMemEval. Same values on both
# benchmarks, and fixed on FactConsolidation before LongMemEval was measured.
M_ANCH = int(os.environ.get("M", "2"))
CAP = int(os.environ.get("CAPN", "1"))
MODEL = os.environ.get("OAMODEL", "gpt-4o-mini-2024-07-18")
TEMP, SEED = float(os.environ.get("TEMP", "0.7")), 42
# The benchmark's own generation_max_length, per cell, transcribed from
# mab_repo/configs/data_conf/**/*.yaml. One number for ten cells truncated 96% of the InfBench
# summaries and 46.5% of the DetectiveQA answers; the first cost 16.4 points and the second 1.4.
GENMAX = {
    "factconsolidation": 10,      # Conflict_Resolution/Factconsolidation_*.yaml
    "icl_banking77": 20, "icl_clinic150": 20, "icl_nlu": 20,
    "icl_trec_coarse": 20, "icl_trec_fine": 20,   # Test_Time_Learning/ICL/ICL_*.yaml
    "eventqa": 40,                # Accurate_Retrieval/EventQA/Eventqa_*.yaml
    "ruler_qa1": 50, "ruler_qa2": 50, "ruler": 50,
    "longmemeval": 50,            # Accurate_Retrieval/LongMemEval/Longmemeval_s*.yaml
    "recsys_redial": 300,         # Test_Time_Learning/Recsys/Recsys_redial_full.yaml
    "infbench_sum": 1200,         # Long_Range_Understanding/InfBench_sum.yaml
    "detective_qa": 2000, "detectiveqa": 2000,   # Long_Range_Understanding/Detective_QA.yaml
    "beam": 1200,                 # BEAM has no generation_max_length; summaries need room, MAXOUT-guard reports truncation
}
MAXOUT_ENV = os.environ.get("MAXOUT", "")
MAXOUT = int(MAXOUT_ENV or "50")        # replaced per row below once the dataset is known


def genmax_for(ds):
    """The benchmark's answer length for this dataset, or 50 with a printed notice.

    An explicit MAXOUT wins so an ablation stays possible, but taking it prints the benchmark's
    value beside it, because the whole defect was a deviation nobody could see.
    """
    key = (ds or "").lower()
    want = None
    for k, v in GENMAX.items():
        if key == k or key.startswith(k):
            want = v
            break
    if MAXOUT_ENV:
        if want is not None and int(MAXOUT_ENV) != want:
            print("    MAXOUT WARNING ds=%s using %s, the benchmark sets %d"
                  % (ds, MAXOUT_ENV, want), flush=True)
            assert os.environ.get("MAXOUT_OK") == "1", (
                "MAXOUT=%s but %s sets %d. A leg that overrides the benchmark's generation cap is "
                "not a render comparison -- it changes what the reader may write. Set MAXOUT_OK=1 "
                "to declare the override deliberate." % (MAXOUT_ENV, ds, want))
        else:
            print("    MAXOUT=%s ds=%s (benchmark %s)"
                  % (MAXOUT_ENV, ds, want if want is not None else "unlisted"), flush=True)
        return int(MAXOUT_ENV)
    if want is None:
        print("    MAXOUT WARNING ds=%s is not in GENMAX, falling back to 50" % ds, flush=True)
        return 50
    print("    MAXOUT=%d ds=%s (the benchmark's generation_max_length)" % (want, ds), flush=True)
    return want

# hop decay for the canonical reach, the value ladder12.py has used since the FactConsolidation runs
DECAY = float(os.environ.get("DECAY", "0.5"))
# ONE reranker budget for the graph branch, at every node level. It was 200 here, 400 for
# FactConsolidation's document level and unbounded for its unit level, with no reason recorded for any
# of the three. The cross-encoder is the cost this bounds, and dense is already bounded at K1.
RCAP = int(os.environ.get("RCAP", "100"))   # budget parity with dense: K1=100 into the same CE
# Expansion degree cap for the reach (0 disables). Binds only on long units; identity on short ones.
RDEG = int(os.environ.get("RDEG", "5"))
OUT = os.environ.get("OUT", "results/LME_OFFICIAL.jsonl")
CAPUSD = float(os.environ.get("CAP_USD", "20.0"))
# retrieval-only pass: no extraction, no router call, no readers, no spend
DRY = os.environ.get("DRY", "") not in ("", "0")
# NO_EXTRACT=1: no read-time extraction and no router call. The reader is the only generative call per
# query, which is what the paper states. _cur and _st are empty, so no [CURRENT] annotation is written.
NO_EXTRACT = os.environ.get("NO_EXTRACT", "") not in ("", "0")

# PAID-RUN SELF-GUARD. The PreToolUse hook reads the Bash command string, so `zsh whatever.sh`
# hides this file's name and slips past it -- four scripts did exactly that on 2026-09-04, and one
# of them lost DRY to a stripped quote and reached the API. A guard that a rename defeats is not a
# guard, so the check lives here, where every launch path must pass: python, shell script, tmux,
# ssh, cron. DRY runs make no model call and are always allowed; a real run needs the user's word.
if not DRY and os.environ.get("APPROVED_BY_USER", "") != "1":
    import sys as _sys
    _sys.stderr.write(
        "REFUSED: this is a PAID run (DRY is not set) and APPROVED_BY_USER=1 is absent.\n"
        "The user's standing instruction is that paid runs need explicit approval first.\n"
        "State the design and a cost estimate built from MEASURED per-question prices, get an\n"
        "explicit yes, then re-run with APPROVED_BY_USER=1 in the environment.\n")
    _sys.exit(3)

# extraction + router only: no readers. The router's evidence is what the arm-independence
# check needs, and it is two calls a question instead of two per arm per view.
ROUTER_ONLY = os.environ.get("ROUTER_ONLY", "") not in ("", "0")
# DUMP_CTX=1 (2026-09-21): also write the rendered context text per arm ("ctx|<arm>") so a reader elsewhere
# (the Colab Qwen notebook) can answer from byte-identical contexts. Off by default: records stay as they were.
DUMP_CTX = os.environ.get("DUMP_CTX", "") not in ("", "0")
# one extraction per arm, over that arm's own units - the deployment computation. Without
# it an arm's marking depends on which other arms are being scored.
PER_ARM = os.environ.get("PER_ARM", "") not in ("", "0")
# The router must see the same evidence whatever is being scored. Restricting `pairs` to a canonical
# window was not enough: the facts inside that window were still extracted over `sel`, the union of the
# selected arms pools, so a larger arm list spread the same 25-line extraction budget over more
# passages and rounds inside the window lost their facts. With ROUTER_CANON the router runs its own
# extraction over the canonical window itself, which is a function of the query and the corpus only.
ROUTER_CANON = os.environ.get("ROUTER_CANON", "1") not in ("", "0")
# A deterministic stand-in for the extractor, for invariant tests only. gpt-4o-mini returned 25 distinct
# system_fingerprints inside a single run, so temperature=0 with a pinned seed still re-rolls the
# extraction on half the questions; arm-independence cannot be MEASURED through that floor, only
# guaranteed by construction and then checked with the LLM removed. The stub reproduces the one
# coupling that matters - a fact budget shared across the whole passage list - so a harness that leaks
# the arm list into the extraction window still fails the test.
EXTRACT_STUB = os.environ.get("EXTRACT_STUB", "") not in ("", "0")
# The extraction is memoised on disk under the hash of (model, system, prompt). This is what makes
# arm-independence hold ON TOP OF a nondeterministic backend rather than merely being unmeasurable
# through it: OpenAI served one run of this harness from 25 distinct system_fingerprints, and seed only
# fixes the sample within a fingerprint, so the same prompt came back different on half the questions.
# Memoised, the extraction is a deterministic function of its prompt; under PER_ARM that prompt is a
# function of a single arm's own pool and under ROUTER_CANON of the canonical window - neither takes
# METHODS as an argument. Cost must be read from a COLD cache; a warm rerun is free and understates it.
EXTRACT_CACHE = os.environ.get("EXTRACT_CACHE", "1") not in ("", "0")
XC_DIR = os.environ.get("XC_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "extract_cache"))
XC_HIT = [0, 0]
STUB_BUDGET = int(os.environ.get("STUB_BUDGET", "25"))
# ablation switch: force the hierarchical merge even on document-level arms, which is what the
# harness did before improvement D and what the -0.067 losses were measured under
FORCE_H = os.environ.get("FORCE_H", "") == "1"
# which member represents a document: best | newest | cand | first
REP_RULE = os.environ.get("REP_RULE", "unified")
# A local reader through the OpenAI-compatible endpoint: OPENAI_BASE_URL points at ollama and the guard
# is told the call is free, rather than given a fake table entry that a later real model could match.
LOCAL = os.environ.get("LOCAL", "") not in ("", "0")
# Qwen reasons unless told not to; the benchmark wants a short exact answer
_NOTHINK = " /no_think" if (LOCAL and not os.environ.get("REASONING_EFFORT")) else ""   # inert on Qwen3.5/ollama; REASONING_EFFORT replaces it
GUARD_NAME = os.environ.get("GUARD", "lme_official_local" if LOCAL else "lme_official")
_THINK = re.compile(r"<think>.*?</think>", re.S)
_BASE = (["ce_only", "union", "union_sess", "prio", "prio_sess", "bal_anchor", "attprio",
          "attprio_sess", "attprio_att", "attprio_sess_att",
          "bal_anchor_fc", "bal_anchor_fc_sess"]
         + ["res%d" % j for j in (1, 2, 3, 5)] + ["ress%d" % j for j in (1, 2, 3, 5)])
METHODS = [m for m in os.environ.get(
    "METHODS", ",".join(b + t for b in _BASE for t in ("", "_full"))).split(",") if m]

if HBM and HBM_FEED:
    # the block feed is wired into the GENERIC assembly path only. flrm / flrm2 / flrK_tok build
    # their own fed lists and would silently receive bare units - refuse instead of misreporting.
    _HBM_BAD = [m for m in METHODS if m.startswith("flrm") or m.startswith("flrK")]
    assert not _HBM_BAD, ("HBM_FEED=1 does not cover the flrm/flrm2/flrK_tok assembly paths; "
                          "remove %s from METHODS" % _HBM_BAD)
    # EXPAND (and the notebook s16 patch that promotes it to the generic path) rewrites the fed
    # unit list from the same `keep`, and its branch returns before HBM's expansion runs, so
    # HBM_FEED would then silently feed nothing extra. Mutually exclusive by assertion.
    # NB: the s16 sentinel token is deliberately NOT spelled here - writing it would make that
    # patch believe it had already been applied and no-op with rc=0 instead of aborting.
    assert not EXPAND, "EXPAND and HBM_FEED both rewrite the fed context - enable exactly one"

_SYSTEM_DEFAULT = ("You are a helpful assistant that can read the context and memorize it for future "
                  "retrieval.")
# A launcher may replace the system message for a dataset whose own prompt has a header the harness
# cannot place before the context (BEAM Listing 44). Default is unchanged for every existing cell.
SYSTEM_MESSAGE = os.environ.get("SYSTEM_MESSAGE") or _SYSTEM_DEFAULT
# EXTRACT_SYSTEM_MESSAGE (2026-09-20): the system message of the two generative calls BEFORE the reader,
# the extraction (disk-memoised on (MODEL, system, prompt)) and the router. Default = SYSTEM_MESSAGE, so
# every existing cell and its extraction cache are unchanged. A reader-prompt experiment pins it to the
# shipped message: the extraction and the router stay byte-identical (and cached); only the reader moves.
EXTRACT_SYSTEM_MESSAGE = os.environ.get("EXTRACT_SYSTEM_MESSAGE") or SYSTEM_MESSAGE


def norm(t):
    t = str(t).lower()
    t = "".join(c for c in t if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", t).split())


def subem(pred, golds):
    p = norm(pred)
    if isinstance(golds, (str, int, float)) or golds is None:
        golds = [golds]
    return float(any(norm(g) in p for g in golds))


def slotkey(a):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(a).lower()).split())


def sessions_from_context(ctx):
    """The store is the `context` column: [date, turns, date, turns, ...] as a Python literal.

    Verified over all five rows - the count of 'Chat Time' markers equals the number of (date, turns)
    pairs the literal yields, and equals the number of distinct sessions across the sixty per-question
    haystacks on rows 17-20 (row 21 has 110 here against 111 there; every gold-bearing turn is still
    present, so the difference does not touch what is measured).
    """
    lit = ast.literal_eval(ctx)
    out, i = [], 0
    while i < len(lit):
        if isinstance(lit[i], str) and i + 1 < len(lit) and isinstance(lit[i + 1], list):
            out.append((lit[i], lit[i + 1])); i += 2
        else:
            i += 1
    return out


def rounds_of(sess):
    """One round = a user turn plus the assistant turn that answers it, tagged with session and date.

    The session index is the document id: it becomes the Corpus parent, the hierarchy's grouping key and
    the session-level graph's node identity. Two lookup tables are returned alongside so that metadata's
    has_answer flags - which sit on user turns and on a few assistant turns - can be resolved to rounds.
    """
    rtext, rkey, sess_id, rdate = [], [], [], []
    by_user, by_asst = {}, {}
    for si, (date, turns) in enumerate(sess):
        for ti, t in enumerate(turns):
            if not isinstance(t, dict) or t.get("role") != "user":
                continue
            nxt = ""
            if ti + 1 < len(turns) and isinstance(turns[ti + 1], dict) \
                    and turns[ti + 1].get("role") != "user":
                nxt = turns[ti + 1].get("content", "")
            j = len(rtext)
            rtext.append(("[" + date + "] " + t.get("content", "") + "\n" + nxt).strip())
            rkey.append(t.get("content", "").strip())
            sess_id.append(si); rdate.append(date)
            by_user.setdefault(t.get("content", "").strip(), j)
            if nxt:
                by_asst.setdefault(nxt.strip(), j)
    return rtext, rkey, sess_id, rdate, by_user, by_asst


def gold_rounds_per_question(haystacks, by_user, by_asst):
    """Which rounds hold the evidence, from metadata's has_answer flags. Diagnostic only, never fed.

    has_answer exists in the metadata and not in the context string, so using it for recall costs nothing
    and leaks nothing. A handful of questions carry no flag at all (four of sixty on row 21, none on row
    17); those get an empty set and are excluded from the recall denominator rather than counted as
    misses.
    """
    out = []
    for g in haystacks:
        gold = set()
        for sess in g:
            for t in sess:
                if not isinstance(t, dict) or not t.get("has_answer"):
                    continue
                c = str(t.get("content", "")).strip()
                j = by_user.get(c, by_asst.get(c))
                if j is not None:
                    gold.add(j)
        out.append(gold)
    return out


def ntok(t):
    return len(ENC.encode(t, disallowed_special=()))


# The packing rule of MemoryAgentBench's chunker (utils/eval_other_utils.py, chunk_text_into_sentences), written
# for this harness: sentence boundaries from nltk, then sentences packed greedily up to chunk_size tokens of
# the reader's tokenizer. Every dataset in the suite except the two we already had a native unit for is
# read through this, so the retrieval unit is the official one and 10 units is the official ceiling.
# XQUOTA: which quotas get the x-channel arms (flrx / flrs / flrK). (4,3,3) alone reproduces
# lme_pf4.py exactly; the default here adds the two the rho gate's high branch uses.
_XQ = os.environ.get("XQUOTAS", "4,3,3;2,0,8;5,0,5")
_XQUOTAS = tuple(tuple(int(x) for x in q.split(",")) for q in _XQ.split(";") if q.strip())


FLOORS_LT10 = os.environ.get("FLOORS_LT10", "")   # "rule": scale floors down for U < 10 (see _floors_lt10)


def _floors_lt10(U, fd, fs, fg):
    """Channel floors for U < 10 from the shipped triple (fd, fs, fg): two dense anchors, one slot for each other
    non-zero channel, the rest by the shipped ratio with largest remainders; ties follow floor_fill's fill order
    (dense, graph, sparse). (4,3,3): U=4 -> (2,1,1), U=5 -> (3,1,1), U=10 -> (4,3,3)."""
    w = (fd, fs, fg); f = [0, 0, 0]
    if fd > 0:
        f[0] = min(2, U)
    for i in (2, 1):
        if w[i] > 0 and sum(f) < U:
            f[i] = 1
    R = U - sum(f)
    if R > 0 and sum(w) > 0:
        q = [R * x / float(sum(w)) for x in w]; a = [int(x) for x in q]; left = R - sum(a)
        for i in sorted((0, 2, 1), key=lambda i: -(q[i] - a[i]))[:left]:
            a[i] += 1
        f = [f[i] + a[i] for i in range(3)]
    return tuple(f)
HDETECT = int(os.environ.get("HDETECT", "0") or 0)
HDET_INFO = {}

# Strongest structural marker first. Each is matched on the RAW text; none uses a line-start anchor
# that chunking could destroy except where the marker is meaningless off a line start.
HDET_CANDIDATES = [
    ("doc_block", r"(?m)^[ \t]*Document\s+\d+\s*:"),
    ("chapter", r"\bCHAPTER\s+(?=[IVXLCDM0-9])"),
    ("chapter_lc", r"(?m)^[ \t]*Chapter\s+(?=[IVXLCDM0-9])"),
    ("part", r"(?m)^[ \t]*(?:PART|BOOK|SECTION|ACT)\s+(?=[IVXLCDM0-9])"),
    ("md_head", r"(?m)^#{1,6}[ \t]+\S"),
    ("para", r"\n[ \t]*\n"),
]


def _hd_chunk_starts(text, chunks):
    """Raw-text character offset of each chunk, WITHOUT re-chunking.

    `chunk_official` builds every chunk as `" ".join(sentences)`, so the sentences appear in the raw
    text in order. Walking them with a cursor recovers each chunk's start exactly, and a sentence
    punkt normalised away simply advances the cursor by its own length - which can only move a
    boundary by a few characters, never across a marker that a whole chunk sits after.
    """
    starts, cur = [], 0
    for c in chunks:
        first = c.split(" ", 1)[0] if c else ""
        j = text.find(first, cur) if first else -1
        if j < 0:
            j = cur
        starts.append(j)
        cur = j + len(c)
    return starts


def detect_parents(text, chunks):
    """(par, info). Dataset-agnostic; returns one parent per unit when no marker earns the band."""
    n = len(chunks)
    tried = []
    for name, pat in HDET_CANDIDATES:
        offs = [m.start() for m in re.finditer(pat, text)]
        p = len(offs) + (1 if (not offs or offs[0] > 0) else 0)
        tried.append((name, len(offs)))
        if len(offs) < 1 or p < 2 or n < 2 * p:
            continue
        starts = _hd_chunk_starts(text, chunks)
        par = []
        for s in starts:
            k = 0
            for o in offs:
                if o <= s:
                    k += 1
                else:
                    break
            par.append("p%d" % k)
        npar = len(set(par))
        if npar < 2 or n < 2 * npar:      # re-check on the REALISED grouping, not the marker count
            continue
        return par, {"rule": name, "npar": npar, "units_per_par": round(n / float(npar), 2),
                     "markers": len(offs), "tried": tried}
    return ["u%d" % i for i in range(n)], {"rule": "per_unit", "npar": n, "units_per_par": 1.0,
                                           "markers": 0, "tried": tried}


def _atom_split(text, name):
    """Atoms for one named separator.  Empty pieces are dropped."""
    if name == "blank":
        return [x for x in re.split(r"\n\s*\n", text) if x.strip()]
    if name == "newline":
        return [x for x in re.split(r"\n", text) if x.strip()]
    import nltk
    for _r in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(_r)
        except LookupError:
            nltk.download(_r.split("/")[-1], quiet=True)
    return nltk.sent_tokenize(text)


_ATOM_JOIN = {"blank": "\n\n", "newline": "\n", "sentence": " "}


def _atom_choose(text, chunk_size):
    """THE RULE: the coarsest separator with at least FIT_Q of its atoms inside chunk_size."""
    for name in ("blank", "newline", "sentence"):
        a = _atom_split(text, name)
        if len(a) < 2:
            continue
        fit = sum(1 for x in a
                  if len(ENC.encode(x, allowed_special={"<|endoftext|>"})) <= chunk_size) / float(len(a))
        if fit >= FIT_Q:
            return name, fit, len(a)
    return "sentence", 0.0, len(_atom_split(text, "sentence"))


def chunk_official(text, chunk_size=4096):
    """Pack ATOMS up to chunk_size.  An atom is never split unless it alone exceeds chunk_size,
    and chunks are rejoined with the separator the atoms were split on."""
    name = ATOM
    if name == "auto":
        name, _fit, _n = _atom_choose(text, chunk_size)
        print("    ATOM=auto chose %s (fit %.3f, %d atoms) at chunk_size=%d"
              % (name, _fit, _n, chunk_size), flush=True)
    atoms = _atom_split(text, name)
    if ATOM1:
        out = [a for a in atoms if a.strip()]
        print("    ATOM1: unit = atom, %d atoms -> %d units (no packing, no splitting)"
              % (len(atoms), len(out)), flush=True)
        return out
    jn = _ATOM_JOIN[name]
    out, cur, n = [], [], 0
    over = 0
    for a in atoms:
        t = ENC.encode(a, allowed_special={"<|endoftext|>"})
        if len(t) > chunk_size:
            # LAST RESORT, per atom and automatic: without this the requested size is a no-op.
            over += 1
            if cur:
                out.append(jn.join(cur))
                cur, n = [], 0
            for o in range(0, len(t), chunk_size):
                out.append(ENC.decode(t[o:o + chunk_size]))
            continue
        if n + len(t) > chunk_size and cur:
            out.append(jn.join(cur))
            cur, n = [], 0
        cur.append(a)
        n += len(t)
    if cur:
        out.append(jn.join(cur))
    print("    ATOM=%s atoms=%d oversized=%d -> %d chunks at chunk_size=%d"
          % (name, len(atoms), over, len(out), chunk_size), flush=True)
    return out
# The sub-dataset names in MemoryAgentBench's metadata, by prefix (its utils/templates.py normalises them the same way).
DATASET_MAPPING = [(("ruler_", "qa"), "ruler_qa"), (("icl_",), "in_context_learning"),
                   (("infbench_", "sum"), "infbench_sum"), (("eventqa_",), "eventqa"),
                   (("recsys_", "redial"), "recsys_redial"), (("longmemeval_",), "longmemeval"),
                   (("factconsolidation_",), "factconsolidation"),
                   (("detective_", "qa"), "detective_qa"), (("beam_",), "beam")]


def norm_dataset(src):
    for pats, name in DATASET_MAPPING:
        if all(x in str(src) for x in pats):
            return name
    raise NotImplementedError("unknown sub_dataset: %s" % src)


# The reader's question template of each dataset is the benchmark's own: MemoryAgentBench's rag_agent query
# (utils/templates.py, [<dataset>]["query"]["rag_agent"]) and, for BEAM, the tail of its RAG answer prompt (src/prompts.py,
# from "QUESTION:" on; the instruction header travels in SYSTEM_MESSAGE because the harness puts the context first).
# Both are read at run time from the benchmark repositories that scripts/fetch_mab_repo.sh and scripts/fetch_beam_repo.sh
# place beside this file, so no copy lives here. Templates reach only the reader: no context depends on them.
_MAB_TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mab_repo", "utils", "templates.py")
_BEAM_PROMPTS = os.path.join(os.environ.get("BEAM_REPO") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                          "beam", "BEAM"), "src", "prompts.py")


def _mab_rag_query():
    """{dataset: rag_agent template} from the benchmark, or {} when its repo is not beside us. MemoryAgentBench renders
    them with str.format(); this harness uses .replace("{question}", q), so the doubled braces format() would collapse
    are collapsed here instead (in_context_learning asks for the literal `{label}`)."""
    if not os.path.exists(_MAB_TPL):
        return {}
    import importlib.util
    _sp = importlib.util.spec_from_file_location("_mabtpl", _MAB_TPL)
    _m = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    out = {}
    for _v in vars(_m).values():
        if not isinstance(_v, dict):
            continue
        for _k, _sub in _v.items():
            if isinstance(_sub, dict) and isinstance(_sub.get("query"), dict):
                _t = _sub["query"].get("rag_agent")
                if isinstance(_t, str):
                    out[_k] = _t.replace("{{", "{").replace("}}", "}")
    return out


def _beam_answer_tail():
    """BEAM's RAG answer prompt from "QUESTION:" on, <question> as {question}, line ends stripped; read, not executed."""
    if not os.path.exists(_BEAM_PROMPTS):
        return None
    _src = open(_BEAM_PROMPTS, encoding="utf-8").read()
    for _n in ast.walk(ast.parse(_src)):
        if isinstance(_n, ast.Constant) and isinstance(_n.value, str) and "ANSWER REQUIREMENTS:" in _n.value \
                and "Be direct and concise" in _n.value:
            _t = _n.value[_n.value.index("QUESTION:"):].replace("<question>", "{question}")
            return "\n".join(_l.rstrip() for _l in _t.split("\n")).strip("\n")
    return None


# The bytes every published read used. A checkout of either repository that yields another template stops the run:
# a prompt is an instrument, and a silently edited one is an instrument change nothing else would catch.
_RAG_SHA1 = {"ruler_qa": "c38334458952e5b3", "eventqa": "8d472924aadcbef8", "longmemeval": "7bd8b12fef2c699e",
             "factconsolidation": "fb10145e33872b5b", "in_context_learning": "b66564212080caa9",
             "recsys_redial": "e1f3f7e938270700", "infbench_sum": "7b45f4c0e0cf6711",
             "detective_qa": "b75f0e821ddd2d12", "beam": "78b941f62e52e278"}
RAG_QUERY = _mab_rag_query()
_BT = _beam_answer_tail()
if _BT is not None:
    RAG_QUERY["beam"] = _BT
for _k, _t in sorted(RAG_QUERY.items()):
    _h = hashlib.sha1(_t.encode()).hexdigest()[:16]
    if _k in _RAG_SHA1 and _h != _RAG_SHA1[_k]:
        raise AssertionError("RAG_QUERY[%r] is not the template the published reads used (sha1 %s, want %s)"
                             % (_k, _h, _RAG_SHA1[_k]))


def _rag_query(ds):
    if ds not in RAG_QUERY:
        raise SystemExit("the reader's question template for %r is the benchmark's own: run scripts/fetch_mab_repo.sh "
                         "(MemoryAgentBench) or scripts/fetch_beam_repo.sh (BEAM) first" % ds)
    return RAG_QUERY[ds]


# BEAM_QTPL_FILE (2026-09-20, prompt experiment on BEAM): a launcher-supplied question template for the
# BEAM key ONLY, read from a file so the exact bytes are auditable (the launcher logs the sha1). Set
# together with SYSTEM_MESSAGE; every MemoryAgentBench template above is untouched. Unset = Listing 44.
_BQT = os.environ.get("BEAM_QTPL_FILE", "")
if _BQT:
    RAG_QUERY["beam"] = open(_BQT).read()
    assert "{question}" in RAG_QUERY["beam"], "BEAM_QTPL_FILE must contain {question}"
    print("    BEAM_QTPL_FILE=%s sha1=%s" % (_BQT, hashlib.sha1(RAG_QUERY["beam"].encode()).hexdigest()[:12]), flush=True)

# P2 (off by default): the ruler_qa template with one instruction of ours inserted after its second sentence.
_P2_ADD = ("Use the most complete and specific form of the answer exactly as written in the documents (include full "
           "names, dates, and location qualifiers). If the question is yes/no, answer only yes or no. ")
RAG_QUERY2 = (RAG_QUERY["ruler_qa"].replace("other words. ", "other words. " + _P2_ADD, 1)
              if "ruler_qa" in RAG_QUERY else None)
P2 = os.environ.get("P2", "") not in ("", "0")


# --- DOCMERGE BEGIN ---
# SPARSEX: score an overlapped window centred on each unit instead of the bare unit.
# DOCFLOOR: reserve top-N UNIQUE DOCUMENTS per channel, by that channel's own score.
# SPARSEX = h, the half-width in UNITS. The window is [i-h, i+h] = 2h+1 units, symmetric.
#   h=1 -> 3 units (1,536 tok)   h=2 -> 5 units (2,560 tok)   h=3 -> 7 units (3,584 tok)
# h=3 is the largest SYMMETRIC window that fits the protocol chunk_size of 4,096 at CHUNK=512:
# 2h+1 <= 4096/512 = 8 gives h <= 3.5, and h must be an integer for the window to be centred.
# An 8-unit window would exactly fill the budget but cannot be symmetric, so it would silently
# anchor the evidence off-centre - the defect this lever exists to remove.
SPARSEX = int(os.environ.get("SPARSEX", "0") or 0)
DOCFLOOR = os.environ.get("DOCFLOOR", "") not in ("", "0")
# SXFEED = h for the FEED window, independent of the scoring window. SXFEED alone (SPARSEX unset)
# is the registered-open cell REPORT16:86 calls SPARSE-PARENTFEED: fine 512 flat admit, coarse
# parent-bounded supply. It is the CE_SUBMAX principle - score fine, make coarse visible - applied
# to supply instead of to scoring, which is why it is admissible where window-SCORING is not.
SXFEED = int(os.environ.get("SXFEED", "0") or 0)
# The benchmark's own chunk_size. Not a tunable: the window is "as much as the protocol affords".
OFFICIAL_CHUNK = 4096


def _ext_windows(rtext, parent, h):
    """(lo, hi) inclusive: the window [i-h, i+h] centred on each unit, clipped at its parent.

    2h+1 units, SYMMETRIC, stride 1 - one window per unit, so consecutive windows share 2h of their
    2h+1 members. That overlap is the point: a span straddling any boundary lies wholly inside the
    window centred near it, which a disjoint partition cannot guarantee.

    CLIPPED, not slid. Near a parent boundary the window is SHORTER rather than shifted, because a
    shifted window is no longer centred and off-centre anchoring is the defect this replaces. The
    cost is that boundary-adjacent units are scored on less text; the alternative cost - silently
    re-introducing anchoring - is worse and harder to see.
    """
    N = len(rtext)
    out = []
    for i in range(N):
        lo = hi = i
        for _ in range(h):
            if lo - 1 >= 0 and parent[lo - 1] == parent[i]:
                lo -= 1
            if hi + 1 < N and parent[hi + 1] == parent[i]:
                hi += 1
        out.append((lo, hi))
    return out


def _win_merge(idxs, sface, win, parent):
    """Feed each SPARSE-FACE unit as its window; merge same-document windows; dedup.

    Three things happen, in this order, and each is one of the principal's three statements:
      1. a face unit is replaced by its overlapped sliding window (centred, parent-bounded)
      2. windows are UNIONED, so a unit covered twice is fed once - deduplication
      3. the union is emitted in position order, so windows that overlap or abut inside one
         document read as ONE continuous passage rather than as repeated fragments
    Non-face units pass through untouched, and with an empty `sface` this is the identity.
    """
    out = []
    for i in idxs:
        if i in sface:
            a, b = win[i]
            for j in range(a, b + 1):
                if j not in out:
                    out.append(j)
        elif i not in out:
            out.append(i)
    return sorted(out)


MR_ALPHA = int(os.environ.get("MR_ALPHA", "0") or 0)          # = alpha; 0 disables the operator
H_MERGE = int(os.environ.get("H_MERGE", "1") or 0)          # the release's chunk_ext, main = 1


# SXMERGE: the sparse face is fed as the WINDOW it was scored on, not as its centre unit.
# Default on whenever a scoring window exists, because scoring a window and feeding its
# centre is not a choice, it is the defect this removes. SXMERGE=0 reproduces the earlier legs.
SXMERGE = int(os.environ.get("SXMERGE", "1") or 0)
# MRPOOL: one candidate pool per channel decides documents AND material; nothing leaves it.
# ALPHA_D/S/G: alpha is the DENSE channel's depth. The release oversamples one ranking, the
# cross-encoder (ref_impl.py:831 slices sort_scores). Giving alpha to all three was ours.
MRPOOL = int(os.environ.get("MRPOOL", "0") or 0)
ALPHA_D = int(os.environ.get("ALPHA_D", "0") or 0) or MR_ALPHA
ALPHA_S = int(os.environ.get("ALPHA_S", "0") or 0) or (1 if MRPOOL else MR_ALPHA)
ALPHA_G = int(os.environ.get("ALPHA_G", "0") or 0) or (1 if MRPOOL else MR_ALPHA)
MRDOCS = os.environ.get("MRDOCS", "chan")
assert MRDOCS in ("chan", "global", "fill"), "MRDOCS=%s" % MRDOCS


MRPOOLK = int(os.environ.get("MRPOOLK", "1") or 0)
# D17: deepen the dense head until the arm holds its ten documents, bounded by the candidate set.
# Legal because the material is capped PER DOCUMENT (`_chan_pool`), so a longer head cannot feed
# more - it can only feed a different tenth document. Asserted live at the feed site.
MRDEEP = int(os.environ.get("MRDEEP", "1") or 0)
# D19. Cross-encoder units ranked above CEFLOOR are exempt from the per-document depth cap, but
# ONLY inside documents already claimed - never at the cost of an eleventh document.
CEFLOOR = int(os.environ.get("CEFLOOR", "0") or 0)
# D20. k*alpha is the POOL (core2.py:374, ref_impl.py:831), not a per-document quota.
MRNOCAP = int(os.environ.get("MRNOCAP", "0") or 0)
# D18: max-min fair (water-filling) per-document budget instead of a flat equal share. OFF by
# default - see the module docstring; the measured token response is DECREASING past ~15K.
MRWFILL = int(os.environ.get("MRWFILL", "0") or 0)


def _mr_head(order, docs, want, floor):
    """Head length: at least `want`, at least `floor`, and ALWAYS reaching the last representative.

    The last clause is the defect this exists to remove. `_doc_claim` walks the FULL ranking looking
    for distinct parents, so at H_SCAN=3 - stride-1 windows sharing 6 of 7 members - sparse's three
    representatives sit far apart while `sparse[:3]` spans a measured mean of 1.6 chapters. A head
    slice is therefore not a superset of the floor it supplies material for, the representatives had
    to be unioned back in afterwards, and that union turned the one assert that would have caught it
    into a tautology.
    """
    pos = {}
    for n, i in enumerate(order):
        if i not in pos:
            pos[i] = n
    need = max([pos[j] for j in docs if j in pos] or [-1]) + 1
    return max(int(want), int(floor), int(need))


def _mr_all(head, parent, docs):
    """Every unit of `head` that belongs to a claimed document, plus the representatives.

    This is `core2.py:386` `pool += sibs` restricted to the k*alpha head - no per-document
    quota. The representative union matters because MRDEEP can discover a tenth document PAST the
    head; without it that document would be claimed and then fed nothing, which is worse than not
    claiming it. Order is head order, then any representative the head did not already carry.
    """
    dset = {parent[j] for j in docs}
    out = [i for i in head if parent[i] in dset]
    seen = set(out)
    for j in docs:
        if j not in seen:
            out.append(j)
            seen.add(j)
    return out


def _chan_pool(pool, parent, docs, alpha):
    """At most `alpha` of this channel's own top-ranked units PER CLAIMED DOCUMENT.

    A flat head is an oversample of ONE chapter, not of the document set: a chapter here is ~67
    units and the cross-encoder's ranks cluster inside it, so dense's own documents were taking ~4
    units each while every supplemented document got exactly its representative. alpha is depth per
    document - that is what `upsample_factor * top_k2` means when top_k2 counts fed documents.

    No representative-repair loop. A representative is its document's earliest-ranked unit in this
    channel's order, so the per-document cap admits it first, and `_mr_head` guarantees it is inside
    `pool`. `set(docs) <= set(out)` is therefore a live invariant rather than a repaired one.
    """
    dset = {parent[j] for j in docs}
    per, out = {}, []
    for i in pool:
        p = parent[i]
        if p not in dset or per.get(p, 0) >= max(alpha, 1):
            continue
        per[p] = per.get(p, 0) + 1
        out.append(i)
    return out


def _mr_flat(pool, parent, docs):
    """The pre-MRPOOLK 3b, verbatim, so MRPOOLK=0 reproduces the legs on disk byte for byte."""
    dset = {parent[j] for j in docs}
    out = [i for i in pool if parent[i] in dset]
    for j in docs:
        if j not in out:
            out.append(j)
    return out


def _doc_claim(order, parent, n_docs, taken, must=0):
    """UP TO `n_docs` units from `order` whose parents are distinct AND unclaimed by an earlier one.

    Returns (units, skipped):
      units    at most `n_docs`. It may come up SHORT, deliberately - a channel that has no
               unclaimed document left must NOT pad with a document somebody already brought, which
               is what a rank-order top-up would do. The caller redistributes the deficit instead
               (dense first), so the ten documents fill from whichever channel can still supply one.
               That is the author's rule: when sparse and graph are full, top up from dense.
      skipped  this channel's OWN representative for each document it wanted but found claimed.
               These are handed to the merge as material (E3), which is what keeps C5c true: the
               channel's own best inside a shared document still reaches the reader, it just does
               not spend a floor slot on a document somebody else already brought.

    The first `must` are unconditional, to keep `assert set(ce_rank[:2]) <= selection`. When those
    two share a document they cost two units for one document, and since the UNIT budget is U, ten
    documents is then arithmetically out of reach - hence the witness threshold is "9 or 10".
    """
    if n_docs <= 0:
        # A zero quota must claim NOTHING. Without this the loop appends one unit and only then
        # tests `len(out) >= n_docs`, so a 0-quota channel would claim a document and, worse, LOCK
        # it away from the channels that do have a quota. The grid contains (5,0,5), (2,0,8),
        # (10,0,0) and (0,10,0), so this is not a hypothetical.
        return [], []
    out = list(order[:must])
    if len({parent[i] for i in out}) >= n_docs:
        # The unconditional anchors alone already fill this channel's quota - f_d = 2 with must = 2.
        # Without this the loop appends one MORE before testing `len(out) >= n_docs` and returns
        # must+1 units, blowing the unit budget. The grid contains (2,6,2), (2,5,3), (2,0,8) and
        # (2,2,6), so this fires on four of sixteen tuples; the budget assert caught it on the first
        # question of the first paid leg. `must <= n_docs` holds for every tuple in the grid (the
        # smallest non-zero floor is 2), so no anchor is ever dropped here.
        return out, []
    seen = {parent[i] for i in out}
    own = set(seen)                  # documents THIS channel actually holds. `seen` also carries the
    skipped = []                     # ones it considered and yielded, so it cannot be the counter.
    for i in order:
        if i in out:
            continue
        p = parent[i]
        if p in seen:
            continue
        seen.add(p)
        if p in taken:
            if len(skipped) < n_docs:
                skipped.append(i)
            continue
        out.append(i)
        own.add(p)
        if len(own) >= n_docs:       # DOCUMENTS claimed, not units spent: two anchors in one
            break                    # document buy one document and cost two units.
    return out, skipped


def _chan_seed(order, parent, docs, n_docs, alpha, skipped=()):
    """One channel's MERGE MATERIAL: its top n_docs*alpha candidates lying in the floor's documents.

    THE ORDER OF THE TWO OPERATIONS IS THE PAPER'S AND IT MATTERS. Oversample FIRST
    (ref_impl.py:831 takes upsample_factor*top_k2 off the head of the ranking), THEN group by
    document (ref_impl.py:888 walks the unique documents of exactly those candidates). Choosing
    documents first and then scanning the WHOLE ranking for members would admit a rank-70 unit purely
    for sharing a chapter with rank 1 - on eventqa a chapter is tens of units, so that inversion
    quietly swallows most of a chapter and the oversample stops being an oversample.

    The floor's own units are material by definition: they are the documents' representatives, the
    very chunks whose rank chose the document.
    """
    dset = {parent[i] for i in docs}
    want = max(n_docs, n_docs * max(alpha, 1))
    # Walk until `want` members of THIS channel's documents are found, bounded at 4x the head so it
    # stays an oversample. A fixed head order[:want] was correct while every channel held its own
    # top documents; under global claiming the later channels' documents sit deeper in their own
    # lists, and a fixed head then lands almost nothing inside them - which silently collapses their
    # effective alpha. The bound is what keeps this from becoming the docs-first-then-scan inversion
    # warned about above: past 4x the head a member is no longer an oversample of the ranking, it is
    # just another unit of the same chapter.
    out = []
    for i in list(order)[:want * 4]:
        if parent[i] in dset:
            out.append(i)
            if len(out) >= want:
                break
    for i in list(docs) + list(skipped):
        if i not in out:                 # representatives are material by definition, claimed or not
            out.append(i)
    return out


def _mr_merge(base, seeds, win, stale, keepset, sface=(), swin=None):
    """As below, but a unit in `sface` expands by `swin` instead of `win`.

    That is the dense/sparse asymmetry at the FEED. Sparse ranked a +-H_SCAN window, so the window
    is what it deduplicates and merges; dense and graph ranked units, so they expand at the paper's
    chunk_ext. Two geometries, dispatched per candidate, never mixed into one array.
    """
    if swin is None or not sface:
        return _mr_merge_one(base, seeds, win, stale, keepset)
    out = []
    for i in list(base) + list(seeds):
        a, b = (swin[i] if i in sface else win[i])
        for j in range(a, b + 1):
            if j in stale and j not in keepset:
                continue
            out.append(j)
    return sorted(set(out))


def _mr_merge_one(base, seeds, win, stale, keepset):
    """The merge: candidate -> its window, UNION, position order. No re-rank, no re-cut.

    `base` is the arm's selected view (the prune view's `keep`, or the composite's pool) and `seeds`
    is the alpha oversample. Both are merged, because the release merges every candidate in the
    upsampled pool and not only the ones that would have been fed without it.

    `stale` is the prune view's superseded set. A merged neighbour that is a superseded round must
    not re-enter through the merge, or the prune view silently stops pruning and the two views
    converge - the file applies the same guard at its other two expansion sites, `j not in _st`.
    Passing an empty `stale` makes this the composite view, which keeps everything by definition.
    """
    if MRGATE and _MR_CAPNOW[0] and _MR_TOKNOW[0]:
        # RESERVATION, not a switch. The author's point, and it is right: a binary "skip the merge when
        # the base already fills the cap" is the trivial case. The invariant that matters is
        # per-unit -- the U units the PROTOCOL grants are inviolable, and expansion may spend only
        # what is left over. Two things made the switch too coarse:
        #   1. at an intermediate fill the merge still ran and _mr_cap could trim a base unit;
        #   2. _mr_cap's anchors are `_MR_SEED` -- the ALPHA seeds -- not `base`, so the units it
        #      protects from the trim are not the ones the protocol granted (call site :2978).
        # Here the base is kept whole and neighbours are admitted, nearest-to-their-seed first,
        # only while they fit in ROW_CAP - weight(base). If nothing fits, the merge is the identity
        # and this arm IS the board leg -- which is what the 4096 cells need.
        _tk, _cap = _MR_TOKNOW[0], _MR_CAPNOW[0] - 64
        _keep = sorted(set(base))
        _spent = sum(_tk[j] + 4 for j in _keep if 0 <= j < len(_tk))
        _cand = []
        for i in list(base) + list(seeds):
            a, b = win[i]
            for j in range(a, b + 1):
                if j in _keep or (j in stale and j not in keepset):
                    continue
                _cand.append((abs(j - i), j))
        _seen = set()
        for _d, j in sorted(_cand):
            if j in _seen:
                continue
            _seen.add(j)
            _w = (_tk[j] + 4) if 0 <= j < len(_tk) else 0
            if _spent + _w > _cap:
                continue
            _keep.append(j); _spent += _w
        return sorted(set(_keep))
    out = []
    for i in list(base) + list(seeds):
        a, b = win[i]
        for j in range(a, b + 1):
            if j in stale and j not in keepset:
                continue
            out.append(j)
    return sorted(set(out))


def _mr_docs(fb, parent):
    """The feed's document order: dense's documents, then sparse's, then graph's, first-seen wins.

    This is the order the governor cuts from the tail of, so it is a priority, not a score.
    """
    out = []
    for k in ("d", "s", "g"):
        for i in fb.get(k, ()):
            p = parent[i]
            if p not in out:
                out.append(p)
    return out


MRCAP = int(os.environ.get("MRCAP", "1") or 0)
_MR_CAPBOUND = [0]


# ---- CROSS-ENCODER SCORE CACHE ------------------------------------------------------------------
# A cross-encoder score is a pure function of (model, question text, unit text). This persists it,
# keyed on a signature that pins all three: the reranker's name and a digest of the ENTIRE corpus
# text list, so a different ATOM/CHUNK/store can never read these back. Unit indices are meaningful
# only within one corpus, which is exactly what the digest fixes.
CE_CACHE = os.environ.get("CE_CACHE", "")
_CE_DIR = CE_CACHE if CE_CACHE not in ("", "0", "1") else "ce_cache"
_CE_ON = CE_CACHE not in ("", "0")
_CE_DB, _CE_PATH, _CE_DIRTY, _CE_HIT, _CE_MISS = {}, [None], [0], [0], [0]
_RG = {}          # RENDER=group diagnostics, written on every call, read only under DRY


_CE_NAME = ["BAAI/bge-reranker-v2-m3"]      # replaced with the built reranker's own .name


def _ce_knobs():
    """Everything outside the (query, unit) pair that changes the number the scorer returns.

    CEML is the truncation point, not a tuning detail: rerankers.py:30-34 records eight eventqa
    runs at the old 320 default averaging 0.615 against 0.700 at 1024. CE_SUBMAX / CE_SUB_TOK /
    CE_SLIDE replace the score with a max over pieces, which is a different quantity, not a more
    precise one. CEBS is in here because a batch changes the padding, and the safe direction on a
    knob we have not measured is to miss the cache, never to read another configuration's score
    back under this one's label.
    """
    return "|".join(["%s=%s" % (k, os.environ.get(k, d)) for k, d in
                     (("CEML", "1024"), ("CEBS", "64"), ("CE_SUBMAX", ""),
                      ("CE_SUB_TOK", ""), ("CE_SLIDE", ""))] + ["name=" + _CE_NAME[0]])


def _ce_sig(texts):
    """CORPUS identity. The file name is this plus the scorer digest, and both must match."""
    h = hashlib.sha1()
    h.update(("ce1|%s|%d\n" % (_CE_NAME[0], len(texts))).encode())
    for t in texts:
        h.update(hashlib.sha1(t.encode("utf-8", "replace")).digest())
    return h.hexdigest()[:16]


def _ce_load(texts):
    if not _CE_ON:
        return
    os.makedirs(_CE_DIR, exist_ok=True)
    _kn = _ce_knobs()
    _CE_PATH[0] = os.path.join(_CE_DIR, "%s__%s.json"
                               % (_ce_sig(texts), hashlib.sha1(_kn.encode()).hexdigest()[:8]))
    _CE_DB.clear()
    try:
        _CE_DB.update(json.load(open(_CE_PATH[0])))
    except Exception:
        pass
    print("    CE_CACHE %s  %d question(s) already scored  [%s]"
          % (_CE_PATH[0], len(_CE_DB), _kn), flush=True)


def _ce_save():
    if not (_CE_ON and _CE_PATH[0] and _CE_DIRTY[0]):
        return
    try:                            # another process may have added questions since we loaded
        _on_disk = json.load(open(_CE_PATH[0]))
    except Exception:
        _on_disk = {}
    # The values are per-question {unit: score} MAPS, not scalars. A plain dict.update replaces
    # the other writer's whole map for a question -- 811 to 3,253 units each on the live rec file.
    # Union them: within one question the same unit is the same score, so the order is free.
    for _q, _m in _on_disk.items():
        _CE_DB.setdefault(_q, {}).update(_m)
    tmp = "%s.%d.tmp" % (_CE_PATH[0], os.getpid())
    with open(tmp, "w") as f:
        json.dump(_CE_DB, f)
    os.replace(tmp, _CE_PATH[0])
    _CE_DIRTY[0] = 0


class _CachedScorer(MC.Scorer):
    """MC.Scorer, preloaded from and written back to the disk cache.

    Identical scores by construction: a score read from disk is the same float the cross-encoder
    produced when it was written; the corpus digest guarantees the same unit text and the scorer
    digest the same truncation and pooling. `cev.sh` + `cecheck.py` prove it end to end by running
    a cell three times -- cache off, cold, warm -- and comparing ctxhash and the fed unit sets,
    rather than trusting this docstring.
    """

    def __init__(self, ce, corpus, query, qvec=None):
        MC.Scorer.__init__(self, ce, corpus, query, qvec)
        self._k = hashlib.sha1(query.encode("utf-8", "replace")).hexdigest()[:16] if _CE_ON else None
        if _CE_ON:
            for i, v in _CE_DB.get(self._k, {}).items():
                self._s[int(i)] = v

    def score(self, items):
        if not _CE_ON:
            return MC.Scorer.score(self, items)
        n0 = len(self._s)
        want = sum(1 for i in items if i not in self._s)
        out = MC.Scorer.score(self, items)
        _CE_HIT[0] += len(items) - want
        _CE_MISS[0] += want
        if len(self._s) != n0:
            _CE_DB[self._k] = dict((str(i), v) for i, v in self._s.items())
            _CE_DIRTY[0] = 1
        return out


def _mr_cap(fed, seeds, parent, tok, cap, protect=()):
    """Keep every document; trim each to an equal share of the budget. Identity when it already fits.

    The protocol's budget is retrieve_num=10 items of chunk_size=4096, so one document's share is
    one item's worth. A document over its share is trimmed from the edge furthest from the chunk
    that selected it - the seed - because that neighbourhood is the reason the document is here.
    """
    if not fed:
        return sorted(fed), 0
    if sum(tok[j] + 4 for j in fed) <= cap - 64:
        return sorted(fed), 0
    _MR_CAPBOUND[0] += 1        # the witness round 1 was diagnosed with; `cut` alone now reads 0
                                # both when the cap did not bind and when it bound and refused all
    # PROTECT (MRGATE). The U units the protocol itself grants are inviolable: their weight is
    # reserved off the top and only the leftover is shared out. Without this the equal-share rule
    # evicts a WHOLE 4096-unit to save 44 tokens -- measured on det, where ten protocol units come
    # to 40,940 against a budget of 40,896, and the cap then dropped 1-2 documents per question.
    prot = [j for j in fed if j in protect]
    rest = [j for j in fed if j not in protect]
    budget = cap - 64 - sum(tok[j] + 4 for j in prot)
    if budget <= 0:
        return sorted(prot), len(rest)
    blocks = {}
    for j in rest:
        blocks.setdefault(parent[j], []).append(j)
    if not blocks:
        return sorted(prot), 0
    if MRWFILL:
        # D18. Max-min fair share: a document that needs less than an equal share takes what it
        # needs and RELEASES the rest, so the surplus reaches the documents that are still over.
        # `str(d)` is an explicit tiebreak - an un-tiebroken sort over document keys is exactly the
        # H12 cross-run variance defect, and this file does not add a second one.
        _dem = {d: sum(tok[j] + 4 for j in js) for d, js in blocks.items()}
        _rem, _ord, per_d = budget, sorted(blocks, key=lambda d: (_dem[d], str(d))), {}
        for _n, _d in enumerate(_ord):
            _s = _rem // (len(_ord) - _n)
            per_d[_d] = min(_dem[_d], _s)
            _rem -= per_d[_d]
    else:
        per_d = dict.fromkeys(blocks, budget // len(blocks))
    out, cut = [], 0
    for d, js in blocks.items():
        per = per_d[d]
        # The anchor is the chunk that SELECTED this document, and under `protect` the seeds sit
        # in the protected set, which `js` excludes by construction. Look for the anchor across the
        # document's whole fed set or the trim silently walks out from the median instead.
        _ajs = list(js) + [j for j in prot if parent[j] == d]
        anchors = [j for j in _ajs if j in seeds] or [js[len(js) // 2]]
        for j in sorted(js, key=lambda x: (min(abs(x - a) for a in anchors), x)):
            if sum(tok[k] + 4 for k in out if parent[k] == d) + tok[j] + 4 > per:
                cut += 1
                continue
            out.append(j)
    return sorted(prot + out), cut


def _mr_gov(fed, docs, parent, tok, cap):
    """Fit the ceiling by dropping WHOLE DOCUMENTS off the tail. Never re-scores, never reorders.

    The release does the same cut - `create_prompt(''.join(rerank[:top_k2]))` drops whole documents
    off the end of the document order without re-scoring anything. Here it is mandatory rather than
    cosmetic: the harness's ceiling check RAISES, so an over-budget question aborts the whole run.

    A document that holds a selected unit but was chosen by no channel sorts FIRST, so it is never
    what gets dropped. `tok[j] + 4` charges the newline and the "[CURRENT] " marker, which is a small
    over-charge and therefore the safe direction.
    """
    rank = {d: n for n, d in enumerate(docs)}
    _last = len(rank) + 1          # a document no channel chose is cut BEFORE any that did
    blocks = {}
    for j in fed:
        blocks.setdefault(parent[j], []).append(j)
    out, spent, budget = [], 0, cap - 64
    for d in sorted(blocks, key=lambda x: rank.get(x, _last)):
        cost = sum(tok[j] + 4 for j in blocks[d])
        if spent + cost > budget:
            if out:
                break
            for j in blocks[d]:          # even one document overruns: keep its head, in position order
                if spent + tok[j] + 4 > budget:
                    break
                out.append(j)
                spent += tok[j] + 4
            break
        out.extend(blocks[d])
        spent += cost
    return sorted(out)


def _doc_unique(order, parent, n_docs, must=0):
    """`n_docs` units from `order`, parents distinct, with the first `must` admitted unconditionally.

    `order` is ONE channel's own ranking - CE for dense, BM25 for sparse, reach for graph - so the
    criterion stays per-channel, which is the whole point.

    `must` exists because the file asserts an invariant this would otherwise break:
        assert set(ce_rank[:2]) <= _pp, ("anchors lost", _cn)
    The cross-encoder's top two are admitted unconditionally, and a document-unique rule drops the
    second whenever it shares a document with the first. That is the same "forced spread discards
    the good region's runner-up" mechanism that cost HBM 0.090 token-matched, so the invariant is
    right and the rule yields to it: the anchors go in first, and document-uniqueness governs only
    the slots after them.

    Falls back to plain rank order if the ranking cannot supply enough distinct parents, so the
    floor is never left short.
    """
    out = list(order[:must])
    seen = {parent[i] for i in out}
    for i in order:
        if i in out:
            continue
        p = parent[i]
        if p in seen:
            continue
        seen.add(p)
        out.append(i)
        if len(out) >= n_docs:
            return out
    for i in order:                      # not enough distinct parents: top up in rank order
        if i not in out:
            out.append(i)
            if len(out) >= n_docs:
                break
    return out
# --- DOCMERGE END ---


# --- STRUCTURE PROFILES ---------------------------------------------------------------------------
# A profile supplies the three things a corpus can carry that a fixed-size chunker destroys: the
# UNIT, the PARENT, and the ORDER. It is chosen by a REGISTRY keyed on the dataset name, never
# inferred from the bytes - a byte sniffer keyed on a copula fires on a re-serialised ICL shot bank
# and deletes 67-85% of it, because the key lands on the record template rather than the entity.
PROFILE = int(os.environ.get("PROFILE", "0") or 0)
# LPROFILE=1 takes unit, parent and order from the LEDGER instead of from the hand-written
# PROFILES registry or from text-marker detection.  It is the same interface, computed at
# write time for every store rather than registered for one.
LPROFILE = int(os.environ.get("LPROFILE", "0") or 0)
PROFILE_INFO = {}


def _fc_profile(text):
    """FactConsolidation: unit = one numbered fact, parent = the RTR slot, order = the serial.

    This is `ladder12.py:109-117` - `parse_facts` then
    `MC.Corpus(ftext, fv, parent_of=fslot, order_of=forder)` - the program that actually produced
    every FC number on the board (prio 0.320, reach_hopce 0.350, attprio_slotc 0.380, best 0.420).
    Imported, not retyped, so the ported path is provably the scored one.
    """
    import sys
    _sd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    if _sd not in sys.path:
        sys.path.insert(0, _sd)
    from real_e1_conflict import parse_facts          # FACT_RE + the registered slot fn
    facts = sorted(parse_facts(text))                 # (serial, body, slot), sorted by serial
    # The serial stays in the fed text: mab_repo/utils/templates.py tells the reader to resolve by
    # "finding the newest fact with larger serial number". The slot key uses the body only.
    # BODIES only. ladder12.py:110-113 embeds `ftext`, the body, and prints the serial at
    # render time. Putting the serial in the unit text pollutes the dense embedding and the
    # BM25 token stream with 18,332 distinct integers - my own error in the first port.
    unit = [t for _, t, _ in facts]
    par = [s for _, _, s in facts]
    order = [o for o, _, _ in facts]
    return unit, par, order


PROFILES = {"factconsolidation": _fc_profile}


def load_row(df, ROW):
    """(rtext, rkey, parent, rdate, gold_of, qs, ans, qtypes, dsname, n_parents).

    longmemeval keeps the round-level loader it was built with. Everything else is read at the official
    chunk size, and the parent is whatever grouping the dataset itself puts ABOVE that chunk - chapters
    in eventqa. ruler has no such grouping (its `Document N:` blocks sit BELOW a 4096-token chunk), so
    every chunk is its own parent and the document-level arms reduce to the unit-level ones. That is
    not a degradation: rep(document) = supersede(argmax_score(members)) is the identity when each
    document has one member, so `attprio_doc_att` and `attprio_att` are the same arm there by
    construction, and the run checks it rather than assuming it.
    """
    m = df["metadata"].iloc[ROW]
    ds = norm_dataset(m.get("source"))
    qs = list(df["questions"].iloc[ROW])
    ans = list(df["answers"].iloc[ROW])
    if ds == "longmemeval":
        sess = sessions_from_context(str(df["context"].iloc[ROW]))
        rtext, rkey, sess_id, rdate, by_user, by_asst = rounds_of(sess)
        gold_of = gold_rounds_per_question(m.get("haystack_sessions"), by_user, by_asst)
        qt = list(m.get("question_types")) if m.get("question_types") is not None else [""] * len(qs)
        return rtext, rkey, sess_id, rdate, gold_of, qs, ans, qt, ds, len(sess)
    if LPROFILE:
        # $0 at read time: every extraction was paid for once at ingest and is on disk.  A
        # missing chunk is fatal rather than skipped - a corpus built from a partial ledger
        # would silently be a different corpus, and would still score.
        import ledger_profile as LP
        _u, _p, _oi, _miss, _nch = LP.build(str(df["context"].iloc[ROW]), CHUNK)
        assert not _miss, ("LEDGER cache incomplete: %d of %d chunks missing at CHUNK=%d - "
                           "build it before scoring" % (_miss, _nch, CHUNK))
        assert _u, "LEDGER produced no units"
        # EMISSION order, not the ingest index: see _ORD_OK at the structural block below.
        _o = list(range(len(_u)))
        PROFILE_INFO.clear()
        PROFILE_INFO.update({"profile": "ledger", "units": len(_u), "parents": len(set(_p)),
                             "units_per_par": round(len(_u) / float(max(len(set(_p)), 1)), 2),
                             "order": "emission", "src_chunks": _nch,
                             "ingest_positions": len(set(_oi))})
        print("    LPROFILE=ledger units=%d parents=%d units/par=%.2f order=emission "
              "(from %d chunks at CHUNK=%d)"
              % (len(_u), len(set(_p)), PROFILE_INFO["units_per_par"], _nch, CHUNK), flush=True)
        gold_of = [{i for i, c in enumerate(_u) if any(norm(str(a)) in norm(c) for a in list(av))}
                   for av in ans]
        return _u, _u, _p, _o, gold_of, qs, ans, [ds] * len(qs), ds, len(set(_p))
    _pf = PROFILES.get(ds) if PROFILE else None
    if _pf is not None:
        # The profile replaces unit, parent and order together. Nothing downstream of here is
        # dataset-aware, so the rest of the harness sees an ordinary corpus with a real hierarchy.
        _u, _p, _o = _pf(str(df["context"].iloc[ROW]))
        PROFILE_INFO.clear()
        PROFILE_INFO.update({"profile": ds, "units": len(_u), "parents": len(set(_p)),
                             "units_per_par": round(len(_u) / float(max(len(set(_p)), 1)), 2),
                             "order": "serial"})
        print("    PROFILE=%s units=%d parents=%d units/par=%.2f order=serial"
              % (ds, len(_u), len(set(_p)), PROFILE_INFO["units_per_par"]), flush=True)
        gold_of = [{i for i, c in enumerate(_u) if any(norm(str(a)) in norm(c) for a in list(av))}
                   for av in ans]
        return _u, _u, _p, _o, gold_of, qs, ans, [ds] * len(qs), ds, len(set(_p))
    if AUTOSTRUCT:
        # unit = one record, serial stripped; order = the serial; parent = the slot.
        _txt = str(df["context"].iloc[ROW])
        _lines = [l for l in _txt.split("\n") if l.strip()]
        _NUM = re.compile(r"^\s*(\d+)\.\s+(.+)$", re.DOTALL)
        _COP = re.compile(r"^(.*?)\s+\b(?:is|are|was|were|has|have|had)\b\s+(.+)$",
                          re.IGNORECASE | re.DOTALL)
        _REL = re.compile(r"^(.*?\b(?:of|by|in|with|for|to)\b)\s+([^.]+)\.?\s*$",
                          re.IGNORECASE | re.DOTALL)
        _AS_SLOT = int(os.environ.get("AS_SLOT", "1") or 1)
        # AS_SLOT=5: the slot key the registry actually uses. Read from
        # src/real_e1_conflict.py:39 -- `normalized_slot` is `first capitalised entity` + "|" +
        # `the sentence with every capitalised entity masked to _`. Nothing in it mentions
        # FactConsolidation: it is a general English heuristic (entity spans + a relation
        # skeleton), which is why calling it "the hand registry" was wrong. What made it a
        # registry was only that it was keyed on the dataset NAME in PROFILES; the function
        # itself generalises, so it can be computed from the bytes like everything else here.
        _CAPRE = re.compile(r"\b([A-Z][\w.'-]+(?:\s+(?:of|the|and|de|van|von|di)?\s*"
                            r"[A-Z][\w.'-]+)*)\b")
        _CAPSTOP = {"The", "A", "An", "In", "On", "At", "Of", "This", "That", "These", "Those",
                    "It", "Its", "Which", "What", "Who", "Where", "When", "There", "He", "She",
                    "They", "His", "Her", "Their"}

        def _ents(t):
            return [m.group(1) for m in _CAPRE.finditer(t or "")
                    if m.group(1).split()[0] not in _CAPSTOP]

        def _slot5(t):
            _e = _ents(t)
            _sub = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", _e[0].lower())).strip() if _e else ""
            _sk = re.sub(r"\s+", " ", _CAPRE.sub("_", t or "")).strip().rstrip(".").strip()
            return _sub + "|" + _sk
        _norm_slot = lambda t: re.sub(r"\s+", " ",
                                     re.sub(r"[^a-z0-9 ]", " ", (t or "").lower())).strip()
        _m = [_NUM.match(l) for l in _lines]
        _numbered = sum(1 for x in _m if x) / float(max(len(_lines), 1))
        _MINREC = float(os.environ.get("AS_MINREC", "0.8"))
        assert _numbered >= _MINREC, (
            "AUTOSTRUCT: only %.3f of lines are `<serial>. <body>`; this store has no record "
            "structure to use and the caller should fall back to MSCALE" % _numbered)
        import ledger_profile as _LP
        ch, par, _ord, _hit = [], [], [], 0
        for _x, _l in zip(_m, _lines):
            if _x:
                _ser, _body = int(_x.group(1)), _x.group(2).strip()
            else:
                _ser, _body = len(ch), _l.strip()
            ch.append(_body)                       # 1: the BODY only, serial stripped
            _ord.append(_ser)                      # 2: order IS the serial
            # AS_SLOT=1 is the copula-only key (coverage .798 on FactConsolidation). AS_SLOT=2
            # adds the general English fact template `<entity> <relation> of|by|in|... <value>`,
            # which this store uses for a fifth of its records ("X plays the position of Y",
            # "X was created by Y"). Coverage .798 -> .998, but the slots also COARSEN
            # (11,961 -> 11,309 parents against the registry's 12,810), so which is better for
            # multi-hop is measured, not argued.
            _sl = None
            if _AS_SLOT >= 5:
                par.append("s:" + _slot5(_body)); _hit += 1
                continue
            _cm = _COP.match(_body)
            if _AS_SLOT >= 2:
                if _cm and _cm.group(1).strip():
                    _r = _REL.match(_cm.group(2))
                    _sl = (_cm.group(1).strip() + " " + _r.group(1)) if _r else _cm.group(1).strip()
                else:
                    _r = _REL.match(_body)
                    _sl = _r.group(1) if _r else None
            elif _cm and len(_cm.group(1).split()) >= 2:
                _sl = _cm.group(1)
            if _sl:
                # AS_SLOT=3: normalise the head and keep it VERBATIM. `ledger_profile.slotkey`
                # reduces a head to a sorted SET of words longer than two characters, truncated to
                # 80 chars -- word order is destroyed, "Dr. No" loses its "No", and long heads
                # collide. Measured on FactConsolidation that puts TWENTY-FIVE facts in one slot;
                # the verbatim key caps the largest slot at FOUR. It matters because prune keeps
                # one unit per slot: an over-merged slot makes it DELETE correct facts, which is
                # survivable single-hop (the answer is one fact) and fatal multi-hop (the chain
                # loses a link). n_sup confirms it: 4.00 under the word-set key against the
                # registry's 1.75, on identical gold.
                par.append("s:" + (_norm_slot(_sl) if _AS_SLOT >= 3 else _LP.slotkey(_sl)))
                _hit += 1
            else:
                par.append("u:%d" % len(ch))
        _cov = _hit / float(max(len(ch), 1))
        HDET_INFO.clear()
        HDET_INFO.update({"rule": "autostruct", "npar": len(set(par)),
                          "units_per_par": round(len(ch) / float(max(len(set(par)), 1)), 2),
                          "markers": _hit, "tried": [("numbered", round(_numbered, 3)),
                                                     ("slot_cov", round(_cov, 3))]})
        print("    AUTOSTRUCT numbered=%.3f slot-cov=%.3f units=%d parents=%d units/par=%.2f"
              % (_numbered, _cov, len(ch), len(set(par)),
                 len(ch) / float(max(len(set(par)), 1))), flush=True)
        gold_of = [{i for i, c in enumerate(ch) if any(norm(str(a)) in norm(c) for a in list(av))}
                   for av in ans]
        return ch, ch, par, _ord, gold_of, qs, ans, [ds] * len(qs), ds, len(set(par))
    if HYBRID:
        # unit = the record, parent = the semantic slot of that record.
        _txt = str(df["context"].iloc[ROW])
        ch = chunk_official(_txt, chunk_size=CHUNK)
        import ledger_profile as _LP
        _COP = re.compile(r"^(.*?)\s+\b(?:is|are|was|were|has|have|had)\b\s+(.+)$",
                          re.IGNORECASE | re.DOTALL)
        par, _hit, _cum = [], 0, 0
        for _s in ch:
            _body = re.sub(r"^\s*\d+\.\s*", "", _s.strip())      # drop a leading serial
            _m = _COP.match(_body)
            if _m and len(_m.group(1).split()) >= 2:
                par.append("s:" + _LP.slotkey(_m.group(1)))
                _hit += 1
            else:
                par.append("p:%d" % (_cum // max(MSCALE or (CHUNK * 8), 1)))
            _cum += len(ENC.encode(_s))
        _cov = _hit / float(max(len(ch), 1))
        assert ch == chunk_official(_txt, chunk_size=CHUNK), "HYBRID must not alter units"
        if _cov < float(os.environ.get("HYB_MINCOV", "0.5")):
            # not a slot-bearing store: fall back to the positional layer rather than build a
            # parent map from a handful of accidental matches.
            par = ["p:%d" % (i // max(1, (MSCALE or CHUNK * 8) // max(CHUNK, 1))) for i in range(len(ch))]
        HDET_INFO.clear()
        HDET_INFO.update({"rule": "hybrid_slot" if _cov >= 0.5 else "hybrid_fallback",
                          "npar": len(set(par)),
                          "units_per_par": round(len(ch) / float(max(len(set(par)), 1)), 2),
                          "markers": _hit, "tried": [("slot_cov", round(_cov, 3))]})
        print("    HYBRID slot-coverage=%.3f units=%d parents=%d units/par=%.2f rule=%s"
              % (_cov, len(ch), len(set(par)),
                 len(ch) / float(max(len(set(par)), 1)), HDET_INFO["rule"]), flush=True)
    elif MSCALE:
        # MULTI-SCALE.  Cut the FINE units over the WHOLE text -- byte-identical to the single
        # condition -- and then LABEL each unit with the coarse block its start token falls in.
        #
        # THE BUG THIS REPLACES.  The first implementation cut coarse first and re-cut fine INSIDE
        # each block. That changed the units themselves (FactConsolidation 537 -> 598, InfBench-Sum
        # 95 -> 117) and emitted a boundary remainder per block, some as small as TEN tokens. The
        # cross-encoder prefers a short fully-relevant chunk to a long diluted one, so those
        # remainders won the ranking and the delivered budget HALVED (40,394 -> 18,899 tokens at
        # fine 4096). Two consequences, both measured:
        #   1. single vs multi compared two different corpora, not one corpus with and without a
        #      parent layer;
        #   2. `ce_only`, the arm that consults no parent at all and must therefore be inert, moved
        #      +0.100 on MH-QA and +0.230 on FC-SH. That was the unit change leaking through, and
        #      it is exactly what a placebo arm exists to catch.
        # Labelling leaves the units alone, so the placebo is inert BY CONSTRUCTION and the only
        # difference between the two conditions is the parent id.
        _txt = str(df["context"].iloc[ROW])
        ch = chunk_official(_txt, chunk_size=CHUNK)
        par, _cum = [], 0
        for _s in ch:
            par.append("m%d" % (_cum // MSCALE))
            _cum += len(ENC.encode(_s))
        assert ch == chunk_official(_txt, chunk_size=CHUNK), "units must be identical to single"
        assert ch, "MSCALE produced no units"
        assert len(set(par)) < len(ch), (
            "MSCALE produced a FLAT layer (%d units, %d parents): the coarse scale is not coarser "
            "than the fine one, so this run would silently be a single-scale run"
            % (len(ch), len(set(par))))
        _ns = lambda t: len(re.sub(r"\s+", "", t))
        _lost = _ns(_txt) - _ns("".join(ch))
        assert abs(_lost) <= 0.01 * max(1, _ns(_txt)), (
            "MSCALE lost or duplicated text: %d non-space chars of %d" % (_lost, _ns(_txt)))
        HDET_INFO.clear()
        HDET_INFO.update({"rule": "mscale", "npar": len(set(par)),
                          "units_per_par": round(len(ch) / float(max(len(set(par)), 1)), 2),
                          "markers": 0, "tried": [("mscale_label", MSCALE)]})
        print("    MSCALE fine=%d coarse=%d units=%d parents=%d units/par=%.2f "
              "(non-space char delta %d)"
              % (CHUNK, MSCALE, len(ch), len(set(par)),
                 len(ch) / float(max(len(set(par)), 1)), _lost), flush=True)
    else:
        ch = chunk_official(str(df["context"].iloc[ROW]), chunk_size=CHUNK)
    if MSCALE or HYBRID:
        pass
    elif HDETECT:
        # One rule for every dataset, chosen by evidence in the raw text rather than by `ds ==`.
        par, _hdi = detect_parents(str(df["context"].iloc[ROW]), ch)
        HDET_INFO.clear()
        HDET_INFO.update(_hdi)
        print("    HDETECT rule=%s npar=%d units/par=%.1f markers=%d  tried=%s"
              % (_hdi["rule"], _hdi["npar"], _hdi["units_per_par"], _hdi["markers"], _hdi["tried"]),
              flush=True)
    elif ds == "eventqa":
        par, cur = [], 0
        for c in ch:
            if re.search(r"(?m)^CHAPTER\b", c):
                cur += 1
            par.append("ch%d" % cur)
        HDET_INFO.clear()
        HDET_INFO.update({"rule": "eventqa_legacy", "npar": len(set(par))})
    else:
        par = ["u%d" % i for i in range(len(ch))]
        HDET_INFO.clear()
        HDET_INFO.update({"rule": "per_unit_legacy", "npar": len(ch)})
    # gold = the chunks that contain an answer string. Same rule FactConsolidation scores presence
    # with, and it needs no labels beyond the answers the split already ships.
    gold_of = [{i for i, c in enumerate(ch) if any(norm(str(a)) in norm(c) for a in list(av))}
               for av in ans]
    return ch, ch, par, list(range(len(ch))), gold_of, qs, ans, [ds] * len(qs), ds, len(set(par))


def main():
    df = pd.read_parquet(PARQ)
    # swap item 4 (2026-10-14): MIRROR_BGE names the encoder (real_e1_lme.BGE); EMB_TRUNCDIM truncates its output; EMB_ATTN sets
    # its attention implementation; CE_NAME names the reranker. All unset = the published encoder and reranker, unchanged.
    _ekw = dict(**({"truncate_dim": int(os.environ["EMB_TRUNCDIM"])} if os.environ.get("EMB_TRUNCDIM") else {}),
                **({"model_kwargs": {"attn_implementation": os.environ["EMB_ATTN"]}} if os.environ.get("EMB_ATTN") else {}))
    emb = SentenceTransformer(BGE, device=__import__("os").environ.get("BGE_DEVICE", "mps"), **_ekw); emb.max_seq_length = int(__import__("os").environ.get("EMB_MAXLEN", "448") or 448)   # EMB_MAXLEN (2026-09-28): default 448 = every published run
    ce = MC.build_reranker(**({"name": os.environ["CE_NAME"]} if os.environ.get("CE_NAME") else {}), device=os.environ.get("CE_DEVICE", "mps"))   # CE_DEVICE (release): default "mps" = every published leg
    # MC.Scorer falls back to the BI-ENCODER when this is None (core2.py:90) and the
    # cache cannot tell the two apart. One broken `import rerankers` would poison a corpus
    # file permanently and silently, so refuse instead of scoring.
    assert ce is not None, "no reranker: MC.build_reranker returned None (rerankers import failed)"
    _CE_NAME[0] = getattr(ce, "name", _CE_NAME[0])
    g = Guard(GUARD_NAME, cap_usd=CAPUSD, price=(0.0, 0.0)) if LOCAL \
        else Guard(GUARD_NAME, cap_usd=CAPUSD)

    def extract_cached_only(prompt, note):
        """DIAGNOSTIC (DRY_XC=1): replay the memoised extraction if it exists, else "". Never calls the API."""
        if os.environ.get("DRY_XC", "") not in ("", "0"):
            import hashlib
            k = hashlib.sha1(("\x00".join((MODEL, EXTRACT_SYSTEM_MESSAGE or "", prompt))).encode()).hexdigest()
            fp = os.path.join(XC_DIR, k[:2], k + ".txt")
            if os.path.exists(fp):
                XC_HIT[0] += 1
                with open(fp, encoding="utf-8") as fh:
                    return fh.read()
        return ""
    def extract_call(prompt, note):
        """The extraction, memoised on disk, so that it is a function of its prompt and of nothing
        else - not of the wall clock, not of which backend build served the request, and not of what
        else happens to be in METHODS."""
        if not EXTRACT_CACHE:
            return g.openai(prompt, model=MODEL, max_tokens=600, temperature=0.0,
                            system=EXTRACT_SYSTEM_MESSAGE, seed=SEED, note=note)
        import hashlib
        k = hashlib.sha1(("\x00".join((MODEL, EXTRACT_SYSTEM_MESSAGE or "", prompt))).encode()).hexdigest()
        fp = os.path.join(XC_DIR, k[:2], k + ".txt")
        if os.path.exists(fp):
            XC_HIT[0] += 1
            with open(fp, encoding="utf-8") as fh:
                return fh.read()
        r = g.openai(prompt, model=MODEL, max_tokens=600, temperature=0.0,
                     system=EXTRACT_SYSTEM_MESSAGE, seed=SEED, note=note)
        XC_HIT[1] += 1
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        tmp = fp + ".tmp%d" % os.getpid()
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(r or "")
        os.replace(tmp, fp)
        return r
    # RESUME. OUT was already opened for append, but nothing skipped the questions already in it, so
    # an interrupted cell either duplicated every record on restart or - the way the queues handled it
    # - was deleted and paid for again. A killed run cost 54 EventQA questions this way. Resuming is
    # only sound when the file was written by the SAME arm set: a five-arm partial must not be topped
    # up by a six-arm run, because the missing arm would be absent on the first half of the cell and
    # every per-arm mean would silently mix two populations. So the arm set is checked, and a mismatch
    # refuses to resume rather than producing a file that looks complete.
    _DONE = set()
    if int(os.environ.get("RESUME", "1") or 0) and os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        _VIEWS = (("prune_annot", "composite")
                  + (("prune_norefill", "slotdiv") if NOREFILL else ()))
        _want = {m + "|" + v for m in METHODS for v in _VIEWS}
        _bad = None
        for _ln in open(OUT):
            if not _ln.strip():
                continue
            try:
                _r = json.loads(_ln)
            except Exception:
                continue
            _have = {k for k in _r if k.endswith(tuple("|" + v for v in _VIEWS))
                     and not k.startswith("ctxhash")}
            if DRY and not _have:
                # 09-19: a DRY record carries no score keys, only ctxhash|<arm>|<view>, so the check above
                # read an empty arm set and refused every DRY resume. Read the arm set off the hashes.
                _have = {k[len("ctxhash|"):] for k in _r if k.startswith("ctxhash|")
                         and k.endswith(tuple("|" + v for v in _VIEWS))}
            if _bad is None:
                _bad = (_have != _want)
                if _bad:
                    print("   RESUME REFUSED: existing %s was written by a different arm set" % OUT, flush=True)
                    print("      have %s" % sorted(_have)[:8], flush=True)
                    print("      want %s" % sorted(_want)[:8], flush=True)
                    break
            _DONE.add((_r.get("row"), _r.get("qi")))
        if _bad:
            raise SystemExit("resume refused: arm-set mismatch in %s (delete it or set RESUME=0)" % OUT)
        if _DONE:
            print("   RESUME: %d question(s) already in %s; they will be skipped" % (len(_DONE), OUT), flush=True)
    pf = open(OUT, "a")
    hits = collections.Counter(); bytype = collections.defaultdict(collections.Counter); nq = 0
    print("rows=%s methods=%s reader=%s" % (ROWS, METHODS, MODEL), flush=True)
    print("       per_arm=%s router_canon=%s extract_cache=%s stub=%s"
          % (PER_ARM, ROUTER_CANON, EXTRACT_CACHE, EXTRACT_STUB), flush=True)

    for ROW in ROWS:
        m = df["metadata"].iloc[ROW]
        (rtext, rkey, sess_id, rdate, gold_of, qs, ans,
         qtypes, DSNAME, NPAR) = load_row(df, ROW)
        N = len(rtext)
        # The ceiling is what the OFFICIAL agent would feed on this row: retrieve_num of the
        # official chunks. On eventqa the official chunker emits a few chunks slightly over
        # chunk_size - a single sentence longer than the limit is not split, which is its own
        # behaviour - so the nominal 10x4096 is not attainable there and is the wrong bound.
        # We select from the SAME chunks, so this cap is exact rather than nominal.
        # ONE ceiling, the benchmark's own: retrieve_num x chunk_size. The derived
        # "sum of the ten longest chunks in this row" bound that used to apply outside
        # longmemeval was ours, not the protocol's, and on EventQA it was eight times tighter
        # than the benchmark's budget (5,164 against 40,960). Arms are still compared in tokens -
        # every record now carries tok_by_arm - but they are no longer REFUSED by a bound the
        # benchmark never set.
        # The protocol bound is TEN UNITS at this cell's chunk size, and OFFICIAL_CAP = 10 x 4096
        # is that bound only when the chunker actually produces 4096-token units. On the
        # in-context-learning stores it does not: the context is thousands of labelled examples, the
        # split lands on example boundaries, and clinic150 fed 56,911 tokens for ten units against a
        # 40,960 ceiling - over by 15,951, which no framing allowance can cover. Refusing that run
        # would refuse a PROTOCOL-CONFORMANT ten-unit delivery. So the ceiling is the larger of the
        # nominal budget and what ten units in THIS store can weigh, which still refuses an arm that
        # delivers more than ten units' worth while never refusing one that delivers ten.
        ROW_CAP = OFFICIAL_CAP
        _MR_CAPNOW[0] = ROW_CAP
        try:
            _top = sorted((ntok(t) for t in rtext), reverse=True)[:U]
            ROW_CAP = max(OFFICIAL_CAP, sum(_top))
            _MR_CAPNOW[0] = ROW_CAP
        except Exception:
            pass
        if False:
            # HBM_FEED delivers blocks, so the honest ceiling is MemoryAgentBench's OWN
            # 10 x 4096 - the budget its BM25 baseline spends - not the sum of ten chunk sizes.
            # A profile unit is one record - about 28 tokens on FactConsolidation - so the
            # sum-of-top-U ceiling would be ~280 tokens, 1/146 of what the chunk protocol
            # spends. The honest ceiling for a record corpus is the benchmark's own budget.
            ROW_CAP = OFFICIAL_CAP if (PROFILE and DSNAME in PROFILES) else (
                OFFICIAL_CAP if (EXPAND or (HBM and HBM_FEED) or SXFEED or (MR_ALPHA and NPAR < N)) else sum(
                    sorted((ntok(t) for t in rtext), reverse=True)[:U]))
        _lab = sum(1 for x in gold_of if x)
        print("\n=== row %d  [%s]  units=%d  parents=%d  questions=%d  gold-labelled=%d/%d"
              % (ROW, DSNAME, N, NPAR, len(qs), _lab, len(gold_of)), flush=True)
        # ONE embedding per row - the sixty questions share this context, which the original-file harness
        # could not exploit because there each question carried its own haystack
        kv = np.asarray(emb.encode(rkey, batch_size=32, normalize_embeddings=True,
                                   show_progress_bar=False), dtype=np.float32)
        # parent = SESSION. Without this the Corpus gives every round a null parent, MC.merge
        # treats each as its own parent and the hierarchy silently degenerates to a flat cut.
        # Does this corpus carry what a structural resolver needs? Two facts about the data,
        # no dataset name: an explicit order, and parents that actually group. When both hold,
        # supersession is a $0 lookup over corpus.parent_of / corpus.order_of; when either
        # fails there is nothing to resolve over and the LLM extractor stays in charge.
        _ORD = rdate if ((PROFILE and DSNAME in PROFILES) or LPROFILE or AUTOSTRUCT) else list(range(N))
        # PROVENANCE, not values. "Is there a real order?" cannot be read off the numbers: on all
        # eight FactConsolidation rows the serial is bit-identical to the line index
        # (18,332/18,332 on rows 3 and 7), so a value test rejects the one corpus that has an
        # explicit order. What IS knowable is whether a registered profile DECLARED one - the
        # profile is where the dataset-specific claim lives, and everything after it is generic.
        global MAXOUT
        MAXOUT = genmax_for(DSNAME)
        # A ledger order is DECLARED in the same sense a profile order is: the write path
        # emitted the assertions in ingest order, so the claim lives at ingest and not in a
        # value test over the numbers.
        # THE SAME DEFECT AS `PROFILES`, ONE LEVEL DOWN. This flag decided whether the structural
        # resolver runs by asking whether the DATASET NAME is registered, not whether an explicit
        # order was actually supplied. AUTOSTRUCT supplies one -- a partition agreeing with the
        # registry's at recall 0.999 / precision 0.995, 12,800 parents against 12,810, the same
        # 1.43 units per parent -- and the harness ignored all of it and fell back to LLM
        # extraction. That is why AS_SLOT=3 and AS_SLOT=5 produced byte-identical rho and n_sup
        # despite partitions that differ by 27%: neither was ever read. Ask the loader, not the name.
        _HAS_ORDER = bool((PROFILE and DSNAME in PROFILES) or LPROFILE or AUTOSTRUCT or HYBRID or FORCE_ORDER)
        # and refuse a declared order that cannot rank: an order with ties everywhere makes
        # keep-newest the identity, silently. Demand that it separates almost all units.
        _ORD_OK = _HAS_ORDER and len(set(_ORD)) >= 0.99 * N
        _STRUCTURAL = _ORD_OK and (len(set(sess_id)) < N)
        if _STRUCTURAL:
            print("    STRUCTURAL resolver: parents=%d < units=%d, explicit order"
                  % (len(set(sess_id)), N), flush=True)
        corpus = MC.Corpus(rtext, kv, parent_of=sess_id, order_of=_ORD)
        _ce_load(rtext)
        # One map per ROW, not per arm per question. It is a pure function of the corpus.
        _NW_ROW = MC.newest_per_parent(range(N), corpus) if _STRUCTURAL else {}
        r_ents = [entities(t) for t in rkey]
        ef, hubs = build_entity_index(r_ents, N)
        # IDF over rounds, so a neighbour sharing a rare entity outranks one sharing a common
        # entity - the same weighting ladder12.py:103 uses
        idf = {e: math.log(N / max(len(v), 1)) for e, v in ef.items()}
        # the supersede step's key: the same slot key FactConsolidation parses from text, applied
        # to rounds. It groups 8 of 3220 rounds here, so the step is the identity almost always -
        # which is the point: one operator, and the data decides which face it shows.
        r_slot = [slotkey(t) for t in rkey]
        _sm = collections.defaultdict(list)
        for _i, _s in enumerate(r_slot):
            if _s:
                _sm[_s].append(_i)
        newest_of_slot = {k: max(v) for k, v in _sm.items()}   # rounds are in chronological order
        # ladder12.py:203-258, the SLOT-LEVEL graph. Built once per row: it is a pure function of
        # the corpus. Nodes are parents, each represented by its CURRENT revision, so the search
        # never steps through superseded text on its way outward. `_NW_ROW` is the corpus's own
        # newest-per-parent map, so the representative is read from the declared order rather than
        # chosen by a cross-encoder.
        _ADJ_SLOT = {}
        if _STRUCTURAL:
            _ent_slots = collections.defaultdict(set)
            for _s2, _i2 in _NW_ROW.items():
                for _e2 in (r_ents[_i2] - hubs):
                    _ent_slots[_e2].add(_s2)
            _cutS = max(3, int(0.05 * max(len(_NW_ROW), 1)))
            _ent_slots = {_e2: _v2 for _e2, _v2 in _ent_slots.items() if len(_v2) <= _cutS}
            _ADJ_SLOT = collections.defaultdict(set)
            for _e2, _ss2 in _ent_slots.items():
                for _s2 in _ss2:
                    _ADJ_SLOT[_s2] |= (_ss2 - {_s2})
            print("    STRUCTURAL slot graph: slots=%d edges=%d (ladder12 slot_reach_adj)"
                  % (len(_NW_ROW), sum(len(v) for v in _ADJ_SLOT.values())), flush=True)
        _SG_HITS = [0]

        def _slot_reach_adj(qents, hops, cap):
            """ladder12.py:238-258 verbatim. Seeds are the slots of facts holding a query entity.

            The return is sorted on (hop, unit id) BEFORE any caller truncates: `hop` is keyed by
            slot strings filled from set iteration and Python randomises string hashing per
            process, so an unsorted return keeps a different 400 on every run.
            """
            _seeds = {sess_id[j] for e in qents if e in ef for j in ef[e]
                      if sess_id[j] is not None}
            _hopd = {s_: 0 for s_ in _seeds}
            _front = set(_seeds)
            for _h in range(1, hops + 1):
                if len(_hopd) >= cap or not _front:
                    break
                _nxt = set()
                for _s3 in _front:
                    for _v3 in _ADJ_SLOT.get(_s3, ()):
                        if _v3 not in _hopd:
                            _nxt.add(_v3)
                for _v3 in _nxt:
                    _hopd[_v3] = _h
                _front = _nxt
            return [_u3 for _, _u3 in sorted((_h3, _NW_ROW[_s3]) for _s3, _h3 in _hopd.items()
                                             if _s3 in _NW_ROW)]
        bm = BM25Okapi([_btok(t) for t in rtext])
        # HBM: the block index is a function of the ROW (units, parents, B) and not of the
        # query, so it is built once here alongside the unit-level BM25.
        _HBM_MEM, _HBM_BLKID, _HBM_OF, _HBM_BM = [], {}, {}, None
        if HBM:
            assert not SPH, "HBM and SPH both rewrite the sparse branch - pick one"
            _HBM_MEM, _HBM_BLKID, _HBM_BTXT = _hbm_blocks(rtext, sess_id, HBM)
            for _m in _HBM_MEM:
                _t = tuple(_m)
                for _i in _m:
                    _HBM_OF[_i] = _t
            _HBM_BM = BM25Okapi([_btok(t) for t in _HBM_BTXT])
            print("    HBM B=%d blocks=%d feed=%d backfill=%d  block_tokens min/med/max=%d/%d/%d"
                  % (HBM, len(_HBM_MEM), HBM_FEED, HBM_BF,
                     min(ntok(t) for t in _HBM_BTXT),
                     sorted(ntok(t) for t in _HBM_BTXT)[len(_HBM_BTXT) // 2],
                     max(ntok(t) for t in _HBM_BTXT)), flush=True)
        _SX_BM = None
        _SX_WIN = None
        _MR_WIN = None
        _MR_TOK = None
        if MR_ALPHA:
            # A _full arm merges and THEN re-ranks with the cross-encoder (core2.py:397),
            # which is the composition this operator exists to avoid. Refuse to be scored on one
            # rather than rely on the launcher never naming it.
            assert not any(m.endswith("_full") for m in METHODS), \
                "MR_ALPHA must not be scored on a _full arm - that path re-ranks after the merge"
            # H_MERGE geometry is independent of the scoring window: the principal raised H_SCAN on
            # purpose while the merge hop stays at the paper's h. Two arrays, so neither can
            # silently become the other.
            _MR_WIN = _ext_windows(rtext, sess_id, H_MERGE)
            _MR_TOK = [ntok(t) for t in rtext]
            _MR_TOKNOW[0] = _MR_TOK
            print("    MR_ALPHA alpha=%d  H_MERGE=%d (%d units)  H_SCAN=%d (%d units)  cap=%d"
                  % (MR_ALPHA, H_MERGE, 2 * H_MERGE + 1, SPARSEX, 2 * SPARSEX + 1, ROW_CAP),
                  flush=True)
        if SPARSEX or SXFEED:
            # one geometry, built at the larger half-width; each lever slices what it needs
            _SX_WIN = _ext_windows(rtext, sess_id, max(SPARSEX, SXFEED))
            _sxw = [b - a + 1 for a, b in _SX_WIN]
            print("    WINDOWS h=%d -> %d units; n=%d  actual min/med/max=%d/%d/%d  (score h=%d, feed h=%d)"
                  % (max(SPARSEX, SXFEED), 2 * max(SPARSEX, SXFEED) + 1, len(_SX_WIN),
                     min(_sxw), sorted(_sxw)[len(_sxw) // 2], max(_sxw), SPARSEX, SXFEED), flush=True)
        if SPARSEX:
            assert not HBM, "SPARSEX and HBM both rewrite the sparse branch - pick one"
            assert not SPH, "SPARSEX and SPH both rewrite the sparse branch - pick one"
            _SC_WIN = _ext_windows(rtext, sess_id, SPARSEX)
            _SX_BM = BM25Okapi([_btok("\n".join(rtext[a:b + 1])) for a, b in _SC_WIN])
        if SPH:
            _wmem = []
            _wtxt = []
            for _s in range(0, max(1, len(rtext)), SPHS):
                _mem = list(range(_s, min(len(rtext), _s + SPHW)))
                if SPHP:
                    _mem = [_m for _m in _mem if sess_id[_m] == sess_id[_s]]
                if _mem:
                    _wmem.append(_mem)
                    _wtxt.append(" ".join(rtext[_m] for _m in _mem))
            if SPHP:
                _cov = set()
                for _mm in _wmem:
                    _cov.update(_mm)
                for _u in range(len(rtext)):
                    if _u not in _cov:
                        _wmem.append([_u])
                        _wtxt.append(rtext[_u])
            bmw = BM25Okapi([_btok(t) for t in _wtxt])
        at = {}
        pos_of = {}
        cnt = collections.Counter()
        for i, si in enumerate(sess_id):
            pos_of[i] = cnt[si]; at[(si, cnt[si])] = i; cnt[si] += 1

        order = list(range(len(qs)))
        if NQ:
            random.Random(7).shuffle(order); order = order[:NQ]
        for qi in order:
            if (ROW, qi) in _DONE:
                continue
            q = str(qs[qi]); gold = ans[qi]; qt = qtypes[qi] if qi < len(qtypes) else ""
            q_ret = _qstrip(q)
            qv = np.asarray(emb.encode([q_ret], normalize_embeddings=True, **_QPK), dtype=np.float32)[0]
            sc = _CachedScorer(ce, corpus, q_ret, qv)
            dense = MC.dense_candidates(corpus, qv, K1)
            ce_rank = sc.rank(dense)
            qe = entities(q_ret)
            # ranked by hop-decayed IDF, so the cut below removes the weakly connected rather
            # than the high-indexed. `multihop` returned a set and its order was the node index.
            _off_u, _all_u, _hop_u, _w_u = MC.reach_ranked(
                qe, r_ents, ef, hubs, idf, H, DECAY, cap=None,
                deg=(RDEG or None), stop_rounds_at=U)
            gph = [int(i) for i in _off_u]          # offered: scored, and the branch's membership
            gseen = set(int(i) for i in _all_u)     # reached: only for "already seen" predicates
            if gph:
                sc.score(gph)
            gset = set(gph)
            grank = sc.rank(gph); drest = [i for i in ce_rank if i not in gset]
            # SESSION-level graph: one node per session, represented by its best-scoring candidate round.
            # The session is this benchmark's document, and lifting the graph to the document is the move
            # that gained 7 points on FactConsolidation when nodes went from revisions to slots.
            cand = set(ce_rank) | gset
            rep = {}
            if REP_RULE == "newest":
                for i in range(N):
                    rep[sess_id[i]] = i                      # rounds are appended in session order
            elif REP_RULE == "first":
                for i in range(N):
                    rep.setdefault(sess_id[i], i)
            elif REP_RULE == "cand":
                for i in range(N):
                    k = sess_id[i]
                    if k not in rep or i in cand:
                        rep[k] = i                           # last scored round, else last round
            else:
                # UNIFIED: pick the best match within the document, then advance it to the current
                # revision of its own slot. Reduces to newest_of_slot where a document IS a slot, and to
                # the plain best match where a document's members are unrelated.
                for i in range(N):
                    k = sess_id[i]
                    if k not in rep:
                        rep[k] = i
                    elif i in cand and (rep[k] not in cand or sc.of(i) > sc.of(rep[k])):
                        rep[k] = i
                if REP_RULE == "unified":
                    for k, i in list(rep.items()):
                        s_ = r_slot[i]
                        if s_:
                            rep[k] = newest_of_slot.get(s_, i)
            ent_s = collections.defaultdict(set)
            for k, i in rep.items():
                for e in (r_ents[i] - hubs):
                    ent_s[e].add(k)
            cut = max(3, int(0.05 * max(len(rep), 1)))
            ent_s = {e: v for e, v in ent_s.items() if len(v) <= cut}
            # the SAME reach function, called over documents. `doc_ents` and `ent_s` are the two
            # directions of the document-level index, exactly as `r_ents` and `ef` are for units, so
            # the document graph is not a second implementation - it is the same one at another level.
            doc_ents = {k: (r_ents[i] - hubs) for k, i in rep.items()}
            # ties break by the representative's unit index, i.e. by corpus order - the recency
            # signal the previous (hop, rep_index) sort carried, on a benchmark that is temporal.
            # BROAD seeding, NARROW expansion - the pre-unification semantics, restored. Any
            # unit holding a query entity seeds its document with that entity's IDF; expansion then
            # runs on the narrow document index (ent_s: representative-based, rarity-filtered).
            _seed_w = collections.defaultdict(float)
            for e in qe:
                for j in ef.get(e, ()):
                    _k = sess_id[j]
                    if _k in rep:
                        _seed_w[_k] += idf.get(e, 0.0)
            _off_s, _all_s, hop, _w_s = MC.reach_ranked(
                [], doc_ents, ent_s, hubs, idf, H, DECAY, cap=RCAP,
                tie=lambda k: rep.get(k, 0), seeds=_seed_w, deg=(RDEG or None))
            gph_s = [rep[k] for k in _off_s if k in rep]
            gseen_s = {rep[k] for k in _all_s if k in rep}
            if gph_s:
                sc.score(gph_s)
            if _STRUCTURAL:
                # ladder12.py:340-344. The generic document reach above stays computed but unused
                # here; it is the g-face that the arm's quota reads, and on a declared store that
                # face is the slot search. Measured: dense and sparse already matched ladder12 unit
                # for unit, and this face was the only divergence in flr_4_3_3's head.
                gph_s = _slot_reach_adj(qe, H, U)
                gseen_s = set(gph_s)
                if gph_s:
                    sc.score(gph_s[:400])
                grank_s = sc.rank(gph_s[:400])
                drest_s = [i for i in ce_rank if i not in gseen_s]
                _SG_HITS[0] += 1
            else:
                grank_s = sc.rank(gph_s); drest_s = [i for i in ce_rank if i not in set(gph_s)]
            # Dense-seeded reach, ladder12.py:305-318. The query supplies about one usable entity, so
            # a query-entity search anchors on a single point and only its noise shell grows. Seeding
            # from what dense already found relevant enlarges the ANCHOR instead. A bridge unit is by
            # definition absent from the question, so this is the branch that can reach one.
            _dseeds = ce_rank[:SEEDN]
            _ds = set()
            for _s0 in _dseeds:
                _ds |= set(int(x) for x in multihop(r_ents[_s0], r_ents, ef, hubs, 1, U))
            _ds -= set(_dseeds)
            _dsl = sorted(_ds)[:400]   # a set cut in iteration order was the same defect class as the reach
            if _dsl:
                sc.score(_dsl)
            dseed_rank = sc.rank(_dsl)
            bs = bm.get_scores(_btok(q_ret))
            # K1 deep like FactConsolidation, then cross-encoder ranked, so BM25 competes on the same
            # scale as the other two branches instead of being cut at the budget before it is scored.
            if SPH:
                _ws = bmw.get_scores(_btok(q))
                _wo = sorted(range(len(_wmem)), key=lambda w: -_ws[w])
                sparse = []
                _qts = set(_btok(q))
                for _w in _wo:
                    _mem = _wmem[_w]
                    if SPHM == "cover":
                        _cand = list(_mem)
                        _covd = set()
                        _order = []
                        while _cand:
                            _bestm = max(_cand, key=lambda i: (
                                sum(bm.idf.get(t, 0.0) for t in (set(_btok(rtext[i])) & _qts) - _covd),
                                bs[i], -i))
                            _order.append(_bestm)
                            _covd |= set(_btok(rtext[_bestm])) & _qts
                            _cand.remove(_bestm)
                    elif SPHM == "run" and len(_mem) > RUNL:
                        _bj = max(range(len(_mem) - RUNL + 1),
                                  key=lambda j: sum(bs[_mem[j + t]] for t in range(RUNL)))
                        _runm = _mem[_bj:_bj + RUNL]
                        _order = _runm + sorted([i for i in _mem if i not in _runm],
                                                key=lambda i: (-bs[i], i))
                    else:
                        _order = sorted(_mem, key=lambda i: (-bs[i], i))
                    for _m in _order:
                        if _m not in sparse:
                            sparse.append(_m)
                    if len(sparse) >= K1:
                        break
                sparse = sparse[:K1]
            elif HBM:
                sparse, _HBM_NPURE = _hbm_champions(
                    _HBM_MEM, _HBM_BLKID, bs, _HBM_BM.get_scores(_btok(q_ret)),
                    K1, HBM_BF, N)
            elif SPARSEX:
                # BM25 over the overlapped centred window, ranking the CENTRE unit. Only the score
                # changes; the selected object is still the unit, so nothing downstream is aware.
                _ws = _SX_BM.get_scores(_btok(q_ret))
                sparse = sorted(range(N), key=lambda i: (-_ws[i], i))[:K1]
            else:
                sparse = sorted(range(N), key=lambda i: -bs[i])[:K1]
            if not HBM:
                _HBM_NPURE = 0
            # which units each arm counts as its reserved SPARSE face. Filled where the floors are
            # built, because that is the only place it is known; empty for every arm that has no
            # sparse floor, which is what keeps those arms byte-identical under HBM_FEED=1.
            _HBM_ARMS = {}
            # a TAIL source: only what neither the graph nor the dense candidate set already holds
            srest = [i for i in sc.rank(sparse)
                     if i not in gseen and i not in gseen_s and i not in set(dense)]
            anch = []
            for src in (grank, ce_rank, sc.rank(sparse)):
                for i in src[:1]:
                    if i not in anch:
                        anch.append(i)
            # ATTACHMENT. For each dense anchor, admit its strongest graph neighbour directly -
            # linked to a unit the query already matched, rather than scored against the query itself.
            # This is the step that names attprio and it was missing from this harness entirely.
            _M_ANCH, _CAP = M_ANCH, CAP
            anchors_a = ce_rank[:_M_ANCH]

            def _neighbours(i, same_doc):
                w = collections.defaultdict(float)
                for e in (r_ents[i] - hubs):
                    for j in ef.get(e, ()):
                        if j != i and not same_doc(i, j):
                            w[j] += idf.get(e, 0.0)
                # see ladder12.neighbours: unit id settles equal weights so attachment is reproducible
                return sorted(w, key=lambda j: (-w[j], j))

            # FactConsolidation's balance, ported exactly: one anchor per source over graph, dense and
            # dense-seeded reach, EVERY anchor mapped to its document representative, then one
            # document-mapped neighbour per anchor as the expansion. bal_anchor and bal_anchor_slotc
            # share both of these there and differ only in the graph branch, so the same holds here.
            # FactConsolidation builds ONE anchor set and ONE expansion, shared by both bal_anchor
            # arms; only the graph branch differs between them (ladder12.py:399-437). The graph anchor
            # comes from round-level reach and is then mapped to its document, and the neighbour search
            # excludes the anchor's own document unconditionally.
            _ba = []
            for _src in (grank, ce_rank, dseed_rank):
                for i in _src[:1]:
                    k = rep.get(sess_id[i], i)
                    if k not in _ba:
                        _ba.append(k)
            _be = []
            for a_ in _ba:
                taken = 0
                for j in _neighbours(a_, lambda x, y: sess_id[x] == sess_id[y]):
                    k = rep.get(sess_id[j], j)
                    if k in _ba or k in _be:
                        continue
                    _be.append(k); taken += 1
                    if taken >= 1:
                        break
            if _be:
                sc.score([i for i in _be if i not in sc._s])
            _be = sc.rank(_be)

            att_r, att_s = [], []
            for a in anchors_a:
                for lst, same, mapto in ((att_r, lambda x, y: False, lambda j: j),
                                         (att_s, lambda x, y: sess_id[x] == sess_id[y],
                                          lambda j: rep.get(sess_id[j], j))):
                    taken = 0
                    for j in _neighbours(a, same):
                        k = mapto(j)
                        if k in anchors_a or k in lst:
                            continue
                        lst.append(k); taken += 1
                        if taken >= _CAP:
                            break
            if att_r or att_s:
                sc.score([i for i in att_r + att_s if i not in sc._s])
            heads = {
                "ce_only":   ce_rank,
                # the parameter-free admission rule: rank the union, let the scores decide how many
                # reach candidates survive rather than reserving slots for them
                "union":      sc.rank(list(dict.fromkeys(ce_rank + grank))),
                "union_sess": sc.rank(list(dict.fromkeys(ce_rank + grank_s))),
                "prio":      grank + drest,
                "prio_sess": grank_s + drest_s,
                "bal_anchor": anch + grank + drest,
                "attprio":      anchors_a + grank + drest,
                # the direct twin of FactConsolidation's best arm: same two dense anchors, same
                # graph-takes-the-rest priority, but nodes at the document level
                "attprio_sess": anchors_a + grank_s + drest_s,
                # the faithful ports of FactConsolidation's attprio / attprio_slotc
                "attprio_att":      anchors_a + att_r + grank + drest,
                "attprio_sess_att": anchors_a + att_s + grank_s + drest_s,
                # the faithful ports of FactConsolidation bal_anchor / bal_anchor_slotc
                "bal_anchor_fc":      _ba + _be + grank + drest,
                "bal_anchor_fc_sess": _ba + _be + grank_s + drest_s,
            }
            # How many of the ten slots may the graph take? j=0 is ce_only and j=U is prio, so the two
            # poles are inside the ladder and every intermediate point is a real configuration rather
            # than a hand-picked arm. Reserved slots go to reach rounds that dense's own top-(U-j) does
            # not already hold, so a reservation only ever costs dense its LOWEST-ranked kept units.
            # Which arms draw their graph candidates from the DOCUMENT-level reach. Registered where the
            # heads are built, because that is where it is known; inferring it from the name or from the
            # pool's composition was tried and both were wrong (see the patch note).
            DOC_ARMS = {"union_sess", "prio_sess", "attprio_sess", "attprio_sess_att",
                        "bal_anchor_fc_sess"}
            # The whole range, so the dense floor is read off a curve rather than chosen. The
            # floor is U-j: j=8 is the two-anchor arm, j=7 the three-anchor one, j=0 is ce_only
            # and j=U is prio, so neither pole is outside the ladder and no point is hand-picked.
            for _j in tuple(range(1, 10)):
                _keep = ce_rank[:U - _j]
                _kset = set(_keep)
                heads["res%d" % _j] = _keep + [i for i in grank if i not in _kset] + drest
                heads["ress%d" % _j] = _keep + [i for i in grank_s if i not in _kset] + drest_s
                DOC_ARMS.add("ress%d" % _j)
            # Three-branch floor arms: dense / sparse / document-graph, one reserved floor each,
            # selection by the shared cross-encoder score. flr_D_S_G reads as (dense, sparse, graph)
            # floors. The grid spans the poles - (10,0,0) is all-dense-guaranteed, (8,0,2) is the
            # shipped reservation's shape - so the sweep reads a surface, not a hand-picked point.
            sc.score(sparse)
            # Branches cut AT the floor: f_b reserves the branch's OWN top-f_b, not "any of its
            # top-K1" - loose lists let dense units deep in BM25's list satisfy the sparse floor by
            # overlap, so the lexical best never entered (measured: +0.02 vs the cut-list
            # counterfactual's +0.10). "x" = unfloored universe for base selection and backfill.
            _MR_SEED = {}
            _MR_DOCS = {}
            _MR_SPLIT = {}
            _MR_POOL = {}
            _MR_DEEP = {}
            _MR_MUST = {}
            _MR_FLOOR = {}
            _MR_SFACE = {}
            _MR_UEFF = {}
            _MR_FED = {}
            _MR_CUT, _MR_BOUND = {}, {}
            for _fd, _fs, _fg in ((10, 0, 0), (8, 0, 2), (8, 2, 0), (7, 3, 0),
                                  (6, 2, 2), (5, 3, 2), (6, 3, 1), (4, 3, 3),
                                  (4, 5, 1), (3, 5, 2), (2, 6, 2), (0, 10, 0), (2, 5, 3),
                                  (2, 0, 8), (2, 2, 6), (5, 0, 5)):
                _nm = "flr_%d_%d_%d" % (_fd, _fs, _fg)
                # QSCALE, hoisted: the floors line below this chain reads _qs on the non-_MR_UEFF
                # path, so it must be defined whichever branch runs. It is a function of U alone.
                _qs = max(1, U // 10)
                _LT10 = (FLOORS_LT10 == "rule" and U < 10)
                _fq = _floors_lt10(U, _fd, _fs, _fg) if _LT10 else (_fd * _qs, _fs * _qs, _fg * _qs)
                if MR_ALPHA:
                    # Selection reserves DOCUMENTS and is cut AT the floor. alpha does NOT enter
                    # here: a loose alpha*f branch list lets a document the dense score already
                    # holds satisfy the sparse floor by overlap, so BM25's own top never enters -
                    # the failure this file's header quotes as measured (+0.02 vs +0.10).
                    _kd = _ks = _kg = ()
                    _rd = 0
                    if MRDOCS == "fill":
                        # Every channel keeps its OWN top-f documents - no cross-channel exclusion,
                        # so sparse's floor still holds sparse's own best three. Overlap is paid for
                        # in UNITS (two channels' representatives of one document) and the document
                        # shortfall is then filled from dense, which is the only ranking deep enough
                        # to supply more: grank_s is one representative per document and has none.
                        _bd, _kd = _doc_claim(ce_rank, sess_id, _fd, set(), must=2)
                        _bs, _ks = _doc_claim(sparse, sess_id, _fs, set())
                        _bg, _kg = _doc_claim(grank_s, sess_id, _fg, set())
                        _tk = {sess_id[i] for i in _bd + _bs + _bg}
                        _target = min(_fd + _fs + _fg, NPAR)
                        if MRPOOL:
                            # ONE POOL PER CHANNEL. alpha is dense's depth; sparse and graph take
                            # their floors only. Everything after this line stays inside these three.
                            # k IS THE DOCUMENT COUNT THE CHANNEL WILL HOLD, not its floor.
                            # Dense is the only channel whose count grows, and the supplement exits
                            # at len(_tk) >= _target while _tk already holds dense's own _fd, so
                            # n_d <= _target always and ALPHA_D*_target is the exact supremum.
                            # Measured: ce_rank[:16] spans 7.2 chapters and reached ten on 0/21
                            # questions; ce_rank[:40] spans 10.7 and reached ten on 17/21.
                            _wd = (ALPHA_D * _target) if MRPOOLK else (_fd * max(ALPHA_D, 1))
                            _ws = (ALPHA_S * _fs) if MRPOOLK else (_fs * max(ALPHA_S, 1))
                            _wg = (ALPHA_G * _fg) if MRPOOLK else (_fg * max(ALPHA_G, 1))
                            _PD = list(ce_rank)[:_mr_head(ce_rank, _bd if MRPOOLK else (), _wd, _fd)]
                            _PS = list(sparse)[:_mr_head(sparse, _bs if MRPOOLK else (), _ws, _fs)]
                            _PG = list(grank_s)[:_mr_head(grank_s, _bg if MRPOOLK else (), _wg, _fg)]
                            _MR_POOL[_nm] = (_PD, _PS, _PG)
                            # 3a: DENSE supplements, unique-document first, and ONLY from its pool.
                            # No fall-through to sparse or graph - dense tops up the remainder - and no
                            # walk past the pool. If the pool is exhausted the arm comes SHORT,
                            # which is ref_impl.py:80-81's own "within upsample_top_k2".
                            _nown = len(_bd)
                            _hd = len(_PD)
                            _deep = 0
                            while len(_tk) < _target:
                                _ex, _ = _doc_claim(_PD, sess_id, 1, _tk)
                                if not _ex:
                                    # D17. The head ran out of UNCLAIMED documents. Walk further
                                    # down the SAME cross-encoder ranking rather than returning
                                    # nine documents. `_PD` is re-bound here, not shadowed by a
                                    # separate slice: the first draft passed `ce_rank[:_hd]`
                                    # straight into `_doc_claim` and left `_PD` stale, so the
                                    # supplement left the pool and `assert all(i in _PD ...)`
                                    # fired on the first question. That assert is labelled a
                                    # tautology upstream - it is one only while the claim source
                                    # and the pool are the SAME object, which is why they are.
                                    # The material stays capped at ALPHA_D per document either
                                    # way; `_chan_pool` does that, not the head length (G2).
                                    if MRPOOLK and MRDEEP and _hd < len(ce_rank):
                                        _hd = min(len(ce_rank), max(_hd * 2, _hd + 8))
                                        _PD = list(ce_rank)[:_hd]
                                        _MR_POOL[_nm] = (_PD, _PS, _PG)
                                        _deep += 1
                                        continue
                                    break
                                _bd.append(_ex[0])
                                _tk.add(sess_id[_ex[0]])
                                _rd += 1
                            # Is a shortfall a HEAD-length artifact or is it DATA? If the union of
                            # the three full rankings cannot reach ten distinct parents, no depth
                            # fixes it and `_target` itself is the thing that is wrong. Recorded
                            # rather than assumed - the previous shortfall was patched twice
                            # without anyone checking which of the two worlds this is.
                            _MR_DEEP[_nm] = {
                                "deep": _deep, "head": _hd,
                                "reach_ce": len({sess_id[i] for i in ce_rank}),
                                "reach_sp": len({sess_id[i] for i in sparse}),
                                "reach_gr": len({sess_id[i] for i in grank_s}),
                                "reach_all": len({sess_id[i] for i in
                                                  list(ce_rank) + list(sparse) + list(grank_s)}),
                                "npar": NPAR, "target": _target}
                            # TAUTOLOGY, kept and LABELLED as one: _doc_claim(_PD, ...) returns
                            # members of _PD by construction. Documentation, not a gate.
                            assert all(i in _PD for i in _bd[_nown:])
                            if MRPOOLK:
                                _PD = list(ce_rank)[:max(_mr_head(ce_rank, _bd, _wd, _fd), _hd)]
                                _MR_POOL[_nm] = (_PD, _PS, _PG)
                                # LIVE. Fires exactly when a representative sits past its own head -
                                # which is what mr_pool_esc read as 1-2 per question and I ignored.
                                assert set(_bd) <= set(_PD), ("dense reps outside dense pool", _nm)
                                assert set(_bs) <= set(_PS), ("sparse reps outside sparse pool", _nm)
                                assert set(_bg) <= set(_PG), ("graph reps outside graph pool", _nm)
                        else:
                            for _bl, _od in ((_bd, ce_rank), (_bs, sparse), (_bg, grank_s)):
                                while len(_tk) < _target:
                                    _ex, _ = _doc_claim(_od, sess_id, 1, _tk)
                                    if not _ex:
                                        break
                                    _bl.append(_ex[0])
                                    _tk.add(sess_id[_ex[0]])
                                    _rd += 1
                        _fb = {"d": _bd, "s": _bs, "g": _bg, "x": list(ce_rank)}
                    elif MRDOCS == "global":
                        # (4,3,3) is TEN DISTINCT DOCUMENTS. A document dense already holds does not
                        # also satisfy sparse. Dense claims first because it must: ce_rank[:2] are
                        # asserted into the selection, so its first two documents are not negotiable.
                        # Each channel still returns f units, so U and every downstream cut are
                        # unchanged - and the three lists are now document-disjoint, hence
                        # unit-disjoint, which is what stops one unit satisfying two floors.
                        _tk = set()
                        _bd, _kd = _doc_claim(ce_rank, sess_id, _fd, _tk, must=2)
                        _tk |= {sess_id[i] for i in _bd}
                        _bs, _ks = _doc_claim(sparse, sess_id, _fs, _tk)
                        _tk |= {sess_id[i] for i in _bs}
                        _bg, _kg = _doc_claim(grank_s, sess_id, _fg, _tk)
                        _tk |= {sess_id[i] for i in _bg}
                        # DEFICIT REDISTRIBUTION. A channel that ran out of unclaimed documents does
                        # not pad with a duplicate - it passes its share on, and the quota is filled
                        # from whichever channel still has an unclaimed document, DENSE FIRST. So
                        # (5,0,5) with a graph that can only supply three becomes dense 7 / graph 3,
                        # not eight documents. The binding constraint is the UNIT budget U, which is
                        # why an anchor collision (ce_rank[:2] in one document, two units for one
                        # document) still lands on nine documents rather than ten.
                        _lists = (("d", _bd, ce_rank), ("s", _bs, sparse), ("g", _bg, grank_s))
                        _target = min(_fd + _fs + _fg, NPAR)
                        for _ck, _bl, _od in _lists:
                            while len(_tk) < _target and len(_bd) + len(_bs) + len(_bg) < U:
                                _ex, _ = _doc_claim(_od, sess_id, 1, _tk)
                                if not _ex:
                                    break
                                _bl.append(_ex[0])
                                _tk.add(sess_id[_ex[0]])
                                _rd += 1
                        _fb = {"d": _bd, "s": _bs, "g": _bg, "x": list(ce_rank)}
                    else:
                        _fb = {"d": _doc_unique(ce_rank, sess_id, _fd, must=2),
                               "s": _doc_unique(sparse, sess_id, _fs),
                               "g": _doc_unique(grank_s, sess_id, _fg), "x": list(ce_rank)}
                    # alpha lives here instead: the material the merge will consume at feed time.
                    # The skipped representatives ride along, so a channel that yielded a document
                    # still puts its OWN best chunk of that document in front of the reader.
                    if MRPOOL and MRDOCS == "fill":
                        # 3b: from the SAME pool, every unit whose document is already chosen. The
                        # pool IS the quota, so the static f_c never reaches this path - which is
                        # what removes D3 (16 units spread over nine documents because _chan_seed
                        # was handed _fd=4 while _fb["d"] had grown) and D4 (_chan_seed walking
                        # order[:want*4] = ce_rank[:64]; measured, CE ranks 17-63 were being fed).
                        _PD, _PS, _PG = _MR_POOL[_nm]
                        if MRPOOLK:
                            if MRNOCAP:
                                # `MC.merge`'s `children[:npar*alpha]` -> `pool += sibs`. The head
                                # is the FIXED k*alpha, not the head MRDEEP deepened: deepening is
                                # for DISCOVERING the tenth document, and letting it also widen the
                                # material would silently re-inflate the feed - the exact knob
                                # overloading that produced D4.
                                _sd = _mr_all(list(ce_rank)[:max(ALPHA_D, 1) * _target],
                                              sess_id, _fb["d"])
                                _ss = _mr_all(list(sparse)[:max(ALPHA_S, 1) * max(_fs, 1)],
                                              sess_id, _fb["s"])
                                _sg = _mr_all(list(grank_s)[:max(ALPHA_G, 1) * max(_fg, 1)],
                                              sess_id, _fb["g"])
                            else:
                                _sd = _chan_pool(_PD, sess_id, _fb["d"], ALPHA_D)
                                _ss = _chan_pool(_PS, sess_id, _fb["s"], ALPHA_S)
                                _sg = _chan_pool(_PG, sess_id, _fb["g"], ALPHA_G)
                            if CEFLOOR:
                                # Claimed documents = the ten this arm is allowed to feed. A CE
                                # top-CF unit inside one of them is free in DOCUMENT terms, which
                                # is the currency retrieve_num=10 is denominated in.
                                #
                                # THE FLOOR IS A FOURTH SOURCE, NOT A CHANNEL. Two earlier drafts
                                # tried to push these units into the channel lists and each broke a
                                # different pre-existing invariant on question one: appending them
                                # all to `_sd` broke "material must cover exactly the claimed
                                # documents", and routing each to the channel that claimed its
                                # document broke "sparse material left its pool" - a CE-ranked unit
                                # is not in the BM25 head and never will be. Both asserts were
                                # right. Keeping the floor in its own list leaves all three
                                # channel invariants untouched: NOT ONE existing assert is modified
                                # by this patch, which is the property D2 says to insist on.
                                _cld = {sess_id[i] for i in _fb["d"] + _fb["s"] + _fb["g"]}
                                _must = [i for i in list(ce_rank)[:CEFLOOR] if sess_id[i] in _cld]
                                _have = set(_sd) | set(_ss) | set(_sg)
                                _MR_MUST[_nm] = _must
                                _MR_FLOOR[_nm] = [i for i in _must if i not in _have]
                                # NOT a gate here - `_sd` was just built to contain these, so a
                                # check at this line is a tautology and is deliberately omitted.
                                # The live check is downstream, after merge and cap, where the
                                # budget governor is the thing that can still drop them.
                            # The old form was `set(_sd) <= set(_PD) | set(_fb["d"])`. _sd was built
                            # ONLY from those two, so it read A <= A - and the union on the right is
                            # the bug itself, written into the assert meant to catch it.
                            assert set(_sd) <= set(_PD), ("dense material left its pool", _nm)
                            assert set(_ss) <= set(_PS), ("sparse material left its pool", _nm)
                            assert set(_sg) <= set(_PG), ("graph material left its pool", _nm)
                            for _ck3, _ml, _fl, _al in (("d", _sd, _fb["d"], ALPHA_D),
                                                        ("s", _ss, _fb["s"], ALPHA_S),
                                                        ("g", _sg, _fb["g"], ALPHA_G)):
                                _cn2 = {}
                                for _i2 in _ml:
                                    _cn2[sess_id[_i2]] = _cn2.get(sess_id[_i2], 0) + 1
                                if not MRNOCAP:
                                    assert max(list(_cn2.values()) or [0]) <= max(_al, 1), \
                                        ("depth exceeds alpha inside one document", _nm)
                                else:
                                    # The REPLACEMENT invariant. Not weaker: it forbids the two
                                    # wrong implementations of "no per-document cap" - walking the
                                    # DEEPENED head instead of the fixed k*alpha one, and admitting
                                    # a unit whose document was never claimed.
                                    _hd2 = {"d": (ce_rank, ALPHA_D, _target),
                                            "s": (sparse, ALPHA_S, max(_fs, 1)),
                                            "g": (grank_s, ALPHA_G, max(_fg, 1))}[_ck3]
                                    _allow = set(list(_hd2[0])[:max(_hd2[1], 1) * _hd2[2]]) | set(_fl)
                                    assert set(_ml) <= _allow, \
                                        ("material outside the k*alpha head + reps", _nm, _ck3)
                                assert set(_fl) <= set(_ml), ("representative must be material", _nm)
                                assert {sess_id[i] for i in _ml} == {sess_id[j] for j in _fl}, \
                                    ("material must cover exactly the claimed documents", _nm)
                        else:
                            _sd = _mr_flat(_PD, sess_id, _fb["d"])
                            _ss = _mr_flat(_PS, sess_id, _fb["s"])
                            _sg = _mr_flat(_PG, sess_id, _fb["g"])
                    else:
                        _sd = _chan_seed(ce_rank, sess_id, _fb["d"], _fd, MR_ALPHA, _kd)
                        _ss = _chan_seed(sparse, sess_id, _fb["s"], _fs, MR_ALPHA, _ks)
                        _sg = _chan_seed(grank_s, sess_id, _fb["g"], _fg, MR_ALPHA, _kg)
                    _MR_SEED[_nm] = sorted(set(_sd + _ss + _sg + _MR_FLOOR.get(_nm, [])))
                    # The sparse face: this channel's floor units and its merge material, the same
                    # objects _chan_seed returned. Membership is not re-decided at the feed.
                    _MR_SFACE[_nm] = set(_ss) | set(_fb["s"])
                    # The unit budget needed to hold this arm's documents. Equals U wherever the
                    # anchors do not collide, so everything downstream is unchanged there.
                    _MR_UEFF[_nm] = max(U, sum(len(_fb[k]) for k in ("d", "s", "g")))
                    _MR_DOCS[_nm] = _mr_docs(_fb, sess_id)
                    _MR_SPLIT[_nm] = {
                        "ndoc": {k: len({sess_id[i] for i in _fb[k]}) for k in ("d", "s", "g")},
                        "nunit": {k: len(_fb[k]) for k in ("d", "s", "g")},
                        "nseed": {"d": len(_sd), "s": len(_ss), "g": len(_sg)},
                        "nskip": {"d": len(_kd), "s": len(_ks), "g": len(_kg)},
                        "redist": _rd,
                        "anchor_collide": int(MR_ALPHA and sess_id[ce_rank[0]] == sess_id[ce_rank[1]]),
                        "short": min(_fd + _fs + _fg, NPAR)
                                 - len({sess_id[i] for i in _fb["d"] + _fb["s"] + _fb["g"]})}
                    if MRDOCS in ("global", "fill"):
                        # E7: independent asserts. The transplant assert below checks the selection
                        # against _fb, so it FOLLOWS whatever _fb says and cannot catch a wrong
                        # claim - the same alarm-disabling shape this operator was already burned by.
                        # These three are structural and cannot fail on a legitimate store; the
                        # document COUNT is deliberately NOT asserted, because a store whose channels
                        # span fewer than ten parents is a real condition and must be read off
                        # mr_split["short"] rather than kill a paid run.
                        # Unit-disjointness is a `global` invariant only. Under `fill` two channels
                        # deliberately hold two DIFFERENT units of one shared document - each keeps
                        # its own representative of it, which is the whole point. The document count
                        # is data-dependent (a store may not have ten parents) so it is recorded in
                        # mr_split["short"] and read as a witness rather than asserted here.
                        if MRDOCS == "global":
                            assert not (set(_bd) & set(_bs)) and not (set(_bd) & set(_bg)) \
                                and not (set(_bs) & set(_bg)), ("channels must be unit-disjoint", _nm)
                        assert set(_bs) <= set(sparse) and set(_bg) <= set(grank_s), \
                            ("a floor unit must come from its OWN channel", _nm)
                        # The budget may exceed U by at most the anchors' collision cost: two
                        # unconditional anchors in one document buy one document with two units.
                        if MRDOCS == "global":
                            assert len(_bd) + len(_bs) + len(_bg) <= U + 1, ("unit budget", _nm)
                        # NOT asserted: that a channel covers its document quota. Whether ce_rank's
                        # k1 candidates even span f_d distinct chapters is a property of the data -
                        # flr_8_0_2 asks dense for eight, and a question whose dense retrieval
                        # concentrates in six chapters cannot supply them. An assert there kills a
                        # paid run over a data property; mr_split["ndoc"] already records it, and
                        # that is the witness to read.
                elif DOCFLOOR:
                    # Each channel reserves DOCUMENTS, by its own score. The graph branch already
                    # selects at document level (grank_s comes from a reach run over documents);
                    # this gives dense and sparse the same treatment, each by its own criterion.
                    _fb = {"d": _doc_unique(ce_rank, sess_id, _fd, must=2),
                           "s": _doc_unique(sparse, sess_id, _fs),
                           "g": _doc_unique(grank_s, sess_id, _fg), "x": list(ce_rank)}
                else:
                    # QSCALE. The quota list at the top of this block is sixteen triples that each
                    # sum to TEN, because ten units is the protocol delivery. Handing those same
                    # absolute counts to floor_fill at U=30 leaves twenty slots unreserved, and
                    # floor_fill fills them from `x` = the whole cross-encoder ranking -- so raising
                    # U did not widen retrieval, it read further down one ranking, and sparse and
                    # graph fell from 30% of the delivery to 10%. Apportion U by the SAME ratio
                    # instead. Both numbers are inherited: the ratio from the quota list, the
                    # multiplier from U = retrieve_num x alpha, so QS is alpha. At the protocol
                    # U=10, QS is 1 and this block is the identity -- verified byte-for-byte.
                    _fb = {"d": list(ce_rank)[:_fq[0]], "s": list(sparse)[:_fq[1]],
                           "g": list(grank_s)[:_fq[2]], "x": list(ce_rank)}
                _q = ({"d": len(_fb["d"]), "s": len(_fb["s"]), "g": len(_fb["g"]), "x": 0}
                      if _nm in _MR_UEFF else {"d": _fq[0], "s": _fq[1], "g": _fq[2], "x": 0})
                heads[_nm] = MC.floor_fill(_fb, _q, sc.of, _MR_UEFF.get(_nm, U))
                DOC_ARMS.add(_nm)
                _HBM_ARMS[_nm] = set(_fb["s"])
                _HBM_ARMS[_nm + "_full"] = set(_fb["s"])
                if (_fd, _fs, _fg) == (4, 3, 3):
                    if gph:
                        sc.score(gph)
                    _gru = sc.rank(gph)
                    _fbu = {"d": list(ce_rank)[:4], "s": list(sparse)[:3],
                            "g": list(_gru)[:3], "x": list(ce_rank)}
                    heads["flru_4_3_3"] = MC.floor_fill(
                        _fbu, {"d": 4, "s": 3, "g": 3, "x": 0}, sc.of, U)
                    _fbb = {"d": list(ce_rank)[:4], "s": list(sparse)[:2],
                            "g": list(grank_s)[:2], "x": list(ce_rank)}
                    _sel8 = MC.floor_fill(_fbb, {"d": 4, "s": 2, "g": 2, "x": 0}, sc.of, 8)
                    _s8 = set(_sel8)
                    _bc = {}
                    for _bi in _sel8:
                        for _be in (r_ents[_bi] - hubs):
                            _bidf = idf.get(_be, 0.0)
                            for _bj in ef.get(_be, ()):
                                if _bj not in _s8:
                                    _sc2 = (_bidf, bs[_bj])
                                    if _bj not in _bc or _sc2 > _bc[_bj]:
                                        _bc[_bj] = _sc2
                    _br = sorted(_bc, key=lambda j: (-_bc[j][0], -_bc[j][1], j))[:2]
                    heads["flrb_4_3_3"] = (_sel8 + _br
                                           + [i for i in ce_rank if i not in _s8 and i not in _br])
                    assert len(set(_sel8 + _br)) == len(_sel8) + len(_br), "flrb overlap"
                    DOC_ARMS.add("flrb_4_3_3")
                    _fb3 = {"d": list(ce_rank)[:3], "s": list(sparse)[:2],
                            "g": list(grank_s)[:2], "x": list(ce_rank)}
                    _sel7 = MC.floor_fill(_fb3, {"d": 3, "s": 2, "g": 2, "x": 0}, sc.of, 7)
                    _s7 = set(_sel7)
                    _bc2 = {}
                    for _bi in _sel7:
                        for _be in (r_ents[_bi] - hubs):
                            _bidf2 = idf.get(_be, 0.0)
                            for _bj in ef.get(_be, ()):
                                if _bj not in _s7:
                                    _bc2[_bj] = _bc2.get(_bj, 0.0) + _bidf2
                    _br2 = sorted(_bc2, key=lambda j: (-_bc2[j], -bs[j], j))[:3]
                    heads["flrb2_4_3_3"] = (_sel7 + _br2
                                            + [i for i in ce_rank
                                               if i not in _s7 and i not in _br2])
                    DOC_ARMS.add("flrb2_4_3_3")
                if (_fd, _fs, _fg) in ((4, 3, 3), (5, 3, 2), (6, 2, 2)):
                    _gatt = list(dict.fromkeys(list(anchors_a) + list(att_s) + list(grank_s)))
                    sc.score(_gatt[:_fg])
                    _fba = {"d": list(ce_rank)[:_fd], "s": list(sparse)[:_fs],
                            "g": _gatt[:_fg], "x": list(ce_rank)}
                    _na = "flra_%d_%d_%d" % (_fd, _fs, _fg)
                    heads[_na] = MC.floor_fill(_fba, {"d": _fd, "s": _fs, "g": _fg, "x": 0},
                                               sc.of, U)
                    DOC_ARMS.add(_na)
                # XQUOTA: the x-channel arms used to exist only at (4,3,3), which made any
                # gate whose branches differ in quota ALSO differ in whether they carry sibling
                # expansion. Admitting the two quotas the gate uses removes that confound. (4,3,3)
                # is listed first and its construction is unchanged, so its arms stay byte-identical.
                if (_fd, _fs, _fg) in _XQUOTAS:
                    _sel0 = list(heads[_nm])
                    _cr0 = set(ce_rank)
                    _expF = []
                    for _i0 in _sel0:
                        for _j0 in corpus.children_of.get(sess_id[_i0], ()):
                            if _j0 not in _expF:
                                _expF.append(_j0)
                    _expS = []
                    for _i0 in _sel0:
                        for _j0 in corpus.children_of.get(sess_id[_i0], ()):
                            if abs(pos_of.get(_j0, 0) - pos_of.get(_i0, 0)) <= 1 and _j0 not in _expS:
                                _expS.append(_j0)

                    def _mk_x(_exp):
                        _newu = [j for j in _exp if j not in _cr0]
                        sc.score([j for j in _newu if j not in sc._s])
                        _newu = sorted(set(_newu), key=lambda j: (-sc.of(j), j))
                        return list(ce_rank) + _newu, len(_newu)

                    _xF, _nF = _mk_x(_expF)
                    _xS, _nS = _mk_x(_expS)
                    _dtail = [i for i in MC.dense_candidates(corpus, qv, K1 + max(_nF, 1))
                              if i not in _cr0][:_nF]
                    sc.score([j for j in _dtail if j not in sc._s])
                    _xK = list(ce_rank) + sorted(set(_dtail), key=lambda j: (-sc.of(j), j))
                    # QSCALE reaches the SHIPPED arm here. flrs_4_3_3 is assembled from this
                    # dict and the literal quota below, not from the _fb/_q pair 130 lines above
                    # that names flr_4_3_3 -- patching only that one left flrs_4_3_3 at 94% of its
                    # unpatched delivery (Jaccard 0.938 on a DRY U=30 pair) and attprio_sess at
                    # 100%. Same _qs as above: max(1, U // 10), which is alpha.
                    _fb0 = {"d": list(ce_rank)[:_fq[0]], "s": list(sparse)[:_fq[1]],
                            "g": list(grank_s)[:_fq[2]]}
                    _qsuf = "_%d_%d_%d" % (_fd, _fs, _fg)
                    # flr+ : EVERY seed is kept and delivered as its +-1 span inside its own parent;
                    # overlapping spans merge. The cap of ten is on MERGED UNITS, and merging only
                    # ever reduces that count, so it holds by construction and no seed is dropped.
                    # Distinct parents are preserved or grow. Raw units and tokens grow, which is
                    # the cost this arm pays and which the caller must report.
                    _pfed, _pspan = [], []
                    for _s0 in heads[_nm]:
                        _par0, _p0 = sess_id[_s0], pos_of.get(_s0)
                        _sp = [at[(_par0, _q)] for _q in (_p0 - 1, _p0, _p0 + 1)
                               if _p0 is not None and (_par0, _q) in at] or [_s0]
                        _pspan.append(_sp)
                        for _j in _sp:
                            if _j not in _pfed:
                                _pfed.append(_j)
                    # count merged units: spans that share a unit are one merged unit
                    _grp = []
                    for _sp in _pspan:
                        _hit = [g for g in _grp if set(g) & set(_sp)]
                        if _hit:
                            _hit[0].update(_sp)
                            for _g in _hit[1:]:
                                _hit[0].update(_g)
                                _grp.remove(_g)
                        else:
                            _grp.append(set(_sp))
                    # pf28, the ONLY change from pf27: the flr+ arm is skipped when the hierarchical
                    # operator is on. Its invariant -- ten merged units, no seed dropped -- is
                    # broken by the operator, which grows the seed set past U, and the assert then
                    # kills the whole question BEFORE flrs/flrx/flrK are built two lines below.
                    # Those three do not depend on _pfed or _grp (they are built from _fb0 and the
                    # _xF/_xS/_xK sets computed above), and the transplant asserts never mention
                    # flrp, so skipping it changes nothing else. With MR_ALPHA=0 -- every board run --
                    # this branch is the pf27 code exactly.
                    if not MR_ALPHA:
                        assert len(_grp) <= U, ("flr+ produced %d merged units, over the cap" % len(_grp))
                        heads["flrp" + _qsuf] = sorted(set(_pfed), key=lambda i: (-sc.of(i), i))
                    # Register the real unit budget, or line 2104 truncates the pool back to U and
                    # takes the LOWEST-scoring units first - the context this arm exists to buy.
                    # The token ceiling ROW_CAP still binds downstream via _mr_cap / _mr_gov.
                        _MR_UEFF["flrp" + _qsuf] = max(U, len(set(_pfed)))
                        DOC_ARMS.add("flrp" + _qsuf)
                    for _tag, _xu in (("flrx", _xF), ("flrs", _xS), ("flrK", _xK)):
                        _fb2 = dict(_fb0)
                        _fb2["x"] = _xu
                        heads[_tag + _qsuf] = MC.floor_fill(
                            _fb2, {"d": _fq[0], "s": _fq[1], "g": _fq[2], "x": 0}, sc.of, U)
                        DOC_ARMS.add(_tag + _qsuf)
                    if MR_ALPHA:
                        # pf29, the ONLY change from pf28. The operator's material was registered
                        # under "flr_%d_%d_%d" alone, so the SIBLING arms -- the ones the board
                        # actually scores -- reached the feed with `_MR_SEED.get(mm, ())` empty.
                        # `_mr_merge_one` merges `list(base) + list(seeds)`, so they still received
                        # the positional window on their own selection, and NONE of the merge
                        # material `_chan_seed` exists to gather: this channel's own strong
                        # candidates lying inside the documents this arm already brought. That is
                        # the step the design turns on, and no scored arm has ever had it.
                        #
                        # Material comes from the arm's OWN selection, not from a floor list. The
                        # documents that matter are the ones this arm brought, and floor_fill has
                        # already ordered them by the shared score.
                        #
                        # `_MR_DOCS` is set alongside it and not left to default: `_mr_gov` ranks
                        # the feed's documents by it, and an empty map makes every document tie, so
                        # the governor cuts by store position instead of by priority. That failure
                        # is silent and was observed on this harness.
                        #
                        # `_MR_UEFF` is deliberately NOT set. It is the entry that lifts the pool
                        # above U at lines 2499 and 2773, and the ten-slot protocol is the thing
                        # being kept: selection stays at ten, only the feed grows, and `_win_merge`
                        # emits same-document material as ONE passage, so the passage count holds.
                        for _tg in ("flrx", "flrs", "flrK"):
                            _an = _tg + _qsuf
                            if _an not in heads:
                                continue
                            _selA = list(heads[_an])
                            _sdA = _chan_seed(ce_rank, sess_id, _selA, U, MR_ALPHA)
                            _ssA = _chan_seed(sparse, sess_id, _selA, U, MR_ALPHA)
                            _sgA = _chan_seed(grank_s, sess_id, _selA, U, MR_ALPHA)
                            _MR_SEED[_an] = sorted(set(_sdA + _ssA + _sgA))
                            _MR_SFACE[_an] = set(_ssA)
                            _MR_DOCS[_an] = _mr_docs({"d": _selA, "s": [], "g": []}, sess_id)
                    # TRANSPLANT ASSERTS, every question, fail loudly: floors and anchors must hold
                    # against the SAME source objects the attach-priority arms consume (ce_rank,
                    # sparse=BM25, grank_s). A violation here is an implementation break, not noise.
                    for _cn in ("flr" + _qsuf, "flrx" + _qsuf, "flrs" + _qsuf, "flrK" + _qsuf):
                        _pp = set(heads[_cn])
                        _sfb = _fb if _cn == ("flr" + _qsuf) else _fb0
                        for _bl, _bf in ((_sfb["d"], _fq[0] if _LT10 else _fd), (_sfb["s"], _fq[1] if _LT10 else _fs), (_sfb["g"], _fq[2] if _LT10 else _fg)):
                            # D6: _bf is the STATIC quota while _bl grew under fill, so this
                            # checked 4 of dense's 9 and the five units the fill produced sat
                            # outside the alarm. On a MR_ALPHA arm the branch list IS the floor.
                            _need = len(set(_bl)) if (MR_ALPHA and _cn == "flr_4_3_3") else \
                                min(len(set(_bl)), _bf)
                            assert len(_pp & set(_bl)) >= _need, ("floor violated", _cn)
                        assert set(ce_rank[:2]) <= _pp, ("anchors lost", _cn)
                    # flrm Layer T: frozen flr selection + per-branch OWN-ORDER tails + sparse rest.
                    _selm = list(heads["flr_4_3_3"])
                    _tsrc = list(ce_rank)[4:24] + list(sparse)[3:13] + list(grank_s)[3:13]
                    _tailm = [i for i in dict.fromkeys(_tsrc) if i not in set(_selm)]
                    heads["flrm_4_3_3"] = _selm + _tailm + [i for i in srest if i not in set(_selm)]
                    DOC_ARMS.add("flrm_4_3_3")
                    heads["flrK_tok_4_3_3"] = _selm + [i for i in ce_rank if i not in set(_selm)]
                    DOC_ARMS.add("flrK_tok_4_3_3")
                    heads["flrm2_4_3_3"] = list(heads["flrm_4_3_3"])
                    DOC_ARMS.add("flrm2_4_3_3")
            # Three sources, decomposed so each one's contribution is separately visible:
            #   base  = graph + dense          _s = + sparse tail          _full = + hierarchy
            # _full uses MC.merge, the same operator and the same arguments as ladder12.py:447, which
            # selects PARENTS by their best child and expands inside them - a session-position window
            # cannot do that step, and that window is what this file used to do.
            # BM25 is kept in the ranking it would occupy but is no longer reported as its own tier:
            # measured over 300 questions it changed the selected pool on none of them, for any arm.
            # D: the container is a graph node OR a hierarchy parent, never both. An arm whose graph
            # branch already collapsed to sessions gets a flat merge, because a hierarchical one would
            # re-expand the same sessions it just selected. Measured on ten arms with no exception:
            # every round-level arm gains from the hierarchy (+0.014 to +0.077) and every document-level
            # arm gains nothing or loses (+0.007 to -0.067).
            def _doc_level_arm(nm):
                return nm in DOC_ARMS
            for mname in list(heads):
                _hs = list(heads[mname]) + srest
                _hm, _ = MC.merge(_hs, corpus, U * 3, hierarchical=FORCE_H or (not _doc_level_arm(mname)),
                                  n_parents=U, alpha=2, hops=None, scorer=sc)
                heads[mname + "_full"] = _hm
            pool = {mm: list(dict.fromkeys(heads[mm]))[:_MR_UEFF.get(mm, U)] for mm in METHODS}

            # The extraction window stays the union of every selected arm's pool, because a unit that is
            # fed but never extracted from cannot be marked [CURRENT] and the views would render wrong.
            sel = list(dict.fromkeys([i for mm in METHODS for i in pool[mm]]))

            _exc = {}

            def extract_over(units):
                """Facts from one extraction over `units`. Cached on the unit set, so arms whose pools
                coincide - most of them - share a call."""
                key = frozenset(units)
                if key in _exc:
                    return _exc[key]
                if EXTRACT_STUB:
                    # Each passage gets its share of the fact budget, so lengthening the passage list
                    # thins what is extracted from every passage in it - the one coupling that makes
                    # the extraction window part of the method rather than a detail of the harness.
                    per = max(1, STUB_BUDGET // max(len(units), 1))
                    fx = []
                    for i in units:
                        w = re.sub(r"[^a-z ]", " ", rtext[i].lower()).split()
                        for t in range(per):
                            if 3 * t >= len(w):
                                break
                            fx.append((slotkey(w[3 * t]),
                                       w[3 * t] + ": " + " ".join(w[3 * t + 1:3 * t + 4]),
                                       rdate[i], i))
                        if len(fx) >= STUB_BUDGET:
                            break
                    fx = fx[:STUB_BUDGET]
                    _exc[key] = fx
                    return fx
                blk = "\n".join("[%d] %s" % (j, rtext[i][:600]) for j, i in enumerate(units))
                raw = "" if NO_EXTRACT else (extract_cached_only(
                    "From the numbered memory passages below, extract facts ABOUT THE USER relevant "
                    "to this question: \"" + q + "\"\nOutput one per line exactly as:  "
                    "attribute :: value :: passage_number\nInclude superseded/older values too. "
                    "Max 25 lines.\n\n" + blk + "\n\nFacts:",
                    "lmeo|extract|r%dq%d|n%d" % (ROW, qi, len(units))) if DRY else extract_call(
                    "From the numbered memory passages below, extract facts ABOUT THE USER relevant "
                    "to this question: \"" + q + "\"\nOutput one per line exactly as:  "
                    "attribute :: value :: passage_number\nInclude superseded/older values too. "
                    "Max 25 lines.\n\n" + blk + "\n\nFacts:",
                    "lmeo|extract|r%dq%d|n%d" % (ROW, qi, len(units))))
                fx = []
                for line in _THINK.sub("", raw or "").split("\n"):
                    pr = [z.strip() for z in line.split("::")]
                    if len(pr) >= 3 and pr[0] and pr[1]:
                        try:
                            jj = int(re.sub(r"[^0-9]", "", pr[2])[:3])
                        except Exception:
                            continue
                        if 0 <= jj < len(units):
                            fx.append((slotkey(pr[0]), pr[0] + ": " + pr[1], rdate[units[jj]], units[jj]))
                _exc[key] = fx
                return fx

            def resolve(fx):
                """Facts -> the stale set and the current-round set that mark a view."""
                nw = {}
                for fi, (sk, txt, ts, ri) in enumerate(fx):
                    if sk and (sk not in nw or ts > fx[nw[sk]][2]):
                        nw[sk] = fi
                st, pr_ = set(), []
                for fi, (sk, txt, ts, ri) in enumerate(fx):
                    wi = nw.get(sk)
                    if sk and wi is not None and wi != fi and fx[wi][3] != ri:
                        pr_.append((fi, wi, ts, fx[wi][2]))
                        if fx[wi][2] > ts:
                            st.add(ri)
                return st, {fx[i][3] for i in nw.values()}, pr_
            # The ROUTER's evidence, however, must not depend on how many arms are in the run. Measured
            # over the same 300 questions, the routed view differed on 14-22 of them between runs that
            # differed only in their arm list. So the router reads suppressions restricted to a canonical
            # window - dense's top-U plus the graph's - which exists whatever is being scored.
            router_window = set(dict.fromkeys(ce_rank[:U] + gph[:U]))
            # ladder12:731-734. When the store declares its own parent and clock the router reads
            # suppressions from the store, not from an LLM extractor that returns nothing here.
            # The pool is dense-then-graph at the SAME budget as the views, deduped on the raw
            # index and cut at U - `fill_keepall(ce_rank + grank, u)` - so rho's denominator is
            # the same object ladder12 divides by and the two numbers are comparable.
            _RB_HITS = [0]
            _SUPP, _SUPNH = None, None
            if _STRUCTURAL:
                _sp, _sn = [], set()
                for _i in list(ce_rank) + list(gph):
                    if _i in _sn:
                        continue
                    _sn.add(_i)
                    _sp.append(_i)
                    if len(_sp) >= U:
                        break
                _SUPP = _sp
                _SUPNH = MC.newest_per_parent(_sp, corpus)
                router_window = set(_sp)
            # With per-arm resolution and a canonical router window nothing reads the shared
            # extraction any more, so it is not paid for.
            # ONE code path for the extraction. The shared window used to be extracted by an inline
            # copy of extract_over/resolve, which meant the disk cache and the deterministic stand-in
            # covered the per-arm path only - and a test of the un-fixed harness passed because under
            # DRY the inline copy simply produced no facts at all.
            _skip_global = PER_ARM and ROUTER_CANON
            facts = [] if _skip_global else extract_over(sel)
            stale, cur_rounds, pairs = resolve(facts)
            _rc = {}

            def _rgen(p, nn=16):
                if "i" not in _rc:
                    _rc["i"] = g.openai(p, model=MODEL, max_tokens=32, temperature=0.0,
                                        system=EXTRACT_SYSTEM_MESSAGE, seed=SEED,
                                        note="lmeo|router|r%dq%d" % (ROW, qi))
                return _rc["i"]
            # suppressions confined to the canonical window; `pairs` itself still drives the marking
            if _STRUCTURAL:
                # ladder12:733-734 verbatim: a unit is suppressed when its parent's newest unit
                # inside this same pool is some other unit. No generator call, nothing to return
                # empty, and the clock is the profile's order rather than an inferred timestamp.
                _rp = [(_i, _SUPNH[sess_id[_i]], _ORD[_i], _ORD[_SUPNH[sess_id[_i]]])
                       for _i in _SUPP
                       if sess_id[_i] is not None and _SUPNH.get(sess_id[_i]) != _i]
                _rf = [(sess_id[_i], rtext[_i], _ORD[_i], _i) for _i in _SUPP]
                _RB_HITS[0] += 1
            elif ROUTER_CANON:
                _rf = extract_over(sorted(router_window))
                _rp = resolve(_rf)[2]
            else:
                _rp = [pr for pr in pairs
                       if facts[pr[0]][3] in router_window and facts[pr[1]][3] in router_window]
                _rf = [f for f in facts if f[3] in router_window]
            try:
                if DRY or NO_EXTRACT:
                    raise RuntimeError("dry")
                view, _ = UR.route(q, _rgen, slots=[f[0] for f in _rf],
                                   values=[f[1] for f in _rf], suppressions=_rp)
            except Exception:
                view = MC.PRUNE
            pick = "prune_annot" if MC.PRUNE in str(view).upper() else "composite"
            # The last four attempts all failed identically: the patch landed in a branch that
            # never runs and every offline check passed on dead code. This dies on question 1.
            assert (not _STRUCTURAL) or _RB_HITS[0] > 0, \
                ("STRUCTURAL selected but the router read the extractor", ROW, qi)
            assert (not _STRUCTURAL) or _SG_HITS[0] > 0, \
                ("STRUCTURAL selected but the g-face used the generic document reach", ROW, qi)

            ctxs = {}
            _FEDIDS = {}
            _CTXHEAD = {}
            _SR_HITS = [0]
            _SR_SUP = {}
            _NR_N = {}
            _UNITS = {}   # arm -> {view: [unit ids]} and the parent of each

            def _srender(mm_, fedk_, fedp_, pool_, ucap_):
                """The structural views, defined ONCE and called from every render site.

                Previously each site carried its own copy of this logic and two of the three copies
                sat in branches that never execute, so the paid run rendered the plain views while
                every offline check passed. One function, three call sites, one counter.

                ladder12.py:313-322 fill()          - substitute to the slot's newest, dedupe on the
                                                      SUBSTITUTED index, refill to ucap DISTINCT slots
                ladder12.py:666-674 fill_keepall()  - top ucap AS RANKED, no substitution
                ladder12.py:709                     - CURRENT over the set actually rendered
                ladder12.py:704-711                 - print the order; prune ASC, composite DESC
                """
                _src = list(fedk_) + list(fedp_) + list(pool_) + list(ce_rank)
                # THE KEEP-ALL SOURCE IS NOT _src. ladder12:672 fill_keepall(head, u) walks
                # `head + ce_rank`; feeding it `fedk_` first seeds it with `keep`, which is the
                # head with the stale units ALREADY REMOVED (lme_pf2.py:2180-2187), so resolved
                # units fill ucap and the superseded revisions composite exists to show are pushed
                # out. Measured against ladder12 on the same 100 questions: composite 0.100 vs
                # 0.200, paired 4-14, p=0.031, while prune - which substitutes to the slot's newest
                # regardless - sat at parity. `fedp_` is the composite-side delivery on the plain
                # path (:2313-2315), so the operator still reaches this view by the same door.
                _srcK = list(fedp_) + list(ce_rank)
                _seen, _pr = set(), []
                for _i in _src:
                    _k = _NW_ROW.get(sess_id[_i], _i)
                    if _k in _seen:
                        continue
                    _seen.add(_k)
                    _pr.append(_k)
                    if len(_pr) >= ucap_:
                        break
                # T8.  prune_norefill walks the SAME source in the SAME order and consumes
                # exactly ucap_ source units instead of collecting ucap_ distinct ones, so a
                # collapse leaves a HOLE.  Same prefix, same order, so _nr is a subset of _pr.
                _nr = []
                if NOREFILL:
                    _rs, _rt = set(), []
                    for _i in _src:
                        if _i in _rs:
                            continue
                        _rs.add(_i)
                        _rt.append(_i)
                        if len(_rt) >= ucap_:
                            break
                    _s3 = set()
                    for _i in _rt:
                        _k = _NW_ROW.get(sess_id[_i], _i)
                        if _k in _s3:
                            continue
                        _s3.add(_k)
                        _nr.append(_k)
                # slotdiv: prune's REFILL with prune's SUBSTITUTION removed.  Dedupe on the slot
                # and keep the unit that was RETRIEVED, so the budget buys ucap distinct slots
                # without any claim about which of a slot's values is live.  Nothing here reads
                # _NW_ROW, so this view is computable on a store no profile names.
                _sd = []
                if NOREFILL:
                    _s4 = set()
                    for _i in _src:
                        _sk = sess_id[_i] if sess_id[_i] is not None else ("#u%d" % _i)
                        if _sk in _s4:
                            continue
                        _s4.add(_sk)
                        _sd.append(_i)
                        if len(_sd) >= ucap_:
                            break
                _seen2, _ka = set(), []
                for _i in _srcK:
                    if _i in _seen2:
                        continue
                    _seen2.add(_i)
                    _ka.append(_i)
                    if len(_ka) >= ucap_:
                        break
                _curk = set(MC.newest_per_parent(_ka, corpus).values())
                _SR_SUP[mm_] = len(_ka) - len(_curk)
                ctxs[mm_ + "|prune_annot"] = "\n".join(
                    "[%s] [CURRENT] %s" % (_ORD[i], rtext[i])
                    for i in sorted(_pr, key=lambda j: _ORD[j]))
                ctxs[mm_ + "|composite"] = "\n".join(
                    (("[%s] [CURRENT] %s" % (_ORD[i], rtext[i])) if i in _curk
                     else ("[%s] %s" % (_ORD[i], rtext[i])))
                    for i in sorted(_ka, key=lambda j: _ORD[j], reverse=True))
                if SUPMARK:
                    # the SAME units, the SAME order, one label changed: bare -> [SUPERSEDED].
                    ctxs[mm_ + "|comp_sup"] = "\n".join(
                        (("[%s] [CURRENT] %s" % (_ORD[i], rtext[i])) if i in _curk
                         else ("[%s] [SUPERSEDED] %s" % (_ORD[i], rtext[i])))
                        for i in sorted(_ka, key=lambda j: _ORD[j], reverse=True))
                # prune sorts ascending and composite descending, so two or more units CANNOT
                # render the same string. When they did, 30 paid questions had silently taken the
                # plain path.
                if NOREFILL:
                    ctxs[mm_ + "|prune_norefill"] = "\n".join(
                        "[%s] [CURRENT] %s" % (_ORD[i], rtext[i])
                        for i in sorted(_nr, key=lambda j: _ORD[j]))
                    ctxs[mm_ + "|slotdiv"] = "\n".join(
                        "[%s] %s" % (_ORD[i], rtext[i])
                        for i in sorted(_sd, key=lambda j: _ORD[j]))
                    assert set(_nr) <= set(_pr), \
                        ("no-refill is not a subset of prune", mm_, len(_nr), len(_pr))
                    _NR_N[mm_] = [len(_pr), len(_nr), len(_sd)]
                    _pof = lambda L: [(sess_id[i] if sess_id[i] is not None else
                                       ("#u%d" % i)) for i in L]
                    _UNITS[mm_] = {"prune": sorted(_pr), "norefill": sorted(_nr),
                                   "slotdiv": sorted(_sd),
                                   "prune_par": _pof(sorted(_pr)),
                                   "slotdiv_par": _pof(sorted(_sd))}
                assert len(_ka) < 2 or ctxs[mm_ + "|prune_annot"] != ctxs[mm_ + "|composite"], \
                    ("structural views identical", mm_, len(_pr), len(_ka))
                _SR_HITS[0] += 1
                return True

            for mm in METHODS:
                if _STRUCTURAL:
                    # ladder12.py:709. The resolution key is the corpus parent and the clock is the
                    # corpus order, so no LLM is consulted and nothing can come back empty. On
                    # FactConsolidation the extractor returned n_facts=0 on 100/100 questions, which
                    # is what left prune_annot and composite rendering the same text.
                    _cur = set(MC.newest_per_parent(pool[mm], corpus).values())
                    _st = set()
                elif PER_ARM:
                    _st, _cur, _ = resolve(extract_over(pool[mm]))
                    if os.environ.get("NO_ST", "") not in ("", "0"):
                        _st = set()          # DIAGNOSTIC: keep _cur from the extraction, drop only its stale set
                else:
                    _st, _cur = stale, cur_rounds
                p_ = pool[mm]
                _ucap = _MR_UEFF.get(mm, U)
                keep = []
                for i in heads[mm]:
                    if i in _st or i in keep:
                        continue
                    keep.append(i)
                    if len(keep) >= _ucap:
                        break
                if len(keep) < _ucap:
                    keep += [i for i in p_ if i not in keep][:_ucap - len(keep)]
                # T8, plain path.  `keep` walks the whole head skipping stale units and then
                # refills from the pool.  keep_nr walks only the head's first _ucap DISTINCT
                # entries - what composite sees - and drops the stale ones without replacement.
                # keep_sd takes one unit per parent, best-ranked first, and never consults _st.
                keep_nr, keep_sd = [], []
                if NOREFILL:
                    _hd, _hs = [], set()
                    for i in heads[mm]:
                        if i in _hs:
                            continue
                        _hs.add(i)
                        _hd.append(i)
                        if len(_hd) >= _ucap:
                            break
                    keep_nr = [i for i in _hd if i not in _st]
                    _sd4 = set()
                    for i in list(heads[mm]) + list(p_):
                        _sk = sess_id[i] if sess_id[i] is not None else ("#u%d" % i)
                        if _sk in _sd4:
                            continue
                        _sd4.add(_sk)
                        keep_sd.append(i)
                        if len(keep_sd) >= _ucap:
                            break
                if mm == "flrm2_4_3_3" and (mm + "|prune_annot") not in ctxs:
                    _fx2 = extract_over(p_)
                    _pr2 = resolve(_fx2)[2]
                    _wr = {}
                    for _fi, _wi, _ts2, _wts in _pr2:
                        if _wts > _ts2:
                            _wr.setdefault(_fx2[_fi][3], set()).add(_fx2[_wi][3])
                    _h2 = heads[mm]
                    _guar = set(_h2[:U])
                    keep2 = []
                    for i in _h2:
                        if i in keep2:
                            continue
                        if i in _st:
                            _ws = _wr.get(i, set())
                            if _ws and (_ws & (_guar | set(keep2))):
                                continue
                        keep2.append(i)
                        if len(keep2) >= U:
                            break
                    if len(keep2) < U:
                        keep2 += [i for i in p_ if i not in keep2][:U - len(keep2)]
                    _sl2 = MC.slices_of(keep2, sess_id, pos_of, at, gap=2, pad=1)
                    _fed2 = []
                    for i in keep2:
                        for j in _sl2.get(i, (i,)):
                            if j not in _fed2 and (j in keep2 or j not in _st):
                                _fed2.append(j)
                    if EXPAND and DSNAME != "longmemeval":
                        _ex = []
                        for i in keep2:
                            for j in range(max(0, i - EXPAND), min(len(rtext), i + EXPAND + 1)):
                                if j not in _ex and (j in keep2 or j not in _st):
                                    _ex.append(j)
                        _fed2 = _ex
                    if _STRUCTURAL and _srender(mm, _fed2, _fed2, p_, _ucap):
                        pass
                    else:
                        ctxs[mm + "|prune_annot"] = "\n".join(
                            (("[CURRENT] " + rtext[i]) if i in _cur else rtext[i]) for i in sorted(_fed2))
                        ctxs[mm + "|composite"] = "\n".join(
                            (("[CURRENT] " + rtext[i]) if i in _cur else rtext[i])
                            for i in sorted(set(_fed2) | set(p_)))
                    continue
                if mm == "flrm_4_3_3" and (mm + "|prune_annot") not in ctxs:
                    _sl = MC.slices_of(keep, sess_id, pos_of, at, gap=2, pad=1)
                    _fed = []
                    for i in keep:
                        for j in _sl.get(i, (i,)):
                            if j not in _fed and j not in _st:
                                _fed.append(j)
                    if _STRUCTURAL and _srender(mm, _fed, _fed, p_, _ucap):
                        pass
                    else:
                        ctxs[mm + "|prune_annot"] = "\n".join(
                            (("[CURRENT] " + rtext[i]) if i in _cur else rtext[i]) for i in sorted(_fed))
                        ctxs[mm + "|composite"] = "\n".join(
                            (("[CURRENT] " + rtext[i]) if i in _cur else rtext[i])
                            for i in sorted(set(_fed) | set(p_)))
                    _nfed = len(_fed)
                    _kt = list(keep) + [i for i in ce_rank if i not in set(keep)][:max(0, _nfed - len(keep))]
                    ctxs["flrK_tok_4_3_3|prune_annot"] = "\n".join(
                        (("[CURRENT] " + rtext[i]) if i in _cur else rtext[i]) for i in sorted(_kt))
                    ctxs["flrK_tok_4_3_3|composite"] = ctxs["flrK_tok_4_3_3|prune_annot"]
                    continue
                if (mm + "|prune_annot") in ctxs:
                    continue
                # HBM_FEED: a unit that occupies this arm's SPARSE face is delivered as its whole
                # disjoint block; dense- and graph-face units stay at the chunk size. Identity when
                # HBM_FEED is 0 or the arm has no sparse floor.
                _fedk = _hbm_expand(keep, _HBM_ARMS.get(mm), _HBM_OF, HBM and HBM_FEED)
                _fedp = _hbm_expand(p_, _HBM_ARMS.get(mm), _HBM_OF, HBM and HBM_FEED)
                # The MR_ALPHA merge/cap/gov chain below rewrites _fedk in four places.  Rather
                # than carry two more copies through it and risk one branch diverging silently -
                # the exact failure that cost 30 paid questions - the plain no-refill legs are
                # DEFINED only where that chain is off, and say so instead of guessing.
                assert not (NOREFILL and MR_ALPHA), \
                    "NOREFILL is not defined on the plain path under MR_ALPHA; run it at MR_ALPHA=0"
                _fednr = (_hbm_expand(keep_nr, _HBM_ARMS.get(mm), _HBM_OF, HBM and HBM_FEED)
                          if NOREFILL else [])
                _fedsd = (_hbm_expand(keep_sd, _HBM_ARMS.get(mm), _HBM_OF, HBM and HBM_FEED)
                          if NOREFILL else [])
                if MR_ALPHA:
                    # Every candidate - the alpha*f oversample, not just the f that were selected -
                    # expands to its +-H_MERGE window inside its own document; the windows are
                    # UNIONED, which is the deduplication; the union is emitted in position order so
                    # same-document pieces read as ONE passage. Nothing is re-ranked afterwards.
                    _seed = _MR_SEED.get(mm, ())
                    # SXMERGE: sparse-face units expand by the geometry they were SCORED on.
                    _sfc = _MR_SFACE.get(mm, ()) if (SXMERGE and (SPARSEX or SXFEED)) else ()
                    _swn = _SX_WIN if _sfc else None
                    # The PROTOCOL base, before the operator touches it. Under MRGATE this set is
                    # what the cap may not evict; without MRGATE it is unused and nothing changes.
                    _prok, _prop = (set(_fedk), set(_fedp)) if MRGATE else (set(), set())
                    _fedk = _mr_merge(_fedk, _seed, _MR_WIN, _st, set(_fedk), _sfc, _swn)
                    _fedp = _mr_merge(_fedp, _seed, _MR_WIN, set(), set(), _sfc, _swn)
                    # The ceiling RAISES rather than truncating, so the tail is cut here, by whole
                    # documents off the end of the document order - the release's rerank[:top_k2].
                    _dpre = len({sess_id[j] for j in _fedp})
                    if MRCAP:
                        # Every document survives; each is trimmed to its share of the protocol
                        # budget. The ceiling becomes structurally unreachable, so the document
                        # invariant cannot be broken by the budget.
                        _sset = set(_MR_SEED.get(mm, ()))
                        _MR_CAPBOUND[0] = 0
                        _fedk, _c1 = _mr_cap(_fedk, _sset, sess_id, _MR_TOK, ROW_CAP, _prok)
                        _fedp, _c2 = _mr_cap(_fedp, _sset, sess_id, _MR_TOK, ROW_CAP, _prop)
                        _MR_CUT[mm] = max(_c1, _c2)
                        _MR_BOUND[mm] = _MR_CAPBOUND[0]
                    else:
                        _fedk = _mr_gov(_fedk, _MR_DOCS.get(mm, ()), sess_id, _MR_TOK, ROW_CAP)
                        _fedp = _mr_gov(_fedp, _MR_DOCS.get(mm, ()), sess_id, _MR_TOK, ROW_CAP)
                    # mr_ndoc is tautological once the quota is filled by construction; the witness
                    # that separates "the lever fired" from "the lever reached the reader" is how
                    # many documents survived the ceiling.
                    _MR_FED[mm] = (_dpre, len({sess_id[j] for j in _fedp}))
                elif SXFEED:
                    # _HBM_ARMS[arm] is already the arm's reserved SPARSE-FLOOR set, populated
                    # unconditionally for every arm in the grid, so the face membership is not
                    # re-decided here - the same set floor_fill used.
                    _sxf = _HBM_ARMS.get(mm) or set()
                    _fedk = _win_merge(_fedk, _sxf, _SX_WIN, sess_id)
                    _fedp = _win_merge(_fedp, _sxf, _SX_WIN, sess_id)
                if CEFLOOR and _MR_MUST.get(mm):
                    # LIVE, and provably so: `_fedk` is produced by `_mr_merge` -> `_mr_cap` ->
                    # `_mr_gov`, none of which has seen `_MR_MUST`. The cap trims window edges by
                    # distance from a seed and will cut a floor unit whenever the arm is over the
                    # ceiling. If this fires, the configuration is over budget and the answer is a
                    # smaller H_MERGE - not a weaker assert.
                    _lost = set(_MR_MUST[mm]) - set(_fedk)
                    assert not _lost, ("CE floor dropped after merge/cap", mm, len(_lost))
                if DRY:
                    # DRY only: the fed sets themselves, so the operator can be compared to the
                    # cross-encoder's own top-k offline. Never runs in a paid leg.
                    _FEDIDS[mm] = (sorted(_fedk), sorted(_fedp))
                # THE site that runs. Guard-mapping every ctxs[...] assignment to its enclosing
                # ifs showed the other two sit inside `mm == "flrm2_4_3_3"` and `mm == "flrm_4_3_3"`
                # and can never fire for flr_4_3_3, which is why two patches changed nothing.
                if not (_STRUCTURAL and _srender(mm, _fedk, _fedp, p_, _ucap)):
                    ctxs[mm + "|prune_annot"] = _render_units(
                        _fedk, rtext, _cur, ce_rank, sess_id, r_ents, hubs, RGROUP_MAX or U)
                    ctxs[mm + "|composite"] = _render_units(
                        _fedp, rtext, _cur, ce_rank, sess_id, r_ents, hubs, RGROUP_MAX or U)
                    if DRY:
                        # DRY only: what the reader actually receives, so the artifact can be read
                        # rather than inferred from counts.
                        _CTXHEAD[mm] = ctxs[mm + "|prune_annot"][:1500]
                    if NOREFILL:
                        ctxs[mm + "|prune_norefill"] = "\n".join(
                            (("[CURRENT] " + rtext[i]) if i in _cur else rtext[i])
                            for i in sorted(_fednr))
                        ctxs[mm + "|slotdiv"] = "\n".join(rtext[i] for i in sorted(_fedsd))
                        assert set(_fednr) <= set(_fedk), \
                            ("no-refill is not a subset of prune", mm, len(_fednr), len(_fedk))
                        _NR_N[mm] = [len(_fedk), len(_fednr), len(_fedsd)]
                        _pof2 = lambda L: [(sess_id[i] if sess_id[i] is not None else
                                            ("#u%d" % i)) for i in L]
                        _UNITS[mm] = {"prune": sorted(_fedk),
                                      "norefill": sorted(_fednr),
                                      "slotdiv": sorted(_fedsd),
                                      "prune_par": _pof2(sorted(_fedk)),
                                      "slotdiv_par": _pof2(sorted(_fedsd))}

            # NULLCTX: blank every fed context, after all assembly, before the reader. Retrieval
            # has already run and its telemetry stays in the record; only what the reader SEES
            # changes. This sits at the same single point DATEANN uses, so it covers every arm by
            # construction rather than by enumerating them.
            if NULLCTX:
                for _ck in list(ctxs):
                    ctxs[_ck] = ""
            # DATEANN: one application point for every context-assembly path above (generic,
            # flrm, flrm2 and the flrK_tok alias), gated on the dataset and on the env flag.
            # Retrieval already happened; only the reader's string is touched.
            _DA_N[0] = 0
            if DATEANN and DSNAME == "longmemeval":
                _qd = _qdate(q)
                if _qd is not None:
                    for _ck in list(ctxs):
                        ctxs[_ck] = _dateann(ctxs[_ck], _qd)
            _g = gold_of[qi] if qi < len(gold_of) else set()
            rec = {"row": ROW, "qi": qi, "qt": qt, "gold": str(gold), "n_rounds": N,
                   "n_reach": len(gph), "n_reach_sess": len(gph_s), "n_facts": len(facts),
                   "n_pairs": len(pairs), "rho": len(_rp) / max(len(router_window), 1),
                   "n_sup": len(_rp), "wsz": len(router_window),
                   "sup_ka": dict(_SR_SUP), "refill": dict(_NR_N),
                   "units_by_arm": dict(_UNITS),
                   "rho_all": len(pairs) / max(len(sel), 1), "router_view": pick,
                   "profile": (PROFILE_INFO.get("profile") or None),
                   "n_gold": len(_g),
                   # the FULL gold set, so delivered-gold is an intersection rather than
                   # a proxy read off some other ranking
                   "gold_ids": sorted(int(x) for x in _g)}
            if DATEANN:
                rec["dateann"] = _DA_N[0]
            import hashlib as _hl
            # recall@10 per arm, free: the evidence rounds are known from metadata's has_answer flags
            if _g:
                for _cn2, _cl2 in (("d", list(ce_rank)[:50]), ("s", list(sparse)[:50]),
                                   ("g", list(grank_s)[:50])):
                    rec["gpos_" + _cn2] = [k + 1 for k, i in enumerate(_cl2) if i in _g]
                    rec["gids_" + _cn2] = [i for i in _cl2 if i in _g]
                rec["r100d"] = len(_g & set(ce_rank[:100])) / len(_g)
                rec["r100s"] = len(_g & set(list(sparse)[:100])) / len(_g)
                rec["rg"] = len(_g & set(gph)) / len(_g)
                rec["rgs"] = len(_g & set(gph_s)) / len(_g)
                rec["runion"] = len(_g & (set(ce_rank[:100]) | set(list(sparse)[:100])
                                          | set(gph) | set(gph_s))) / len(_g)
                _a3 = set()
                for _pi in pool[METHODS[0]]:
                    for _pe in (r_ents[_pi] - hubs):
                        _a3.update(ef.get(_pe, ()))
                rec["a3reach"] = len(_g & _a3) / len(_g)
                rec["n_a3"] = len(_a3)
                for mm in METHODS:
                    rec["recall|" + mm] = len(_g & set(pool[mm])) / len(_g)
            # the budget receipt - retrieve_num is honoured exactly and the token count is recorded
            # against MemoryAgentBench's own ceiling rather than assumed to be under it
            # Charged PER ARM. Taking the max over the arm list made the ceiling check itself
            # depend on how many arms were being scored: a twenty-arm run could abort on an arm that a
            # one-arm run never builds. Each arm is a candidate deployment and is held to the ceiling
            # on its own account.
            _tka = {a: ntok(v) for a, v in ctxs.items()}
            _tk = max(_tka.values())
            rec["tok_max"] = _tk
            if HBM:
                rec["hbm_b"] = HBM
                rec["hbm_feed"] = HBM_FEED
                rec["hbm_nblk"] = len(_HBM_MEM)
                rec["hbm_npure"] = _HBM_NPURE
                rec["hbm_nfed"] = {a: len(v.split("\n")) for a, v in ctxs.items()}
                rec["hbm_tok"] = _tka
            rec["n_units"] = max(len(pool[mm]) for mm in METHODS)
            if DRY:
                # The channel heads, so a swapped unit can be attributed to the channel that caused
                # it. ce30 is what the U=30 arm is made of; ce100 is the whole dense candidate set.
                rec["ce30"] = [int(i) for i in list(ce_rank)[:30]]
                rec["ce100"] = [int(i) for i in list(ce_rank)[:100]]
                rec["sparse30"] = [int(i) for i in list(sparse)[:30]]
                rec["grank30"] = [int(i) for i in list(grank_s)[:30]]
                rec["par"] = [str(p) for p in sess_id]
                for _mm, (_fk, _fp) in _FEDIDS.items():
                    rec["fedk|" + _mm] = [int(i) for i in _fk]
                    rec["fedp|" + _mm] = [int(i) for i in _fp]
                    rec["poolids|" + _mm] = [int(i) for i in pool[_mm]]
            if MR_ALPHA:
                # W1 (did it fire) and W2 (was it sized) in the record itself, so a null can be
                # told apart from a no-op without re-running anything.
                rec["mr_alpha"] = MR_ALPHA
                rec["mr_docs_rule"] = MRDOCS
                rec["mr_pool"] = MRPOOL
                rec["mr_alpha_dsg"] = [ALPHA_D, ALPHA_S, ALPHA_G]
                # THE NO-OP GATE. Units the selection or the material took from OUTSIDE the
                # channel pool. On the old code this is non-zero (measured 9-11 per question
                # from CE ranks 16-63); under MRPOOL it must be 0. If it reads 0 on BOTH, the
                # patch changed nothing and no paid leg should be run on it.
                rec["mr_poolk"] = MRPOOLK
                rec["mr_deep"] = {m: _MR_DEEP.get(m, {}) for m in METHODS}
                rec["mr_wfill"] = MRWFILL
                rec["mr_ceflr"] = CEFLOOR
                rec["mr_nocap"] = MRNOCAP
                rec["hdetect"] = dict(HDET_INFO)
                rec["mr_nmust"] = {m: len(_MR_MUST.get(m, ())) for m in METHODS}
                rec["mr_headlen"] = {m: [len(p) for p in _MR_POOL[m]]
                                     for m in METHODS if m in _MR_POOL}
                # mr_ndoc_fed is computed on the COMPOSITE view while the score is the prune
                # view, and it conflates "claimed fewer" with "fed fewer". Record CLAIMED.
                rec["mr_ndoc_claimed"] = {m: min(10, NPAR) - ((_MR_SPLIT.get(m) or {})
                                          .get("short") or 0) for m in METHODS}
                rec["mr_pool_esc"] = {}
                for _m in METHODS:
                    _pp = _MR_POOL.get(_m)
                    if _pp is None:
                        continue
                    _pall = set(_pp[0]) | set(_pp[1]) | set(_pp[2])
                    rec["mr_pool_esc"][_m] = len(set(_MR_SEED.get(_m, ())) - _pall)
                rec["mr_cap"] = MRCAP
                rec["mr_cap_trimmed"] = {m: _MR_CUT.get(m, 0) for m in METHODS}
                # how many times the budget BOUND, which `cut` can no longer distinguish from
                # "never bound" once the protocol base is reserved off the top
                rec["mr_cap_bound"] = {m: _MR_BOUND.get(m, 0) for m in METHODS}
                rec["mr_sxmerge"] = int(bool(SXMERGE and SPARSEX))
                rec["mr_nsface"] = {m: len(_MR_SFACE.get(m) or ()) for m in METHODS}
                rec["mr_split"] = {m: _MR_SPLIT.get(m) for m in METHODS}
                rec["mr_ndoc_fed"] = {m: (_MR_FED.get(m) or (0, 0))[1] for m in METHODS}
                rec["mr_gov_dropped"] = {m: (_MR_FED.get(m) or (0, 0))[0]
                                            - (_MR_FED.get(m) or (0, 0))[1] for m in METHODS}
                rec["mr_hmerge"] = H_MERGE
                rec["mr_hscan"] = SPARSEX
                rec["mr_nseed"] = {m: len(_MR_SEED.get(m, ())) for m in METHODS}
                rec["mr_ndoc"] = {m: len(_MR_DOCS.get(m, ())) for m in METHODS}
                rec["mr_nfed"] = {a: len(v.split("\n")) for a, v in ctxs.items()}
            # Every arm's delivered token count, always recorded. flr+ is matched on MERGED
            # UNITS and not on tokens, so its number is meaningless without this column, and the
            # only way to guarantee that is to write it next to the score for every arm.
            rec["tok_by_arm"] = {_a: int(_t) for _a, _t in _tka.items()}
            rec["row_cap"] = int(ROW_CAP)
            rec["render_over"] = {_a: int(max(0, _t - ROW_CAP)) for _a, _t in _tka.items() if _t > ROW_CAP}
            for _a, _t in sorted(_tka.items()):
                # flr+ exceeds the ceiling BY CONSTRUCTION: it keeps all ten seeds and delivers each
                # as a span. The ceiling is what keeps the other arms comparable, so it is exempted
                # here and nowhere else, and the exemption is visible in the record.
                # The protocol bound is TEN UNITS at the cell chunk size. On a 4096-chunk cell -
                # DetectiveQA, the ICL sets - ten units IS 40,960 exactly, so the rendered string
                # trips this guard on its separators alone: DetectiveQA q1 fed 40,980, over by 20
                # tokens of framing, and the cell died at one record. The guard exists to catch an
                # arm smuggling extra CONTENT, and 64 tokens is a sixty-fourth of one unit, so a
                # render allowance that small cannot hide content. Any overshoot is recorded either
                # way, so the allowance is visible in every record rather than argued for here.
                # NO TOKEN CEILING. MemoryAgentBench does not have one. Its RAG agent joins the
                # top_k retrieved passages and hands them to the reader unmodified -
                # embedding_retriever.py:226-231 - and its max_tokens is generation_max_length, the
                # ANSWER length, not an input budget. OFFICIAL_CAP = 10 x 4096 was ours, and calling
                # it "the benchmark's own ceiling" in this file was wrong. It never bound any arm
                # (every arm delivers exactly ten units, measured at 4,978-5,036 tokens on the 512
                # cells and 274 on the profiled ones) and the one thing it did was refuse a
                # protocol-conformant ten-unit delivery on in-context-learning, where the chunker
                # splits on example boundaries and ten units weigh 56,911 tokens. The protocol bound
                # is TEN UNITS; that is asserted where the units are built. Tokens are recorded on
                # every record so any arm's cost is visible, and TOKCAP can restore a hard bound for
                # a deliberate budget experiment.
                _tc = int(os.environ.get("TOKCAP", "0") or 0)
                if _tc and _t > _tc and not _a.startswith("flrp"):
                    raise RuntimeError("row %d q%d arm %s fed %d tokens, over TOKCAP %d"
                                       % (ROW, qi, _a, _t, _tc))

            # If the structural path was selected but no structural render ran, a branch was
            # missed - which is exactly what happened for 30 paid questions. Fail on the
            # first question, not after the money is spent.
            assert (not _STRUCTURAL) or _SR_HITS[0] > 0, \
                ("STRUCTURAL selected but no structural render executed", ROW, qi)
            for _a, _c in ctxs.items():
                rec["ctxhash|" + _a] = _hl.sha1(_c.encode()).hexdigest()[:12]
                if DUMP_CTX:
                    rec["ctx|" + _a] = _c

            def ask(a):
                # OFFICIAL_PROMPT uses MemoryAgentBench's own rag_agent query template for the
                # dataset. It is the default for every dataset except longmemeval, where the numbers
                # already measured used the harness's own wording and switching silently would make
                # them incomparable; set OFFICIAL_PROMPT=1 to move that one too.
                _p2 = a.endswith("#p2")
                _ak = a[:-3] if _p2 else a
                if OFFICIAL_PROMPT or DSNAME != "longmemeval":
                    prompt = (ctxs[_ak] + "\n\n"
                              + ((RAG_QUERY2 or _rag_query("ruler_qa")) if _p2 else _rag_query(DSNAME)).replace("{question}", q)
                              + _NOTHINK)
                else:
                    prompt = ("You are answering from a user's long-term chat memory.\nRetrieved memory:\n"
                              + ctxs[_ak] + "\n\nQuestion: " + q
                              + "\nAnswer with the shortest exact answer:" + _NOTHINK)
                r_ = g.openai(prompt, model=MODEL, max_tokens=MAXOUT, temperature=TEMP,
                              system=SYSTEM_MESSAGE, seed=SEED, note="lmeo|%s|r%dq%d" % (a, ROW, qi))
                return a, _THINK.sub("", r_ or "").strip()
            if ROUTER_ONLY:
                pf.write(json.dumps(rec) + "\n"); nq += 1; bytype[qt]["n"] += 1
                continue
            if DRY:
                for mm in METHODS:
                    rec["pool|" + mm] = list(pool[mm])
                    rec["nsess|" + mm] = len({sess_id[i] for i in pool[mm]})
                rec["gold_ids"] = sorted(_g)
                if _RG:
                    rec["rgroup"] = dict(_RG)
                    if DRY and _CTXHEAD:
                        rec["ctxhead"] = dict(_CTXHEAD)
                pf.write(json.dumps(rec) + "\n"); nq += 1; bytype[qt]["n"] += 1
                if nq % 10 == 0:
                    _ce_save()
                if nq % 60 == 0:
                    print("   dry %3d  ce hit %d miss %d"
                          % (nq, _CE_HIT[0], _CE_MISS[0]), flush=True)
                continue
            _asklist = list(ctxs) + ([_an + "#p2" for _an in ctxs] if P2 else [])
            # One wave, not two. A five-arm cell asks ten contexts per question and a pool of six ran
            # them as 6+4, so every question paid two round trips to the reader when one would do.
            # ThreadPoolExecutor.map returns in input order whatever the worker count, each ask() is
            # independent, and the results are assigned only after the map completes, so this is a
            # wall-clock change and not a semantic one. Capped at ten so a three-arm cell (six asks)
            # is untouched and no cell bursts harder than a five-arm one already would.
            with ThreadPoolExecutor(max_workers=min(len(_asklist), 12)) as ex_:
                for a, pred in ex_.map(ask, _asklist):
                    s_ = subem(pred, gold)
                    rec[a] = s_; rec[a + "_pred"] = pred
                    hits[a] += s_; bytype[qt][a] += s_
            for mm in METHODS:
                rec["router|" + mm] = rec["%s|%s" % (mm, pick)]  # raw-key-ok: rec[arm|view] here is the subem this harness just computed, so copying it to router|arm IS the router score at generation time; cellscore re-maps the judged and benchmark cells downstream
                hits["router|" + mm] += rec["router|" + mm]
                bytype[qt]["router|" + mm] += rec["router|" + mm]
            bytype[qt]["n"] += 1
            nq += 1
            pf.write(json.dumps(rec) + "\n")
            if nq % 10 == 0:
                _ce_save()
            if nq % 20 == 0:
                print("   %3d  $%.4f" % (nq, g.spent), flush=True)
        pf.flush()
        _ce_save()
        del kv
        (torch.mps.empty_cache() if torch.backends.mps.is_available() else (torch.cuda.empty_cache() if torch.cuda.is_available() else None))

    pf.close()
    if DRY:
        print("\n=== DRY: retrieval only, n=%d questions, $0.0000 ===" % nq)
        print("    XC_HIT cache-hits=%d api-calls=%d" % (XC_HIT[0], XC_HIT[1]), flush=True)
        print("LME_OFFICIAL_DRY_DONE", flush=True)
        return
    print("\n=== LongMemEval, MemoryAgentBench Accurate_Retrieval split, SubEM, n=%d, $%.4f ==="
          % (nq, g.spent))
    print("   %-18s %-11s %-11s %-11s %-11s %s"
          % ("method", "prune_annot", "composite", "no_refill", "slotdiv", "router"))
    for mm in METHODS:
        print("   %-18s %-11.3f %-11.3f %-11s %-11s %.3f"
              % (mm, hits[mm + "|prune_annot"] / nq, hits[mm + "|composite"] / nq,
                 ("%.3f" % (hits[mm + "|prune_norefill"] / nq)) if NOREFILL else "-",
                 ("%.3f" % (hits[mm + "|slotdiv"] / nq)) if NOREFILL else "-",
                 hits["router|" + mm] / nq))
    if NOREFILL:
        print("\n   T8, a 2x2         as ranked        ucap distinct slots")
        print("      keep retrieved   composite        slotdiv")
        print("      substitute       no_refill        prune_annot")
        print("   composite -> no_refill is SUBSTITUTION; composite -> slotdiv is the BUDGET")
        print("   channel alone, and slotdiv reads no ledger, no extractor and no profile.")
    print("\n   by question type (router column)")
    for t in sorted(bytype):
        c = bytype[t]
        print("   %-26s n=%-4d %s" % (t, c["n"],
              "  ".join("%s %.3f" % (mm, c["router|" + mm] / c["n"]) for mm in METHODS[:4])))
    print("   CAR on knowledge-update (n=45): deterministic 0.578, its own baseline 0.644")
    print("LME_OFFICIAL_DONE", flush=True)


if __name__ == "__main__":
    main()
