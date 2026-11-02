"""The build path of lme_pf29.py, cut for timing (plus two helpers it imports). Do not edit.
Every block between '# >>> COPY' and '# <<< COPY' is the named lines of the harness verbatim (minus the stated indent),
which verify_copies.py checks; the rest is glue and says what it stands for. No torch, no network."""
import ast, collections, math, os, re, string, sys, types   # glue: the imports of lme_pf29.py (subset used below)
import numpy as np                                          # glue: as lme_pf29.py
import tiktoken                                             # glue: as lme_pf29.py
from rank_bm25 import BM25Okapi                             # glue: as lme_pf29.py
# >>> COPY lme_pf29.py:35-35 dedent=0
BM25LC = os.environ.get("BM25LC", "") not in ("", "0")
# <<< COPY
# >>> COPY lme_pf29.py:162-177 dedent=0
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
# <<< COPY
# >>> COPY lme_pf29.py:259-259 dedent=0
ENC = tiktoken.encoding_for_model("gpt-4o-mini")
# <<< COPY
# >>> COPY lme_pf29.py:832-832 dedent=0
CHUNK = int(os.environ.get("CHUNK", "4096"))
# <<< COPY
# >>> COPY lme_pf29.py:836-836 dedent=0
MSCALE = int(os.environ.get("MSCALE", "0") or 0)
# <<< COPY
# >>> COPY lme_pf29.py:849-850 dedent=0
HYBRID = int(os.environ.get("HYBRID", "0") or 0)
AUTOSTRUCT = int(os.environ.get("AUTOSTRUCT", "0") or 0)
# <<< COPY
# >>> COPY lme_pf29.py:853-854 dedent=0
FORCE_ORDER = int(os.environ.get("FORCE_ORDER", "0") or 0)
assert not MSCALE or MSCALE > CHUNK, ("MSCALE must be COARSER than CHUNK", MSCALE, CHUNK)
# <<< COPY
# >>> COPY lme_pf29.py:858-859 dedent=0
ATOM = os.environ.get("ATOM", "sentence")
FIT_Q = float(os.environ.get("FIT_Q", "0.98"))
# <<< COPY
# >>> COPY lme_pf29.py:864-864 dedent=0
ATOM1 = int(os.environ.get("ATOM1", "0") or 0)
# <<< COPY
# >>> COPY lme_pf29.py:869-869 dedent=0
assert ATOM in ("sentence", "blank", "newline", "auto"), ("bad ATOM", ATOM)
# <<< COPY
# >>> COPY lme_pf29.py:1055-1056 dedent=0
def slotkey(a):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(a).lower()).split())
# <<< COPY
# >>> COPY lme_pf29.py:1059-1101 dedent=0
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
# <<< COPY
# >>> COPY lme_pf29.py:1161-1162 dedent=0
HDETECT = int(os.environ.get("HDETECT", "0") or 0)
HDET_INFO = {}
# <<< COPY
# >>> COPY lme_pf29.py:1224-1239 dedent=0
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
# <<< COPY
# >>> COPY lme_pf29.py:1255-1292 dedent=0
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
# <<< COPY
# >>> COPY lme_pf29.py:1294-1305 dedent=0
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
# <<< COPY
# >>> COPY lme_pf29.py:1979-1979 dedent=0
PROFILE = int(os.environ.get("PROFILE", "0") or 0)
# <<< COPY
# >>> COPY lme_pf29.py:1983-1983 dedent=0
LPROFILE = int(os.environ.get("LPROFILE", "0") or 0)
# <<< COPY
PROFILES = {}   # glue: PROFILE=0 in every shipped leg, so it is never read
# >>> COPY real_e1_conflict_killer.py:26-27 dedent=0
_CAP = re.compile(r"\b([A-Z][\w.'-]+(?:\s+(?:of|the|and|de|van|von|di)?\s*[A-Z][\w.'-]+)*)\b")
def entities(text): return {m.strip() for m in _CAP.findall(text or "") if len(m.strip()) > 2}
# <<< COPY
# >>> COPY real_e1_conflict_killer.py:29-54 dedent=0
def build_entity_index(fact_ents, n_facts, hub_frac=0.05):
    """Env HUB selects the hub rule: rel (ships) | abs | off.

    The relative rule is wrong in both directions, measured: at N=641 its cut of 32 postings removes
    load-bearing conversational entities and caps reach coverage at 0.513 where an absolute cut gives
    0.744; at N=18332 its cut of 916 postings removes 0.0% of the vocabulary - no proper noun appears
    in 917 fact sentences - so the device is silently off in the largest cells. It stays the default
    ONLY until the C3 measurement decides; the expansion-degree cap subsumes its intent (a hub is by
    definition low-IDF, so a top-k-by-IDF expansion never selects one) with strictly better failure
    semantics: this filter deletes an entity for every query at index time, the cap merely declines to
    expand through it while direct matches survive.
    """
    import os
    ent_facts = {}
    for i, es in enumerate(fact_ents):
        for e in es: ent_facts.setdefault(e, set()).add(i)
    mode = os.environ.get("HUB", "rel")
    if mode == "off":
        hubs = set()
    elif mode == "abs":
        cut = int(os.environ.get("HUB_ABS", "200"))
        hubs = {e for e, fs in ent_facts.items() if len(fs) > cut}
    else:
        hub_cut = max(3, int(hub_frac * n_facts))
        hubs = {e for e, fs in ent_facts.items() if len(fs) > hub_cut}
    return {e: fs for e, fs in ent_facts.items() if e not in hubs}, hubs
# <<< COPY
# >>> COPY core2.py:429-438 dedent=0
def newest_per_parent(pool, corpus, key_of=None):
    """Index of the highest-order unit per resolution key. The key is the SLOT, which is not necessarily
    the retrieval parent - changing what we retrieve by must not silently change what we resolve by."""
    kf = key_of or (lambda i: corpus.parent_of[i])
    newest = {}
    for i in pool:
        k = kf(i)
        if k is not None and (k not in newest or corpus.order_of[i] > corpus.order_of[newest[k]]):
            newest[k] = i
    return newest
# <<< COPY
MC = types.SimpleNamespace(newest_per_parent=newest_per_parent)   # glue: `import core2 as MC` in lme_pf29.py, the one function used below


def units_row(df, ROW):
    """load_row (lme_pf29.py:2015-2267) minus evaluation-only lines (questions, answers, gold labels, question
    types) and minus the branches no shipped leg reaches (LPROFILE, PROFILE, HYBRID are 0 in every leg).
    Returns (rtext, rkey, parent, order, dataset): rtext feeds BM25 and the reader, rkey is what is embedded."""
# >>> COPY lme_pf29.py:2026-2027 dedent=0
    m = df["metadata"].iloc[ROW]
    ds = norm_dataset(m.get("source"))
# <<< COPY
# >>> COPY lme_pf29.py:2030-2032 dedent=0
    if ds == "longmemeval":
        sess = sessions_from_context(str(df["context"].iloc[ROW]))
        rtext, rkey, sess_id, rdate, by_user, by_asst = rounds_of(sess)
# <<< COPY
        return rtext, rkey, sess_id, rdate, ds   # glue: the rest of this branch is gold rounds and question types, evaluation only
# >>> COPY lme_pf29.py:2072-2111 dedent=0
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
# <<< COPY
        # glue: `import ledger_profile as _LP` omitted; _LP is read only when AS_SLOT < 3, every FC leg runs AS_SLOT=5
# >>> COPY lme_pf29.py:2113-2163 dedent=0
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
# <<< COPY
        return ch, ch, par, _ord, ds   # glue: the rest of this branch is gold_of, evaluation only
    if HYBRID:   # glue: HYBRID=0 in every shipped leg
        raise NotImplementedError("HYBRID=0 in every shipped leg")
# >>> COPY lme_pf29.py:2198-2262 dedent=0
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
# <<< COPY
    return ch, ch, par, list(range(len(ch))), ds   # glue: the rest of load_row is gold_of, evaluation only


def embed_row(emb, rkey):
    """The one dense pass per row (lme_pf29.py:2411-2412)."""
# >>> COPY lme_pf29.py:2411-2412 dedent=4
    kv = np.asarray(emb.encode(rkey, batch_size=32, normalize_embeddings=True,
                               show_progress_bar=False), dtype=np.float32)
# <<< COPY
    return kv


def bm25_row(rtext):
    """The unit-level BM25 index per row (lme_pf29.py:2508; HBM, SPARSEX and SPH are 0 in every shipped leg)."""
# >>> COPY lme_pf29.py:2508-2508 dedent=4
    bm = BM25Okapi([_btok(t) for t in rtext])
# <<< COPY
    return bm


def graph_row(rtext, rkey, sess_id, rdate, DSNAME):
    """The per-row entity index of the graph channel and the slot maps (lme_pf29.py:2419-2481), reported apart."""
    N = len(rtext)   # glue: `N = len(rtext)` at the top of the row's build in lme_pf29.py
# >>> COPY lme_pf29.py:2419-2419 dedent=4
    _ORD = rdate if ((PROFILE and DSNAME in PROFILES) or LPROFILE or AUTOSTRUCT) else list(range(N))
# <<< COPY
# >>> COPY lme_pf29.py:2437-2437 dedent=4
    _HAS_ORDER = bool((PROFILE and DSNAME in PROFILES) or LPROFILE or AUTOSTRUCT or HYBRID or FORCE_ORDER)
# <<< COPY
# >>> COPY lme_pf29.py:2440-2441 dedent=4
    _ORD_OK = _HAS_ORDER and len(set(_ORD)) >= 0.99 * N
    _STRUCTURAL = _ORD_OK and (len(set(sess_id)) < N)
# <<< COPY
    corpus = types.SimpleNamespace(parent_of=list(sess_id), order_of=list(_ORD))   # glue: the two MC.Corpus fields newest_per_parent reads (core2.py:47-48)
# >>> COPY lme_pf29.py:2448-2481 dedent=4
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
# <<< COPY
    return dict(entities=len(ef), hubs=len(hubs), slots=len(newest_of_slot), structural=bool(_STRUCTURAL),
                slot_edges=sum(len(v) for v in _ADJ_SLOT.values()))
