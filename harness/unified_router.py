#!/usr/bin/env python3
"""
UNIFIED CURRENCY ROUTER — one implementation shared by every benchmark harness
(MQuAKE-CF, FactConsolidation, LongMemEval). Importing the same module in all harnesses is what
lets the paper claim a single, dataset-independent router.

DECISION (1 cheap LLM call on the QUERY ONLY + three $0 store-side gates):

    intent = LLM(q) in {current, history, aggregate}      # the only model cost
    if intent != "current":            -> KEEP_ALL        # history needs old values; aggregate needs all evidence
    if max_slot_conflict(pool) < 2:    -> KEEP_ALL        # nothing to supersede => byte-identical to baseline
    if not prune_safe(...):            -> KEEP_ALL        # suppression would destroy evidence
    else:                              -> PRUNE           # drop superseded values, keep the current one

WHY EACH GATE (each was forced by a measured failure):
  G0 intent      : LongMemEval multi-session collapsed (prune .400 vs dense .800) because aggregation
                   questions need every occasion, not the latest value.
  G1 conflict    : if no slot holds >=2 distinct values, pruning is a no-op; returning KEEP_ALL makes the
                   router provably no-harm on non-currency workloads (FC/MQuAKE non-conflict slots,
                   HotpotQA-style tasks).
  G2 topic       : two values only conflict if their evidence is about the same thing
                   (cos(source_i, source_winner) >= topic_tau); free-text slot keys collide across
                   unrelated sessions at LongMemEval-S scale (~540 chunks) and caused false suppression.
  G3 recency     : only suppress when the surviving value is strictly newer.
  G4 evidence    : never suppress the single most query-relevant item (LongMemEval L_prune showed naive
                   slot-collapse evicts the gold 42-58% of the time when slots come from free-text).

The router NEVER sees gold, answer ids, question_type, or any oracle metadata: only the query string and
the retrieved pool.
"""
from __future__ import annotations
import os as _os

PROMPT = (
    "Classify what this long-term-memory question needs:\n"
    "  current   - the value that holds NOW for one attribute of the user. The user may have stated it\n"
    "              several times and updated it; the answer is the most recent statement. A COUNT or a\n"
    "              NUMBER that the user updated over time is STILL 'current' (e.g. 'how many X have I\n"
    "              tried', 'what is my best time') because one latest statement answers it.\n"
    "  history   - an EARLIER/ORIGINAL value that was later replaced ('what was it BEFORE', 'the FIRST X').\n"
    "  aggregate - the answer only exists by COMBINING several separate occasions that were never stated\n"
    "              together (e.g. 'list everything I did across these months', 'total across all trips').\n"
    "Mere past tense ('what was my X') usually still wants the LATEST value -> current.\n"
    "IMPORTANT TIE-BREAK: if the question could plausibly need BOTH the latest value AND evidence from\n"
    "several separate occasions (e.g. 'how many projects have I led OR am currently leading'), answer\n"
    "aggregate. Only answer current when ONE latest statement fully answers the question.\n"
    "Reply with exactly one word: current, history, or aggregate.\nQuestion: %s\nAnswer:"
)

KEEP_ALL, PRUNE = "keep_all", "prune"


STATS = {"calls": 0, "empty": 0, "unparsed": 0}


PROMPT_MULTI = (
    "Which of these readings apply to this long-term-memory question? Answer with ALL that apply,\n"
    "comma-separated, using only these words: current, history, aggregate.\n"
    "  current   - it asks for the value that holds NOW for one attribute. The user may have updated\n"
    "              that attribute over time; the answer is the most recent statement.\n"
    "  history   - it asks for an EARLIER/ORIGINAL value that was later replaced ('what was it BEFORE',\n"
    "              'the FIRST X'), or for how a value changed.\n"
    "  aggregate - the answer only exists by COMBINING several separate occasions that were never\n"
    "              stated together in one place.\n"
    "Answer with more than one word if more than one reading applies.\n"
    "Question: %s\nAnswer:")

_EXACT = frozenset(("current", "history", "aggregate"))
_LABELS = (("aggregate", ("aggreg", "combin", "total across", "list everything")),
           ("history", ("hist", "past", "orig", "earlier", "previous", "before")),
           ("current", ("current", "latest", "now")))


PROMPT_VALIDATED = ("Classify this question intent as ONE word - current (present/latest value), "
                    "list (all values or full history), or previous (a specific past value). "
                    "Question: %s Intent:")


def classify_intent_validated(q, gen):
    """
    The intent function of the VALIDATED router, reproduced verbatim: one word from
    {current, list, previous}, first match in the order previous > list > current, and - importantly -
    a failed or unparseable reply FALLS BACK TO `current`, i.e. it fails toward pruning. That default is
    part of why the policy works on stores where supersession is the norm; abstaining to keep-all instead
    is what made the gated module trail it.
    """
    STATS["calls"] += 1
    r = (gen(PROMPT_VALIDATED % q, 8) or "").strip().lower()
    if not r or r.startswith("__err__"):
        STATS["empty"] += 1
    for k in ("previous", "list", "current"):
        if k in r:
            return k
    return "current"


import json as _json
import atexit as _atexit
_ICACHE_PATH = _os_early = None


def _icache_load():
    import os as _o
    p = _o.environ.get("UR_INTENT_CACHE")
    if not p:
        return None, None
    d = {}
    if _o.path.exists(p):
        try:
            d = _json.load(open(p))
        except Exception:
            d = {}
    return p, d


_ICACHE_PATH, _ICACHE = _icache_load()


def _icache_put(q, labels):
    if _ICACHE is not None:
        _ICACHE[q] = sorted(labels)


def _icache_save():
    if _ICACHE_PATH and _ICACHE is not None:
        try:
            _json.dump(_ICACHE, open(_ICACHE_PATH, "w"))
        except Exception:
            pass


_atexit.register(_icache_save)


_NO_LLM = _os.environ.get("UR_NO_LLM", "") == "1"


def classify_intents(q, gen, strict=True):
    """
    MULTI-LABEL intent. Returns a frozenset of {current, history, aggregate}, or an EMPTY set when the
    backend failed or the reply was unparseable (V0: never a silent default).

    The model is not asked to disambiguate - only to report. route() then applies the most restrictive
    permission across the reported labels, so an ambiguous question always suppresses LESS, never more.
    """
    if _NO_LLM:
        # The LLM-free variant does not classify intent at all: with HISTVETO/AGGSTRICT disabled no gate
        # reads the labels, so asking would spend a call on an answer nothing consumes. A neutral
        # non-empty set is returned so V0 - which refuses to prune on a FAILED classification - does not
        # confuse "we chose not to ask" with "the classifier broke".
        return frozenset(["current"])
    if _ICACHE is not None and q in _ICACHE:
        return frozenset(_ICACHE[q])
    STATS["calls"] += 1
    raw = gen(PROMPT_MULTI % q, 16)
    r = (raw or "").strip().lower()
    if not r or r.startswith("__err__"):
        STATS["empty"] += 1
        _icache_put(q, frozenset())
        return frozenset()
    # 1) EXACT TOKENS: only the literal label words count, so a negated or explanatory mention
    #    ('current, not a previous value') cannot smuggle in a label that vetoes every suppression.
    toks = set(_re.findall(r"[a-z]+", r))
    out = toks & {"current", "history", "aggregate"}
    if not out:
        # 2) fallback for paraphrases ('the latest value', 'combine several occasions')
        out = {name for name, keys in _LABELS if any(k in r for k in keys)}
    if not out:
        STATS["unparsed"] += 1
        return frozenset()
    _icache_put(q, frozenset(out))
    return frozenset(out)


def classify_intent(q, gen, strict=True):
    """
    gen(prompt, max_tokens) -> str, supplied by the harness (same backbone as its reader).

    INSTRUMENTATION INTEGRITY (V0): a dead backend previously returned "" for every call and the old
    code silently defaulted to "current", producing a meaningless 8/8-current measurement. Now an empty
    or unparseable response is recorded and (strict=True) surfaced as "unknown" instead of a silent
    default, so a harness can abort / mark the run invalid. Check STATS["empty"]/STATS["calls"].
    """
    STATS["calls"] += 1
    raw = gen(PROMPT % q, 12)
    r = (raw or "").strip().lower()
    if not r:
        STATS["empty"] += 1
        return "unknown" if strict else "current"
    if any(w in r for w in ("aggreg", "combin", "total across", "list everything")):
        return "aggregate"
    if any(w in r for w in ("hist", "past", "orig", "earlier", "previous", "before")):
        return "history"
    if "current" in r or "latest" in r or "now" in r:
        return "current"
    STATS["unparsed"] += 1
    return "unknown" if strict else "current"


def integrity_ok(max_empty_rate=0.01):
    """V0 gate: refuse to trust a run whose router backend was failing."""
    c = max(1, STATS["calls"])
    return (STATS["empty"] + STATS["unparsed"]) / c <= max_empty_rate


def max_slot_conflict(slots, values):
    """G1: largest number of DISTINCT values carried by any single slot in the retrieved pool."""
    per = {}
    for s, v in zip(slots, values):
        if s is None:
            continue
        per.setdefault(s, set()).add(v)
    return max((len(x) for x in per.values()), default=0)


import re as _re

_NUM = _re.compile(r"-?\d[\d,\.:]*")


_TEMPORAL = _re.compile(
    r"\b(when|how long|how many (?:days|weeks|months|years|hours)|before|after|during|since|until|"
    r"earlier|first time|last time|earliest|prior to|ago|"
    r"which (?:month|year|day|week|season)|what (?:month|year|day|date))\b|"
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
    _re.I)


_PAST_CUE = _re.compile(
    # SUPERSESSION markers only. Deliberately EXCLUDES past tense ("what was the total...") and
    # recency windows ("in the past month"), which are evidence of aggregation, not replacement,
    # and which caused 3.3% false vetoes on LongMemEval.
    r"\b(before|previously|prior to|originally|former|formerly|used to|earlier|initially|"
    r"the first|at the time|back then|changed from|switched from|moved from|since then|"
    r"no longer|instead of|replaced|renamed|rather than)\b", _re.I)


def has_past_reference(q):
    """
    G8 - HISTORY-CUE REQUIREMENT ($0, deterministic).

    A `history` label vetoes every suppression, so it is the router's most destructive decision and must
    be evidence-backed. A question that genuinely wants a superseded value says so ("what was it BEFORE",
    "the ORIGINAL X", "who was the ... "). Without such a cue the label is dropped as unsupported - which
    matters because on an edit-style benchmark the classifier reports `history` on chained questions that
    contain no past reference at all.
    """
    return bool(_PAST_CUE.search(q or ""))


_AGG_CUE = _re.compile(
    # counting / enumeration
    r"\b(all|every|each|both|list|listed|how many|how much|total|totals|count|counts|"
    r"sum|average|mean|entire|altogether|combined|overall|throughout|ones|"
    # superlatives: an argmax over per-occasion values needs every value kept
    r"most|least|highest|lowest|largest|smallest|biggest|longest|shortest|best|worst|"
    r"maximum|minimum|top|first|last|earliest|latest|"
    # deltas and comparisons: a difference needs BOTH endpoints, so the older one cannot be pruned
    r"increase|increased|decrease|decreased|change|changed|difference|differ|compare|compared|"
    r"growth|grew|gain|gained|drop|dropped|rise|risen|fell|fallen|trend|"
    r"more than|less than|fewer than|between|since|so far|over time|"
    r"over the past|in the past|during the past|across)\b", _re.I)


def has_aggregate_reference(q):
    """
    G9 - AGGREGATE-CUE REQUIREMENT ($0, deterministic). Symmetric to G8/has_past_reference.

    An `aggregate` label sets strict_sub, which makes G5 demand proof that two values substitute rather
    than coexist, which vetoes the suppression. So the label is nearly as destructive as `history` and
    deserves the same evidence bar: a question that genuinely aggregates says so ("ALL the", "HOW MANY",
    "EVERY", "in TOTAL"). Without a cue the label is dropped as unsupported, which matters because on a
    multi-hop benchmark the classifier reports `aggregate` on compositional questions that aggregate
    nothing - 50% of FactConsolidation, where the true rate is zero.
    """
    return bool(_AGG_CUE.search(q or ""))


def is_temporal_query(q):
    """
    G6 - TEMPORAL-EVIDENCE GUARD ($0, deterministic, no model call).

    A question ABOUT TIME makes the timeline evidence rather than noise: two dated values of the same
    slot are then two EVENTS to be reasoned over, not an old value and its replacement. When this fires
    we disable G5's numeric shortcut, so a numeric/date difference alone can no longer justify
    suppression. This is the cheap discriminator between LongMemEval temporal-reasoning (timeline is the
    answer) and knowledge-update (timeline is stale clutter), which the intent label alone conflates.
    """
    return bool(_TEMPORAL.search(q or ""))


_COLLECTIVE = _re.compile(
    r"\b(all|every|each|list|both|several|various|multiple|any of|kinds? of|types? of|"
    r"projects?|places?|cities|countries|languages|friends|books|movies|shows|hobbies|skills|"
    r"courses|pets|siblings|children|trips|visits|events|meetings|purchases|items|tools|teams|"
    r"members|topics|goals|habits|activities|restaurants|recipes|apps|devices)\b", _re.I)


def slot_is_functional(slot):
    """
    Rule D - CARDINALITY LEXICON ($0, deterministic, query-independent).

    FUNCTIONAL slot  : holds exactly ONE value at a time, so a newer value REPLACES the older one no
                       matter how different the two strings look ("capital: Paris" -> "capital: Lyon",
                       "employer: Acme" -> "employer: Globex"). This is the case the value-string
                       heuristic used to miss, and it is the central case of FactConsolidation/MQuAKE.
    MULTI-VALUED slot: accumulates ("projects led", "cities visited"). Values coexist; suppression
                       destroys evidence.

    Signals: an explicitly collective word anywhere in the key, or a plural head noun. Both are crude;
    both only ever push the decision toward the SAFE side (multi-valued => keep everything).
    """
    t = (slot or "").strip().lower()
    if not t:
        return False
    if _COLLECTIVE.search(t):
        return False
    words = _re.findall(r"[a-z]+", t)
    if not words:
        return False
    h = words[-1]
    if len(h) > 3 and h.endswith("s") and not h.endswith(("ss", "us", "is", "ous")):
        return False                                  # plural head noun => accumulates
    return True


_COLLECTIVE_Q = _re.compile(
    r"\b(all|every|each|both|list|enumerate|name them|several|various|multiple|"
    r"how many (?:different|distinct)|which ones|what were the|throughout|over the years|"
    r"in total|altogether|combined)\b", _re.I)


def is_collective_query(q):
    """
    G7 - COLLECTIVE-QUERY GUARD ($0, deterministic).

    Rule D reads the STORE ("this slot holds one value at a time"); it cannot see that the QUESTION
    asked for the slot's whole history ("summarise EVERY role I have held"). When a quantifier or an
    enumeration verb appears in the query, the functional-slot shortcut is switched off and suppression
    must be justified by the value strings alone. Restricted to genuine quantifiers, so singular
    questions such as "what is my current project?" are untouched.
    """
    return bool(_COLLECTIVE_Q.search(q or ""))


def cooccurring_slots(slots, values, source_of):
    """
    Rule A - CO-OCCURRENCE VETO ($0, deterministic, store-side).

    If the store carries >=2 DISTINCT values of the same slot from the SAME source (same chunk /
    session / document), the user stated them TOGETHER, so they coexist by construction and no
    supersession relation exists between them. Sequential multiplicity - one value per source, ordered
    in time - is what an overwrite actually looks like. This veto overrides every positive signal,
    including Rule D, because it is direct evidence about the slot rather than a lexical guess.
    """
    if source_of is None:
        return set()
    seen, bad = {}, set()
    for i, (sl, v) in enumerate(zip(slots or [], values or [])):
        if sl is None:
            continue
        k = (sl, source_of.get(i))
        if k in seen and seen[k] != v:
            bad.add(sl)
        seen.setdefault(k, v)
    return bad


def substitutive(v_old, v_new, strict=False, temporal=False, functional=False):
    """
    G5 — SUBSTITUTIVE vs ENUMERATIVE conflict, decided from the VALUE STRINGS alone ($0).

    A slot holding two values can mean two different things:
      SUBSTITUTIVE : the newer value REPLACES the older one   ("PB 27:12" -> "PB 25:50";
                     "citizenship Portugal" -> "United Kingdom")   => safe to suppress the old one
      ENUMERATIVE  : both values COEXIST and the answer needs them all
                     ("projects led: Apollo", "projects led: Zephyr")  => suppressing destroys evidence

    Decision rule (in order):
      1. both values are numeric/temporal  -> SUBSTITUTIVE (a measurement was updated)
      2. one value string CONTAINS the other (extension/refinement) -> SUBSTITUTIVE
      3. otherwise: token-overlap. Values that are lexically disjoint proper nouns look like distinct
         coexisting entities -> ENUMERATIVE; values sharing most tokens are a rewrite -> SUBSTITUTIVE.
    Risk is asymmetric (wrong suppression = real harm, missed suppression = forfeited gain), so the
    default on ambiguity is ENUMERATIVE (i.e. do NOT suppress).
    """
    a, b = (v_old or "").strip().lower(), (v_new or "").strip().lower()
    if not a or not b or a == b:
        return True                                   # identical/empty: suppression is a no-op
    na, nb = _NUM.findall(a), _NUM.findall(b)
    if na and nb and not temporal:
        return True                                   # 1. numeric/temporal update
        # G6: under a TEMPORAL question each dated/measured value is an EVENT, so this shortcut
        # is disabled and the pair must prove substitution lexically instead.
    if a in b or b in a:
        return True                                   # 2. refinement / extension
    if functional and not temporal:
        return True                                   # 2b. Rule D: single-valued slot => replacement
    ta = {w for w in _re.findall(r"[a-z0-9']+", a) if len(w) > 2}
    tb = {w for w in _re.findall(r"[a-z0-9']+", b) if len(w) > 2}
    if not ta or not tb:
        return False
    if strict:
        return False                                  # aggregate intent: lexical overlap is NOT proof
    jac = len(ta & tb) / len(ta | tb)
    return jac >= 0.5                                 # 3. mostly-shared tokens = rewrite, else enumerate


_REL_MARGIN = float(_os.environ.get("UR_REL_MARGIN", "0.0"))   # G4b: 0.0 = v7 behaviour
_MARGIN_LOG = [] if _os.environ.get("UR_MARGIN_LOG") else None   # collect G4b margins for calibration
_OFF = {t.strip() for t in _os.environ.get("UR_DISABLE", "").split(",") if t.strip()}


def _on(gate):
    """Every gate must earn its place; UR_DISABLE lets one be removed and the loss measured."""
    return gate not in _OFF


_G4B_MODE = _os.environ.get("UR_G4B_MODE", "off")     # off | fixed | sigma | rho
_SIGMA_Z = float(_os.environ.get("UR_SIGMA_Z", "2.0"))
_SIGMA_MIN_N = int(_os.environ.get("UR_SIGMA_MIN_N", "30"))
_MSTAT = {"n": 0, "s1": 0.0, "s2": 0.0}


def _margin_observe(m):
    """Streaming mean/variance of the G4b margin. Label-free: it only sees encoder similarities."""
    _MSTAT["n"] += 1
    _MSTAT["s1"] += m
    _MSTAT["s2"] += m * m


def margin_sigma():
    """
    The encoder's tie-noise on same-slot pairs, estimated from this store's own margins.

    Returns None until MIN_N pairs have been observed. A warm start borrowed from another corpus would
    put that corpus's scale into this one's decisions for the first queries - a small leak, but this
    router is meant to carry no constant fitted anywhere else, so before it has calibrated it simply
    declines to use the threshold and falls back to the strict v7 behaviour. Cold start is therefore
    conservative (more vetoes, less pruning), never permissive.
    """
    n = _MSTAT["n"]
    if n < _SIGMA_MIN_N:
        return None
    mu = _MSTAT["s1"] / n
    var = max(0.0, _MSTAT["s2"] / n - mu * mu)
    return var ** 0.5


def g4b_threshold(rho=None):
    """The margin G4b must clear before it may veto."""
    if _G4B_MODE == "off":
        return 0.0
    if _G4B_MODE == "fixed":
        return _REL_MARGIN
    sig = margin_sigma()
    if sig is None:
        return 0.0                              # not calibrated yet -> strict v7 behaviour
    d = _SIGMA_Z * sig
    if _G4B_MODE == "rho":
        r = min(0.95, max(0.0, rho if rho is not None else 0.0))
        d *= r / (1.0 - r)                      # odds(rho): permissive where supersession is the norm
    return d
_AGG_CUE_REQUIRED = _os.environ.get("UR_AGG_CUE", "") == "1"  # G9: off = v7 behaviour


def _v(d, k):
    """Count one gate veto. A no-op when the caller did not ask for attribution."""
    if d is not None:
        d[k] = d.get(k, 0) + 1


PROMPT_PAIR = (
    "A memory store holds several statements about the same attribute, recorded at different times.\n"
    "For EACH numbered pair below, decide whether hiding the OLDER statement would make the question\n"
    "impossible to answer correctly.\n"
    "Hiding is SAFE when the newer statement replaces the older one (the attribute was updated).\n"
    "Hiding is UNSAFE when the question needs the older value too - because it asks for a past value,\n"
    "for how something changed, or because both values are simultaneously true and must be listed.\n"
    "Question: %s\n\n%s\n"
    "Answer with one line per pair, exactly: <number>: SAFE   or   <number>: UNSAFE")


def pair_safe_llm(q, pairs, value_of, ts_of, gen):
    """
    Ask, per candidate pair and WITH the two values visible, whether dropping the older one loses the
    answer. Returns the subset of pairs that may be suppressed.

    Batched into a single call so this costs exactly what the classification it replaces cost. An
    unparseable or missing verdict is read as UNSAFE: the router must never suppress because a backend
    hiccuped, which is the same V0 discipline the intent path already follows.
    """
    if not pairs:
        return set()
    lines = []
    for j, (lo, wi, lts, wts) in enumerate(pairs):
        lines.append("%d) older: %s\n   newer: %s"
                     % (j + 1, str(value_of.get(lo, ""))[:200], str(value_of.get(wi, ""))[:200]))
    raw = (gen(PROMPT_PAIR % (q, "\n".join(lines)), 8 * len(pairs) + 32) or "").lower()
    ok = set()
    for j, (lo, wi, lts, wts) in enumerate(pairs):
        m = _re.search(r"(?m)^\W*%d\s*[^a-z]{0,4}\s*(safe|unsafe)" % (j + 1), raw)
        if m and m.group(1) == "safe":
            ok.add(lo)
    return ok


def prune_safe(pairs, sim_source=None, topic_tau=0.5, protected=None, sim_query=None, values=None,
               strict_sub=False, temporal=False, slot_of=None, multi_slots=None, collective=False,
               veto_out=None, rel_threshold=None):
    """
    G2/G3/G4. `pairs` = list of (loser_id, winner_id, loser_ts, winner_ts) proposed suppressions.
    sim_source(a, b) -> cosine similarity between the evidence carrying a and b (None = skip G2).
    sim_query(x)     -> cosine similarity of item x to the query (None = skip the relevance guard).
    protected = ids that must never be suppressed (e.g. the top-1 retrieved item).
    Returns (ok, kept_suppressions).
    """
    protected = protected or set()
    keep = []
    for loser, winner, lts, wts in pairs:
        if _on("G4a") and loser in protected:
            _v(veto_out, "G4a_protected"); continue    # G4a: never drop the best-retrieved evidence
        if not (wts > lts):
            _v(veto_out, "G3_not_newer"); continue     # G3: winner must be strictly newer
        if _on("G2") and sim_source is not None and sim_source(loser, winner) < topic_tau:
            _v(veto_out, "G2_topic"); continue         # G2: different topic => false conflict
        if _on("G4b") and sim_query is not None:
            _m = sim_query(loser) - sim_query(winner)
            if _MARGIN_LOG is not None:
                _MARGIN_LOG.append(round(float(_m), 5))
            _margin_observe(float(_m))
            if _m > (rel_threshold if rel_threshold is not None else _REL_MARGIN):
                _v(veto_out, "G4b_query_rel"); continue  # G4b: loser more query-relevant -> keep it
        sl = (slot_of or {}).get(loser)
        if _on("RuleA") and multi_slots and sl in multi_slots:
            _v(veto_out, "RuleA_coassert"); continue   # Rule A: values co-asserted => they coexist
        fn = (slot_is_functional(sl) and not collective) if sl is not None else False
        #    ^ G7: a collective query asks for the enumeration, so Rule D must not fire
        if _on("G5") and values is not None and not substitutive(values.get(loser, ""), values.get(winner, ""),
                                                   strict=strict_sub, temporal=temporal, functional=fn):
            _v(veto_out, "G5_not_substitutive"); continue  # G5/G6: coexisting values -> keep both
        keep.append(loser)
    return (len(keep) > 0), set(keep)


PROMPT_BINARY = (
    "A memory store has recorded several values for the same attribute at different times.\n"
    "Conflicting values found:\n%s\n"
    "Question: %s\n\n"
    "To answer this question, is it enough to see only the MOST RECENT value of each attribute,\n"
    "or are the older values needed too?\n"
    "Answer exactly one word: RECENT (only the newest is needed) or ALL (older ones are needed too).")


def binary_view(q, suppressions, values, value_of, gen, mode):
    """
    Choose PRUNE or KEEP_ALL for the whole query. Returns (view, info).

    No suppression subset is computed: PRUNE means every superseded value goes, which is the arm that
    actually wins where pruning wins. rho is reported in every mode because it is free and it is what
    makes the no-harm property structural - with no candidate pairs there is nothing any predictor
    could do.
    """
    sup = suppressions or []
    rho = len(sup) / max(1, len(values) if values is not None else 1)
    info = {"intent": "binary/" + mode, "labels": [], "conflict": 0,
            "suppress": {l for (l, w, lt, wt) in sup if wt > lt},
            "rho": rho, "n_pairs": len(sup), "reason": ""}
    if not info["suppress"]:
        info["reason"] = "no-supersession"
        return KEEP_ALL, info                       # nothing to prune: the two views are identical
    if mode == "always_composite":
        info["suppress"] = set()
        info["reason"] = "pole/composite"
        return KEEP_ALL, info
    if mode == "always_prune":
        info["reason"] = "pole/prune"
        return PRUNE, info
    if mode == "rho":
        info["reason"] = "rho>0"                    # parameter-free: a supersession exists, so prune
        return PRUNE, info
    if mode == "conflict":
        # $0 lexical: the only queries that need a superseded value are those asking for a past value
        # or for an enumeration, and both say so. Same two cues the router already had, used once, at
        # query level, instead of as vetoes threaded through six per-pair gates.
        if has_past_reference(q) or is_collective_query(q) or has_aggregate_reference(q):
            info["suppress"] = set()
            info["reason"] = "conflict/keep(list-or-past-cue)"
            return KEEP_ALL, info
        info["reason"] = "conflict/prune"
        return PRUNE, info
    if mode == "llm":
        lines = []
        seen = set()
        for (l, w, lt, wt) in sup[:12]:
            k = (str(value_of.get(l, ""))[:120], str(value_of.get(w, ""))[:120])
            if k in seen:
                continue
            seen.add(k)
            lines.append("  older: %s\n  newer: %s" % k)
        r = (gen(PROMPT_BINARY % ("\n".join(lines), q), 8) or "").strip().lower()
        if "all" in r and "recent" not in r:
            info["suppress"] = set()
            info["reason"] = "llm/ALL->keep"
            return KEEP_ALL, info
        if "recent" in r:
            info["reason"] = "llm/RECENT->prune"
            return PRUNE, info
        info["suppress"] = set()                    # V0: an unparseable answer never prunes
        info["reason"] = "llm/unparsed->keep"
        return KEEP_ALL, info
    info["suppress"] = set()
    info["reason"] = "unknown-mode->keep"
    return KEEP_ALL, info


def route(q, gen, slots=None, values=None, suppressions=None, sim_source=None,
          topic_tau=0.5, protected=None, sim_query=None, intent=None, value_of=None, log=None,
          source_of=None, policy="gated", use_g1=False):
    """
    Full decision. Returns (view, info) where view in {KEEP_ALL, PRUNE} and info carries the
    intent, the conflict degree and the surviving suppression set (for PRUNE).
    KEEP_ALL always means the ANNOTATED keep-all view (composite): every retrieved item, with the
    current value per slot marked [CURRENT] — annotation was never harmful in any measured cell.
    `intent` may be supplied to reuse one classification across arms (saves the LLM call).
    """
    if policy == "validated":
        # reproduce the router this project validated: one label, no store-side gates, fail toward
        # prune. PRUNE here means the caller applies its FULL keep-newest-per-slot view.
        vi = intent if isinstance(intent, str) and intent in ("current", "list", "previous") \
            else classify_intent_validated(q, gen)
        info = {"intent": vi, "labels": [vi], "conflict": 0, "suppress": set(),
                "policy": "validated", "reason": "validated/" + vi}
        return (PRUNE if vi == "current" else KEEP_ALL), info
    _bm = _os.environ.get("UR_BINARY", "")
    if _bm:
        return binary_view(q, suppressions, values, value_of or {}, gen, _bm)
    if _os.environ.get("UR_PAIR_LLM") == "1":
        # The taxonomy is what this replaces, so it runs BEFORE classification: no labels, no lexicons,
        # one call per query. G3 (winner strictly newer) and G4a (never drop the top-retrieved item)
        # still apply - neither is a judgement the LLM is being asked for.
        info = {"intent": "pair-llm", "labels": [], "conflict": 0, "suppress": set(), "reason": ""}
        _sup = suppressions or []
        _cand = [(lo, wi, lts, wts) for (lo, wi, lts, wts) in _sup
                 if wts > lts and lo not in (protected or set())]
        _ts = {}
        for (lo, wi, lts, wts) in _sup:
            _ts[lo] = str(lts)
            _ts[wi] = str(wts)
        keep = pair_safe_llm(q, _cand, value_of or {}, _ts, gen)
        info["suppress"] = keep
        info["reason"] = "pair-llm/%d-of-%d" % (len(keep), len(_cand))
        info["rho"] = len(_sup) / max(1, len(values) if values is not None else 1)
        info["n_pairs"] = len(_sup)
        return (PRUNE if keep else KEEP_ALL), info
    if intent is None:
        labels = classify_intents(q, gen)
    elif isinstance(intent, str):
        labels = frozenset() if intent in ("unknown", "") else frozenset([intent])
    else:
        labels = frozenset(intent)
    info = {"intent": "+".join(sorted(labels)) if labels else "unknown",
            "labels": sorted(labels), "conflict": 0, "suppress": set(), "reason": ""}
    if not labels:                             # V0: never prune on a failed/unparseable classification
        info["reason"] = "intent-unknown->safe-default"
        return KEEP_ALL, info
    # ---- CONSERVATIVE POLICY JOIN: the most restrictive reported reading wins ----
    if _on("G9") and _AGG_CUE_REQUIRED and "aggregate" in labels and not has_aggregate_reference(q):
        labels = labels - {"aggregate"}           # G9: unsupported aggregate label dropped
        if not labels:
            labels = frozenset(["current"])
        info["labels"] = sorted(labels)
        info["intent"] = "+".join(sorted(labels))
        info["g9"] = "aggregate-rejected"
    if _on("G8") and "history" in labels and not has_past_reference(q):
        # G8: the veto forfeits every suppression, so it must be backed by the query, not by a
        # label alone. Unsupported `history` is dropped and recorded for the ablation table.
        labels = labels - {"history"}
        info["history_dropped"] = True
        info["intent"] = "+".join(sorted(labels)) if labels else "unknown"
        info["labels"] = sorted(labels)
        if not labels:
            # the history claim was REJECTED as unsupported, so acting as if it held would be
            # self-contradictory; the remaining reading is the current value (also the validated
            # router's default). V0 abstain still applies to an EMPTY/unparseable reply above.
            labels = frozenset(["current"])
            info["intent"], info["labels"] = "current", ["current"]
            info["reason"] = "history-rejected->current"
    if _on("HISTVETO") and "history" in labels:   # most restrictive: the old value IS the answer
        info["reason"] = "intent=history(veto)"
        return KEEP_ALL, info
    # intent is now `current` or `aggregate`. `aggregate` NO LONGER VETOES: an aggregate over an
    # OVERWRITTEN measurement still wants only its latest value, while ENUMERATIVE items must all
    # survive - and that is exactly the per-pair judgement G5 makes. Vetoing on the label made the
    # router a constant function on 2/3 LongMemEval subsets.
    strict_sub = _on("AGGSTRICT") and ("aggregate" in labels)   # aggregate => G5 must PROVE it,
    #                                                 even when `current` was also reported
    temporal = _on("G6") and is_temporal_query(q)     # G6
    collective = _on("G7") and is_collective_query(q)  # G7
    info["strict_sub"], info["temporal"], info["collective"] = strict_sub, temporal, collective
    if strict_sub and value_of is None:
        info["reason"] = "aggregate-without-values->safe-default"
        return KEEP_ALL, info
    # G1: the conflict must be on a slot we would actually act on (not any slot in the pool)
    if suppressions:
        act = {l for l, w, lt, wt in suppressions} | {w for l, w, lt, wt in suppressions}
        if slots is not None and values is not None:
            sl = [s for i, s in enumerate(slots) if i in act] if len(slots) == len(values) else slots
            vl = [v for i, v in enumerate(values) if i in act] if len(slots) == len(values) else values
            info["conflict"] = max_slot_conflict(sl, vl) or max_slot_conflict(slots, values)
        if use_g1 and info["conflict"] and info["conflict"] < 2:
            # G1 counted DISTINCT values, so duplicate facts looked like 'nothing to do' even though
            # dropping the repeated copy changes the reader's input. The tautological check below
            # answers the same question exactly, so G1 only costs correct prunes.
            info["reason"] = "no-conflict"
            return KEEP_ALL, info
        _slot_of = {i: slots[i] for i in range(len(slots))} if slots is not None else None
        _multi = cooccurring_slots(slots, values, source_of)
        info["multi_slots"] = len(_multi)
        _veto = {}
        # rho = how much of the retrieved pool is a superseded copy of something else in it. Label-free,
        # computed per query. At rho = 0 there is nothing to suppress and the router is a no-op.
        _rho = len(suppressions) / max(1, len(values) if values is not None else len(suppressions))
        _thr = g4b_threshold(_rho)
        ok, keep = prune_safe(suppressions, sim_source, topic_tau, protected, sim_query, value_of,
                              strict_sub=strict_sub, temporal=temporal,
                              slot_of=_slot_of, multi_slots=_multi, collective=collective,
                              veto_out=_veto, rel_threshold=_thr)
        info["veto"] = _veto
        info["n_pairs"] = len(suppressions)
        info["rho"] = _rho
        info["g4b_threshold"] = _thr
        info["sigma"] = margin_sigma()
        info["suppress"] = keep
        if not ok:
            # TAUTOLOGICAL KEEP_ALL: nothing survived the gates, so PRUNE would equal KEEP_ALL.
            # Returning KEEP_ALL here is what makes no-harm hold BY CONSTRUCTION, not by measurement.
            info["reason"] = "no-safe-pair(taut)"
            return KEEP_ALL, info
        info["reason"] = "prune/" + info["intent"] + ("/temporal" if temporal else "") + \
                         ("/collective" if collective else "")
        return PRUNE, info
    info["reason"] = "no-suppression-candidates"
    return KEEP_ALL, info
