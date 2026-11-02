#!/usr/bin/env python3
# real_e1_conflict.py — MemoryAgentBench CONFLICT RESOLUTION track (factconsolidation_mh_6k).
# The store = a numbered list of facts WITH CONFLICTS: the same (subject, relation) "slot"
# appears at multiple positions with different values (a simulated knowledge update). Multi-hop
# questions require the RESOLVED value. Zero-LLM supersession = keep ONE fact per slot (latest
# position = newest). Tests whether deterministic read-time conflict resolution helps a fixed
# reader. Full-context (all facts to the reader) so the ONLY variable is conflict handling.
# Arms: OFF (all facts incl conflicts) / ON_latest (keep newest per slot) / ON_earliest (control)
#       / FOIL (LLM prose summary = lossy consolidation). Metric = SubEM. Requires GEMINI key.
from __future__ import annotations
import argparse, glob, os, re, sys, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from real_e1_reader import ask, _client, MODEL
from metrics import subem
from stats import paired_bootstrap, significant

FACT_RE = re.compile(r'^\s*(\d+)\.\s*(.+?)\s*$')
_CAP = re.compile(r"\b([A-Z][\w.'-]+(?:\s+(?:of|the|and|de|van|von|di)?\s*[A-Z][\w.'-]+)*)\b")
_STOP = {"The", "A", "An", "In", "On", "At", "Of", "This", "That", "These", "Those", "It", "Its", "Which",
         "What", "Who", "Where", "When", "There", "He", "She", "They", "His", "Her", "Their"}
VAL_RE = re.compile(r'\s+[A-Z][^.]*\.\s*$')   # legacy regex slot (main-paper key; ablation only)


def _ents_ordered(text):
    return [m.strip() for m in _CAP.findall(text or "")
            if len(m.strip()) > 2 and m.strip().split()[0] not in _STOP]


def _norm(e):
    return re.sub(r'[^a-z0-9]', '', e.lower())


def regex_slot(text):
    """Main-paper slot key: fact minus its trailing ' Capitalized value.'"""
    return VAL_RE.sub('', text).rstrip(' .').strip()


def normalized_slot(text):
    """Canonical RTR slot key: normalized leading entity + relation skeleton (entities masked)."""
    es = _ents_ordered(text)
    subj = _norm(es[0]) if es else ""
    skel = re.sub(r'\s+', ' ', _CAP.sub('_', text)).strip().rstrip('.').strip()
    return subj + "|" + skel


# Default = normalized-subject key (canonical RTR). Set RTR_SLOT=regex to reproduce the main paper.
_SLOT_FN = {"regex": regex_slot, "normalized": normalized_slot}[
    os.environ.get("RTR_SLOT", "normalized").lower()]


def parse_facts(ctx, slot_fn=None):
    slot_fn = slot_fn or _SLOT_FN
    out = []
    for line in ctx.splitlines():
        m = FACT_RE.match(line)
        if not m:
            continue
        pos, text = int(m.group(1)), m.group(2).strip()
        out.append((pos, text, slot_fn(text)))
    return out


def resolve(facts, keep='latest'):
    best = {}
    for pos, text, slot in facts:
        if slot not in best or (keep == 'latest') == (pos > best[slot][0]):
            best[slot] = (pos, text)
    return [t for _, t in sorted(best.values())]


def summarize(fact_block):
    prompt = ("Rewrite the following list of facts as a concise connected prose paragraph "
              "that preserves the key relationships. Do not use a numbered list.\n\n" + fact_block)
    for att in range(6):
        try:
            r = _client.models.generate_content(model=MODEL, contents=prompt)
            return (r.text or "").strip()
        except Exception:
            if att == 5:
                return fact_block[:2000]
            time.sleep(2 * (att + 1))


def golds_of(a):
    xs = list(a) if hasattr(a, "__len__") and not isinstance(a, str) else [a]
    return [str(x) for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq_per_row", type=int, default=12)
    ap.add_argument("--rows", default="all")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    df = pd.read_parquet(glob.glob("data_all/**/Conflict_Resolution*.parquet", recursive=True)[0])
    rows = range(len(df)) if a.rows == "all" else [int(x) for x in a.rows.split(",")]
    rng = random.Random(a.seed)
    arms = ["OFF", "ON_latest", "ON_earliest", "FOIL"]
    per = {arm: [] for arm in arms}
    n = 0
    print(f"reader={MODEL} | Conflict Resolution (factconsolidation)", flush=True)
    for row in rows:
        ctx = str(df["context"].iloc[row])
        facts = parse_facts(ctx)
        slots = [s for _, _, s in facts]
        n_conf = len(slots) - len(set(slots))            # duplicated slots = conflicts
        txt = {
            "OFF": "\n".join(f"{p}. {t}" for p, t, _ in sorted(facts)),
            "ON_latest": "\n".join(resolve(facts, 'latest')),
            "ON_earliest": "\n".join(resolve(facts, 'earliest')),
        }
        txt["FOIL"] = summarize(txt["OFF"])              # one lossy summary per row
        qs = list(df["questions"].iloc[row]); ans = list(df["answers"].iloc[row])
        idx = list(range(len(qs))); rng.shuffle(idx); idx = idx[:a.nq_per_row]
        print(f"row {row}: {len(facts)} facts, {n_conf} conflicting-slot dups, {len(qs)} Q -> {len(idx)} sampled", flush=True)
        for qi in idx:
            q = str(qs[qi]); g = golds_of(ans[qi])
            for arm in arms:
                per[arm].append(subem(ask(txt[arm], q), g))
            n += 1
            if n % 20 == 0:
                print(f"  {n} Q done", flush=True)
    M = lambda x: sum(x) / len(x) if x else 0.0
    d_on = [per["ON_latest"][i] - per["OFF"][i] for i in range(n)]
    d_ea = [per["ON_earliest"][i] - per["OFF"][i] for i in range(n)]
    c_on, c_ea = paired_bootstrap(d_on, 5000, 1), paired_bootstrap(d_ea, 5000, 1)
    print(f"\n=== CONFLICT RESOLUTION reader SubEM (n={n}) ===", flush=True)
    print("  " + "  ".join(f"{arm}={M(per[arm]):.3f}" for arm in arms), flush=True)
    print(f"  Δ(ON_latest − OFF) ={c_on[0]:+.3f}[{c_on[1]:+.3f},{c_on[2]:+.3f}]{' *' if significant(c_on) else ''}"
          "   <== zero-LLM supersession (keep latest per slot) resolves conflicts for the reader?", flush=True)
    print(f"  Δ(ON_earliest − OFF)={c_ea[0]:+.3f}[{c_ea[1]:+.3f},{c_ea[2]:+.3f}]{' *' if significant(c_ea) else ''}"
          "   (control: keep earliest — should hurt if latest=correct)", flush=True)
    print(f"  FOIL(prose summary) SubEM={M(per['FOIL']):.3f} vs OFF={M(per['OFF']):.3f} (non-monotonicity: lossy consolidation)", flush=True)


if __name__ == "__main__":
    main()
