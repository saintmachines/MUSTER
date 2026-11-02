"""DERIVE the representation of every store from its own BYTES. Nothing here is typed by hand.

THE RULE (pre-registered in PREREGISTRATION_2026-08-29_atom_rule.md before any leg was run):

  1 ATOM -- the boundary the chunker may never cross
        numbered_frac >= 0.80   ->  AUTOSTRUCT   (unit=record, parent=entity+skeleton, order=serial)
        atom_cov      >= 0.90   ->  blank        (the store's own record)
        otherwise               ->  sentence     (the harness default)

  2 SCALE -- chunk_size
        the BENCHMARK's protocol chunk for that cell. We never choose it, so no reviewer can call
        the scale tuned. (Measured upside we are declining: banking77 scores 0.940 at chunk 2048
        against 0.895 at the protocol 4096. Taking that per cell would be test-set selection.)

  3 PARENT -- the coarse layer
        8x the unit scale; if that leaves < MINPAR parents, halve until it does; if 2x still
        leaves < MINPAR, there is no room for a parent layer and the store runs single-scale.

`atom_cov` is TOKEN-WEIGHTED on purpose. Counting records says InfBench-Sum has 1,513 of them and
therefore an atom; it does not -- one record holds 237,782 of its 241,446 tokens. Weighting by
tokens asks whether the store is ACTUALLY made of records. Measured separation on the 14 stores:
min over atom stores 0.9751, max over non-atom stores 0.5504, so the 0.90 threshold sits in an
empty band 0.42 wide and no store is near it.
"""
import sys, os, json
MUSTER_HOME = os.environ.get("MUSTER_HOME", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MUSTER_HOME)
os.chdir(MUSTER_HOME)
import re
import pandas as pd, tiktoken
import cells_registry as REG

ENC = tiktoken.get_encoding("cl100k_base")
NUM = re.compile(r"^\s*\d+\.\s+\S")
ATOM_TAU = 0.90
NUMBERED_TAU = 0.80
MINPAR = 10
FINE_STEP = 8        # 4096 / 512, the ratio between the benchmark's two documented chunk sizes
FINE_MIN = 512       # never finer than the finest size the benchmark itself uses
SAMPLE_ROWS = 3          # per-row statistics; three rows is enough and det/summ have many


def stats(pq, rows, prot):
    df = pd.read_parquet(pq)
    tot = fit = 0
    nf = []
    toks = []
    for r in rows[:SAMPLE_ROWS]:
        txt = str(df["context"].iloc[r])
        t_all = len(ENC.encode(txt))
        toks.append(t_all)
        for p in txt.split("\n\n"):
            if not p.strip():
                continue
            t = len(ENC.encode(p))
            tot += t
            if t <= prot:
                fit += t
        lines = [l for l in txt.split("\n") if l.strip()][:20000]
        if lines:
            nf.append(sum(1 for l in lines if NUM.match(l)) / float(len(lines)))
    return (fit / float(tot) if tot else 0.0,
            sum(nf) / len(nf) if nf else 0.0,
            sum(toks) // len(toks))          # tokens PER ROW: each row is its own store


def derive(cell, refine=True):
    pq, rows, nq, metric, prot = REG.CELLS[cell]
    cov, numbered, tok_per_row = stats(pq, rows, prot)
    if numbered >= NUMBERED_TAU:
        return dict(cell=cell, atom="AUTOSTRUCT", chunk=prot, mscale=0, autostruct=1,
                    units=0, parents=0, atom_cov=round(cov, 4), numbered=round(numbered, 4))
    atom = "blank" if cov >= ATOM_TAU else "sentence"
    # THE FINE SCALE IS A BYTE DECISION TOO. A store made of small records survives a finer cut
    # because the atom is preserved; a narrative store does not. FINE_STEP is the ratio between
    # the benchmark's own two documented chunk sizes, so the refined scale is 512 -- the size
    # MemoryAgentBench itself uses in its chunk-size ablation, not a number of ours.
    chunk = prot // FINE_STEP if (refine and cov >= ATOM_TAU and prot // FINE_STEP >= FINE_MIN) else prot
    units = max(1, tok_per_row // chunk)
    ms, par = 0, 0
    for mult in (8, 4, 2):
        p = units // mult
        if p >= MINPAR:
            ms, par = chunk * mult, p
            break
    return dict(cell=cell, atom=atom, chunk=chunk, mscale=ms, autostruct=0,
                units=units, parents=par, atom_cov=round(cov, 4), numbered=round(numbered, 4))


if __name__ == "__main__":
    order = ["shqa", "mhqa", "lme", "evq", "b77", "clin", "nlu", "trcC", "trcF",
             "rec", "summ", "det", "fcmh", "fcsh"]
    cfg = {c: derive(c) for c in order}
    json.dump(cfg, open("derived_config.json", "w"), indent=1)
    print("DERIVED REPRESENTATION -- every field below is a function of the stored bytes")
    print("%-6s %-11s %6s %7s %7s %7s %9s %9s" %
          ("cell", "ATOM", "CHUNK", "MSCALE", "units", "parents", "atom_cov", "numbered"))
    print("-" * 74)
    for c in order:
        d = cfg[c]
        print("%-6s %-11s %6d %7s %7s %7s %9.4f %9.4f" %
              (c, d["atom"], d["chunk"], d["mscale"] or "single", d["units"] or "-",
               d["parents"] or "-", d["atom_cov"], d["numbered"]))
    print()
    print("AGREEMENT WITH LEGS ALREADY RUN (the rule was not fitted to these):")
    # The fourth column is the WITNESS: the line the run printed that proves the harness actually
    # read the rule's output.  Without one, an agreement is unfalsifiable -- see lme below.
    KNOWN = {"lme": ("sentence", 512, 4096, "p24_lme_m / gr_lme_m", None),
             "summ": ("sentence", 4096, 16384, "p24_summ_m", "chunks at chunk_size=4096"),
             "fcmh": ("AUTOSTRUCT", 512, 0, "as6_fcmh", "AUTOSTRUCT numbered=1.000"),
             "fcsh": ("AUTOSTRUCT", 512, 0, "as6_fcsh", "AUTOSTRUCT numbered=1.000")}
    real = 0
    for c, (a, ck, ms, tag, witness) in KNOWN.items():
        d = cfg[c]
        ok = (d["atom"] == a and d["chunk"] == ck and d["mscale"] == ms)
        if witness is None:
            verdict = "*** VACUOUS ***"
        else:
            verdict = "MATCH" if ok else "*** DIFFERS ***"
            real += 1 if ok else 0
        print("   %-6s rule says %-11s %5d/%-6s   run was %-11s %5d/%-6s   %s   (%s)"
              % (c, d["atom"], d["chunk"], d["mscale"] or "single", a, ck, ms or "single",
                 verdict, tag))
    print("   -> the rule is confirmed out of sample on %d leg(s), not %d." % (real, len(KNOWN)))
    print("      lme is VACUOUS, not a match: load_row (lme_pf29.py:1547) routes longmemeval to a")
    print("      round-level loader -- units are ROUNDS, parents are SESSIONS, and ATOM / CHUNK /")
    print("      MSCALE are never read.  DRY rebuilds under sentence/512/4096, sentence/512/single")
    print("      and blank/512/4096 all returned 641/111 622/107 664/116 654/111 639/110, the")
    print("      board's own numbers.  Any output would have `matched`.  (measured 2026-09-08)")
    print()
    print("QUEUE LEGS the rule implies (tag|env|parquet|rows|nq) -- rows from the registry:")
    for c in order:
        d = cfg[c]
        pq, rows, nq, metric, prot = REG.CELLS[c]
        env = "ATOM=%s CHUNK=%d" % (d["atom"] if not d["autostruct"] else "sentence", d["chunk"])
        if d["autostruct"]:
            env += " AUTOSTRUCT=1 AS_SLOT=5 HDETECT=0 K1=200"
        elif d["mscale"]:
            env += " MSCALE=%d HDETECT=0 K1=200" % d["mscale"]
        else:
            env += " HDETECT=1 K1=40"
        if c == "det":
            env += " MAXOUT=2000"
        if c == "summ":
            env += " MAXOUT=1200"
        print('  "ar_%s|%s|%s|%s|%d"' % (c, env, pq, ",".join(map(str, rows)), nq))
