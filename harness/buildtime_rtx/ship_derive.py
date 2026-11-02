"""derive_config.py, cut for timing. Do not edit.
Every block between '# >>> COPY' and '# <<< COPY' is the named lines of the harness verbatim, which verify_copies.py
checks; the rest is glue. The one change: stats() reads the store from _DF (filled by the caller once per store, before
timing) instead of pd.read_parquet(pq) on every call (derive_config.py:44)."""
import sys, os, json                                        # glue: derive_config.py:25
import re                                                   # glue: derive_config.py:29
import pandas as pd, tiktoken                               # glue: derive_config.py:30
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # glue: this folder's copy of cells_registry.py
import cells_registry as REG                                # glue: derive_config.py:31
_DF = {}                                                    # glue: parquet path (as in cells_registry) -> DataFrame
# >>> COPY derive_config.py:33-40 dedent=0
ENC = tiktoken.get_encoding("cl100k_base")
NUM = re.compile(r"^\s*\d+\.\s+\S")
ATOM_TAU = 0.90
NUMBERED_TAU = 0.80
MINPAR = 10
FINE_STEP = 8        # 4096 / 512, the ratio between the benchmark's two documented chunk sizes
FINE_MIN = 512       # never finer than the finest size the benchmark itself uses
SAMPLE_ROWS = 3          # per-row statistics; three rows is enough and det/summ have many
# <<< COPY


# >>> COPY derive_config.py:43-43 dedent=0
def stats(pq, rows, prot):
# <<< COPY
    df = _DF[pq]   # glue: replaces derive_config.py:44 `df = pd.read_parquet(pq)`; the store is read once, before timing
# >>> COPY derive_config.py:45-64 dedent=0
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
# <<< COPY


# >>> COPY derive_config.py:67-87 dedent=0
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
# <<< COPY
