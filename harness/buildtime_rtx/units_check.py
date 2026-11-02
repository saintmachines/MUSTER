#!/usr/bin/env python3
"""CPU check of the units the timed build cuts. No torch, no model, no network. Per store it runs exactly the calls that
time_beam_build_rtx.py times for statistics, units and graph (SD.derive, K.configure, SL.units_row, SL.graph_row) and
folds the unit texts, the embedded texts and the parent ids into sha256 digests (beambt_common.digest_update), then
compares every store with expected.json: counts, digests, EMB_MAXLEN, the leg's env, the derive() output and the graph
counts. Exit 1 on any difference. expected.json is not written here. It holds the harness's own values (lme_pf29.py
imported with the reported leg's env, its load_row on the leg's parquet row 0, derive_config.derive and the graph lines on
that row), so this check compares this folder's cut with the harness.
--report PATH writes what this run computed, for the record.
Usage: BM25LC=1 PYTHONHASHSEED=0 python -B units_check.py [--cells a,b] [--data DIR] [--expected PATH] [--report PATH]"""
import argparse, hashlib, json, os, platform, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(HERE, "data"))
ap.add_argument("--expected", default=os.path.join(HERE, "expected.json"))
ap.add_argument("--cells", default="")
ap.add_argument("--report", default="")
A = ap.parse_args()
os.environ["BEAMBT_DATA"] = os.path.abspath(A.data)
os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(os.path.abspath(A.data), "tiktoken_cache")
assert os.environ.get("BM25LC") == "1" and os.environ.get("PYTHONHASHSEED") == "0", "launch with BM25LC=1 PYTHONHASHSEED=0"
assert sys.flags.hash_randomization == 0, "PYTHONHASHSEED=0 must take effect (do not run with -I or -E)"
import pandas as pd
import beambt_common as K
import ship_derive as SD
import ship_lme as SL
REG = SD.REG
STORES_J = K.stores()
ENV = K.shipped_env(STORES_J)
K.register(REG, STORES_J)
CELLS = [c for c in K.ORDER if not A.cells or c in A.cells.split(",")]
EXP = json.load(open(A.expected))
assert EXP["order"] == K.ORDER, "expected.json and beambt_common disagree on the store order"
# the verbatim blocks must be the recorded bytes, cut from the harness named in expected.json
_MAN = json.load(open(os.path.join(HERE, "MANIFEST_copies.json")))
for _n in ("ship_lme.py", "ship_derive.py", "cells_registry.py", "MANIFEST_copies.json"):
    assert K.sha256(os.path.join(HERE, _n)) == EXP["package_verbatim"][_n], ("differs from the recorded copy", _n)
assert _MAN["sources"]["lme_pf29.py"]["sha256"] == EXP["harness"]["lme_pf29.py"], "the blocks come from another harness"
assert _MAN["sources"]["derive_config.py"]["sha256"] == EXP["harness"]["derive_config.py"], "derive_config.py differs"
t0 = time.time()
SL.chunk_official.__globals__["print"] = lambda *a, **k: None   # silence the chunker's per-call lines

STORES, SEC = {}, {}
for c in CELLS:
    s = STORES_J["stores"][c]
    assert s["sha256"] == EXP["inputs"][s["file"]]["sha256"], ("input bytes differ", c)
    df = K.store_df(c, STORES_J)
    SD._DF[REG.CELLS[c][0]] = df
    pq, rows, nq, metric, prot = REG.CELLS[c]
    t = time.time()
    d = SD.derive(c); ts = time.time() - t
    maxlen = K.configure(SL, ENV[c])
    hU, hE, hP = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    n_units = n_par = n_emb = chars = 0
    graph = None
    tu = tg = 0.0
    for ROW in rows:
        t = time.time(); rtext, rkey, par, order, ds = SL.units_row(df, ROW); tu += time.time() - t
        t = time.time(); graph = dict(SL.graph_row(rtext, rkey, par, order, ds)); tg += time.time() - t
        n_units += len(rtext); n_par += len(set(par)); n_emb += len(rkey); chars += sum(len(x) for x in rkey)
        K.digest_update(hU, ROW, rtext); K.digest_update(hE, ROW, rkey); K.digest_update(hP, ROW, [str(p) for p in par])
    STORES[c] = {"name": K.NAME[c], "file": s["file"], "rows": rows, "units": n_units, "parents": n_par,
                 "texts_embedded": n_emb, "chars_embedded": chars, "units_sha256": hU.hexdigest(),
                 "embedded_sha256": hE.hexdigest(), "parents_sha256": hP.hexdigest(), "emb_maxlen": maxlen,
                 "leg_env": dict(ENV[c]), "derive": d, "graph": graph,
                 "hdet_info": json.loads(json.dumps(SL.HDET_INFO))}
    SEC[c] = {"statistics": round(ts, 3), "units": round(tu, 3), "graph": round(tg, 3)}
    print("%-9s %-9s units %6d parents %5d embedded %6d maxlen %4d | statistics %.1f s units %.1f s graph %.1f s"
          % (c, K.NAME[c], n_units, n_par, n_emb, maxlen, ts, tu, tg), flush=True)

KEYS = ("units", "parents", "texts_embedded", "chars_embedded", "units_sha256", "embedded_sha256", "parents_sha256",
        "emb_maxlen", "leg_env", "derive", "graph", "hdet_info")
bad = 0
for c in CELLS:
    diff = {k: (STORES[c][k], EXP["stores"][c][k]) for k in KEYS
            if json.loads(json.dumps(STORES[c][k])) != EXP["stores"][c][k]}
    if diff:
        bad += 1
        print("DIFF %s %s" % (c, json.dumps(diff)[:4000]), flush=True)
print("nltk imported: %s, torch imported: %s" % ("nltk" in sys.modules, "torch" in sys.modules))
print("units check: %d of %d stores equal expected.json (%.0f s)" % (len(CELLS) - bad, len(CELLS), time.time() - t0))

if A.report:
    with open(A.report, "w") as fh:
        json.dump({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "python": sys.version.split()[0], "exe": sys.executable,
                   "platform": platform.platform(), "home": os.environ.get("HOME"),
                   "versions": {"pandas": pd.__version__, "numpy": __import__("numpy").__version__,
                                "tiktoken": __import__("importlib.metadata").metadata.version("tiktoken"),
                                "rank_bm25": __import__("importlib.metadata").metadata.version("rank-bm25")},
                   "nltk_imported": "nltk" in sys.modules, "torch_imported": "torch" in sys.modules,
                   "expected_sha256": K.sha256(A.expected), "equal": len(CELLS) - bad, "stores_checked": len(CELLS),
                   "seconds": SEC, "stores": STORES}, fh, indent=1)
sys.exit(1 if bad else 0)
