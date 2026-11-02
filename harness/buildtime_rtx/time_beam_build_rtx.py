#!/usr/bin/env python3
"""MUSTER's build time on BEAM conversation 0 (100K, 1M, 10M) on one NVIDIA RTX PRO 6000, the driver behind
gate/build_time_rtx.md. No key and no generative call. The build path is the reported BEAM legs' (len_legs.tsv lines 19 to
24, lz_u20 and lz_u40 at 100K, 1M and 10M, row 0 = conversation 0), cut by ship_lme.py and ship_derive.py, which hold the
lines of lme_pf29.py and derive_config.py verbatim (verify_copies.py). The model is read from the local Hugging Face cache
(HF_HUB_OFFLINE=1).

Per store, timed separately (time.perf_counter):
  statistics  derive(store): the two rule statistics (atom_cov, numbered_frac) and the tokens of the store's one row,
              then the rule (derive_config.py:43-87 via ship_derive; the store is read once, before timing)
  units       units and parents of the row with the leg's env (load_row, lme_pf29.py:2015-2267, MSCALE branch 2198-2238, via
              ship_lme.units_row; includes the two shipped assertions, the re-cut (2221), timed apart as recut, and
              the non-space character check (2227-2230, three re.sub passes), timed apart as nonspace)
  embed       one bge-m3 pass over exactly the texts the shipped build embeds (rkey, lme_pf29.py:2411-2412)
  bm25        BM25Okapi over the unit texts (lme_pf29.py:2508)
  graph       the entity index and slot maps (lme_pf29.py:2419-2481), timed apart and added for the build time the paper
              reports (total_with_graph_s)
Model load and warm-up are timed apart and excluded from every build time.
Load and warm-up on cuda: max_seq_length 448, model.half(), then encode(['warm up'] * 64, batch_size=32); each store then
sets its leg's EMB_MAXLEN, 2048 on all three (len_legs.tsv lines 19 to 24). torch.cuda.empty_cache() after the row and
between stores, outside the timed sections.
Before any time of a store is reported, its units, parents, statistics output (derive), graph counts and the sha256 of its
unit texts (and of the embedded texts and the parent ids) must equal expected.json, which holds the harness's own values
(lme_pf29.py imported with the leg's env, its load_row on the leg's parquet row 0, on the CPU). Otherwise the run stops.
After the repeats, one fp32 embed pass per store in the same process, saved in its own file and never added to the build
totals: a second SentenceTransformer at the same revision and no .half(), the precision of the scored legs
(lme_pf29.py:2276), max_seq_length 2048, over the same units (checked against expected.json). Its model load and warm-up
are timed apart. beam_buildtime_rtx_fp32.json, embed_fp32_s. --no-fp32 skips it.
After each repeat, beam_buildtime_rtx_repeatN.json and summary.md are written to --out. --resume skips a repeat (and the
fp32 file) already saved there with the same setup (expected.json, stores, device, the sha256 of every code file, the
installed library versions, the GPU name and memory), and summary.md averages only the repeats of that setup.
Without CUDA the run refuses before anything is loaded or timed (exit 6, REFUSED_<tag>.json in --out). On cuda the GPU must
be an RTX PRO 6000 with at least --min-gpu-mem-mib (90,000 MiB), unless --force.
Usage: python -B time_beam_build_rtx.py --out RESULTS_DIR [--runs 3] [--resume] [--force] [--cells a,b] [--no-fp32]
CPU smoke test, never a measurement: python -B time_beam_build_rtx.py --device cpu --emb-limit 16 --tag smoke --out DIR"""
import argparse, gc, hashlib, json, os, platform, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--cells", default="")
ap.add_argument("--tag", default="rtx")
ap.add_argument("--out", required=True)
ap.add_argument("--data", default=os.path.join(HERE, "data"))
ap.add_argument("--expected", default=os.path.join(HERE, "expected.json"))
ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
ap.add_argument("--emb-limit", type=int, default=0)                                     # CPU smoke test only
ap.add_argument("--resume", action="store_true")
ap.add_argument("--force", action="store_true")
ap.add_argument("--max-gpu-used-mib", type=int, default=2048)
ap.add_argument("--min-gpu-mem-mib", type=int, default=90000)
ap.add_argument("--min-free-pct", type=int, default=50)
ap.add_argument("--no-fp32", action="store_true")
A = ap.parse_args()
assert (A.device == "cpu") == (A.emb_limit > 0), "the CPU path is the smoke test and needs --emb-limit (and only it)"
os.environ["BEAMBT_DATA"] = os.path.abspath(A.data)                                    # read by beambt_common
os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(os.path.abspath(A.data), "tiktoken_cache")

for k, v in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("BM25LC", "1"), ("PYTHONHASHSEED", "0")):
    assert os.environ.get(k) == v, "launch with %s=%s" % (k, v)
_KEYS = [k for k in os.environ if re.search(r"(API_KEY|_TOKEN|SECRET)", k)]
assert not _KEYS, ("no keys in this environment", _KEYS)
assert sys.flags.hash_randomization == 0, "PYTHONHASHSEED=0 must take effect (do not run with -I or -E)"
pc = time.perf_counter
LOG = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256f(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def free_pct():                                     # MemAvailable / MemTotal from /proc/meminfo
    try:
        m = {}
        for ln in open("/proc/meminfo"):
            f = ln.split()
            m[f[0].rstrip(":")] = int(f[1])
        return int(100 * m["MemAvailable"] / m["MemTotal"])
    except Exception:
        return None


def gpu_state():                                    # what nvidia-smi says about the GPU (None on the CPU smoke test)
    if A.device != "cuda":
        return None
    q = "name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm,clocks.max.sm,power.draw"
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=30).stdout.strip().splitlines()[0]
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except Exception as e:
        return {"error": str(e)}
    d = dict(zip(q.split(","), [x.strip() for x in o.split(",")]))
    d["compute_apps"] = [x.strip() for x in a if x.strip()]
    return d


def proc_check(tag, first=False):
    """Before our own CUDA start, FAIL if the GPU already holds more than --max-gpu-used-mib (another job on it); every
    time, FAIL if system free memory is below the floor. The GPU state is recorded each time."""
    viol = []
    g = gpu_state()
    if first and g and "memory.used" in g:
        try:
            if float(g["memory.used"]) > A.max_gpu_used_mib:
                viol.append({"why": "GPU memory already used before this run: %s MiB" % g["memory.used"],
                             "apps": g.get("compute_apps")})
        except ValueError:
            pass
    fp = free_pct()
    if fp is not None and fp < A.min_free_pct:
        viol.append({"why": "system free %d%% < %d%%" % (fp, A.min_free_pct)})
    info = {"tag": tag, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "free_pct": fp, "loadavg": list(os.getloadavg()),
            "gpu": g, "violations": viol}
    LOG("proc_check %s: free %s%%, load %s, gpu %s, violations %d"
        % (tag, fp, [round(x, 2) for x in os.getloadavg()],
           None if not g else {k: g.get(k) for k in ("memory.used", "utilization.gpu", "temperature.gpu", "clocks.sm")},
           len(viol)))
    for v in viol:
        LOG("   VIOLATION", v)
    return info


def save_json(p, d):                                # write, flush, fsync, then rename (a synced folder keeps whole files)
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh, indent=1)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    try:
        os.replace(tmp, p)
    except OSError:
        with open(p, "w") as fh:
            json.dump(d, fh, indent=1)
        try:
            os.remove(tmp)
        except OSError:
            pass


os.makedirs(A.out, exist_ok=True)
# no time without CUDA. Checked first, before any file is read or any model is loaded.
if A.device == "cuda":
    import torch
    if not torch.cuda.is_available():
        save_json(os.path.join(A.out, "REFUSED_%s.json" % A.tag),
                  {"REFUSED": "no CUDA device in this runtime; nothing was loaded or timed", "torch": torch.__version__,
                   "torch_cuda": torch.version.cuda, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv,
                   "python": sys.version.split()[0], "platform": platform.platform()})
        LOG("REFUSED: torch %s sees no CUDA device, so nothing is loaded or timed. A build time is measured only on the "
            "GPU (Colab G4, RTX PRO 6000)." % torch.__version__)
        sys.exit(6)
EXP = json.load(open(A.expected))
EXP_SHA = sha256f(A.expected)
ORDER0 = EXP["order"]
CELLS = [c for c in ORDER0 if not A.cells or c in A.cells.split(",")]
RPATH = lambda run: os.path.join(A.out, "beam_buildtime_rtx_repeat%d.json" % run)
FPATH = os.path.join(A.out, "beam_buildtime_rtx_fp32.json")
CODE_FILES = ("ship_lme.py", "ship_derive.py", "cells_registry.py", "MANIFEST_copies.json", "beambt_common.py",
              "time_beam_build_rtx.py", "summarize_rtx.py")
SRC_SHA = {n: sha256f(os.path.join(HERE, n)) for n in CODE_FILES}
SRC_SHA["stores.json"] = sha256f(os.path.join(HERE, "stores.json"))
import importlib.metadata as _md


def dist_version(n):                                # installed versions (package metadata, not the imported modules)
    try:
        return _md.version(n)
    except Exception:
        return None


DIST = {n: dist_version(n) for n in ("torch", "transformers", "sentence-transformers", "tokenizers", "huggingface-hub",
                                     "numpy", "pandas", "pyarrow", "rank-bm25", "tiktoken")}
GPU_ID = None
if A.device == "cuda":                              # the BEAM table's GPU, the full card, before anything loads
    _g = gpu_state() or {}
    GPU_ID = {"name": _g.get("name"), "memory_total_mib": _g.get("memory.total")}
    try:
        _mem = float(_g.get("memory.total"))
    except (TypeError, ValueError):
        _mem = 0.0
    if not A.force and ("RTX PRO 6000" not in str(_g.get("name")) or _mem < A.min_gpu_mem_mib):
        LOG("STOP: this runtime's GPU is %r with %s MiB. The measurement needs an RTX PRO 6000 with at least %d MiB "
            "(Colab's G4 runtime), or --force." % (_g.get("name"), _g.get("memory.total"), A.min_gpu_mem_mib))
        sys.exit(5)
SETUP = {"expected_sha256": EXP_SHA, "cells": CELLS, "tag": A.tag, "device": A.device, "emb_limit": A.emb_limit,
         "sources": SRC_SHA, "versions": DIST, "gpu": GPU_ID}


def saved_ok(run):                                  # AUTO_RESUME accepts only a complete repeat of this exact setup
    try:
        d = json.load(open(RPATH(run)))
    except Exception:
        return False
    return d.get("complete") is True and d.get("setup") == SETUP


def extra_ok(p):                                    # the same rule for the fp32 file
    try:
        d = json.load(open(p))
    except Exception:
        return False
    return d.get("complete") is True and d.get("setup") == SETUP


TODO = [r for r in range(1, A.runs + 1) if not (A.resume and saved_ok(r))]
DO_FP32 = not A.no_fp32 and not (A.resume and extra_ok(FPATH))
import summarize_rtx as SUMM
if not TODO and not DO_FP32:
    SUMM.write(A.out, want=A.runs, setup=SETUP)
    LOG("all %d repeats%s are already saved in %s (AUTO_RESUME); summary.md rewritten"
        % (A.runs, "" if A.no_fp32 else ", the fp32 pass", A.out))
    sys.exit(0)
for r in range(1, A.runs + 1):
    if r not in TODO:
        LOG("repeat %d already saved in %s, skipped (AUTO_RESUME)" % (r, RPATH(r)))
if not A.no_fp32 and not DO_FP32:
    LOG("fp32 embed pass already saved in %s, skipped (AUTO_RESUME)" % FPATH)

META = {"tag": A.tag, "pid": os.getpid(), "argv": sys.argv, "python": sys.version.split()[0], "exe": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(), "node": platform.node(),
        "session": time.strftime("%Y%m%dT%H%M%S") + "_%d" % os.getpid(), "device": A.device, "emb_limit": A.emb_limit,
        "cpu_count": os.cpu_count(), "passes": {"fp32": not A.no_fp32},
        "env": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "BM25LC", "PYTHONHASHSEED",
                                                "TOKENIZERS_PARALLELISM", "TIKTOKEN_CACHE_DIR", "OMP_NUM_THREADS",
                                                "CUDA_VISIBLE_DEVICES")}}
if A.emb_limit:
    META["TEST_ONLY"] = ("SMOKE: CPU code-path test with the embedding limited to %d texts per row; not a measurement, "
                         "never reported" % A.emb_limit)
pre = proc_check("start", first=True)
META["preflight"] = pre
if pre["violations"]:
    save_json(os.path.join(A.out, "FAIL_%s_preflight.json" % A.tag),
              {"meta": META, "FAIL": "preflight", "violations": pre["violations"]})
    LOG("FAIL preflight"); sys.exit(2)

import pandas as pd
import beambt_common as K
import ship_derive as SD
import ship_lme as SL
REG = SD.REG
STORES = K.stores()
ENV = K.shipped_env(STORES)
assert ORDER0 == K.ORDER, "expected.json and beambt_common disagree on the store order"
K.register(REG, STORES)                                                                 # the three stores as cells

# the verbatim blocks must be the recorded bytes and the harness the one named in expected.json
META["sources"] = dict(SRC_SHA)
META["sources"]["expected.json"] = EXP_SHA
_MAN = json.load(open(os.path.join(HERE, "MANIFEST_copies.json")))
for _n in ("ship_lme.py", "ship_derive.py"):
    assert META["sources"][_n] == _MAN["outputs"][_n]["sha256"] == EXP["package_verbatim"][_n], ("verbatim copy changed", _n)
for _n in ("cells_registry.py", "MANIFEST_copies.json"):
    assert META["sources"][_n] == EXP["package_verbatim"][_n], ("source differs from the recorded copy", _n)
assert _MAN["sources"]["lme_pf29.py"]["sha256"] == EXP["harness"]["lme_pf29.py"], "the blocks come from another harness"
# the stores, read once (not part of any build total)
DF, READ = {}, {}
for c in CELLS:
    s = STORES["stores"][c]
    assert s["sha256"] == EXP["inputs"][s["file"]]["sha256"], ("input bytes differ", c)
    assert {k: ENV[c].get(k) for k in EXP["stores"][c]["leg_env"]} == EXP["stores"][c]["leg_env"], ("leg env differs", c)
    t = pc(); DF[c] = K.store_df(c, STORES); READ[c] = round(pc() - t, 3)
    SD._DF[REG.CELLS[c][0]] = DF[c]
META["store_read_s"] = READ

# the recut probe: chunk_official is wrapped (the function text stays verbatim); in the MSCALE branch the first call cuts
# the units and the second is the shipped assertion `assert ch == chunk_official(...)` (lme_pf29.py:2221)
CO = []
_co = SL.chunk_official


def _timed_co(*a, **k):
    t = pc(); r = _co(*a, **k); CO.append(pc() - t); return r


SL.chunk_official = _timed_co

# The non-space probe. While units_row runs (and only then), ship_lme's module name `re` is a proxy whose sub()
# times the calls re.sub(r"\s+", "", text) and passes everything else to the re module (the function text stays
# verbatim). In the MSCALE branch they are exactly the three calls of the shipped check's `_ns` (lme_pf29.py:2227-2230):
# over the text and over the joined units (line 2228), and over the text again in the assertion's condition (2229). Each
# call records the line it came from, which must be the `_ns` line of ship_lme.py. units_nonspace_s is the span from the
# start of the first call to the end of the third, so it holds the join and the comparison between them too. Like the
# re-cut, it stays inside units_s.
NS = []
_RE = SL.re
_NS_SRC = r'        _ns = lambda t: len(re.sub(r"\s+", "", t))'                         # lme_pf29.py:2227 in ship_lme.py
_NS_LINE = [i + 1 for i, ln in enumerate(open(os.path.join(HERE, "ship_lme.py")).read().split("\n")) if ln == _NS_SRC]
assert len(_NS_LINE) == 1, ("the non-space check of lme_pf29.py:2227 is not in ship_lme.py once", _NS_LINE)
_NS_LINE = _NS_LINE[0]


class _ReProbe:
    def __getattr__(self, n):
        return getattr(_RE, n)

    @staticmethod
    def sub(pattern, repl, string, *a, **k):
        if pattern == r"\s+" and repl == "" and not a and not k:
            t0 = pc(); r = _RE.sub(pattern, repl, string); t1 = pc()
            f = sys._getframe(1)
            NS.append((t0, t1, os.path.basename(f.f_code.co_filename), f.f_lineno))
            return r
        return _RE.sub(pattern, repl, string, *a, **k)


RE_PROBE = _ReProbe()

import torch, transformers, sentence_transformers
from sentence_transformers import SentenceTransformer
META["versions"] = {"torch": torch.__version__, "sentence_transformers": sentence_transformers.__version__,
                    "transformers": transformers.__version__, "numpy": __import__("numpy").__version__,
                    "rank_bm25": _md.version("rank-bm25"), "tiktoken": _md.version("tiktoken"),
                    "pandas": pd.__version__, "pyarrow": _md.version("pyarrow"),
                    "tokenizers": _md.version("tokenizers"), "huggingface_hub": _md.version("huggingface-hub")}
if A.device == "cuda":
    assert torch.cuda.is_available(), "no CUDA device in this runtime"
    _gpu = torch.cuda.get_device_name(0)
    if "RTX PRO 6000" not in _gpu and not A.force:
        LOG("STOP: this runtime's GPU is %r, not an RTX PRO 6000. Choose the G4 runtime, or pass --force." % _gpu)
        sys.exit(5)
    META["cuda"] = {"gpu": _gpu, "capability": list(torch.cuda.get_device_capability(0)), "torch_cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(), "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1),
                    "driver": (pre.get("gpu") or {}).get("driver_version")}
SYNC = torch.cuda.synchronize if A.device == "cuda" else (lambda: None)
EMPTY = torch.cuda.empty_cache if A.device == "cuda" else (lambda: None)
BGE = os.environ.get("MIRROR_BGE", "BAAI/bge-m3")      # real_e1_lme.py:17, the name lme_pf29.py:2276 loads
assert BGE == "BAAI/bge-m3"
REV = EXP["model"]["revision"]                         # the pinned revision
emb = None
if TODO:                                               # the fp16 model serves the repeats
    t = pc()
    emb = SentenceTransformer(BGE, device=A.device, revision=REV)   # lme_pf29.py:2276 (device cuda, revision pinned)
    emb.max_seq_length = 448                                        # for the load and the warm-up only
    if A.device == "cuda":
        emb.half()                                                  # fp16
    SYNC()
    META["model_load_s"] = round(pc() - t, 3)
    META["model"] = {"name": BGE, "revision": REV, "device": str(emb.device), "dtype": str(next(emb.parameters()).dtype),
                     "class": type(emb[0].auto_model).__name__,
                     "attn_implementation": getattr(emb[0].auto_model.config, "_attn_implementation", None),
                     "loaded_from": "the local files of the pinned revision (the download is not included)"}
    assert A.device != "cuda" or META["model"]["dtype"] == "torch.float16", META["model"]
    LOG("MODEL LOAD %.2f s (excluded from every build time) %s" % (META["model_load_s"], META["model"]))
    # the warm-up, one call before any timing
    t = pc()
    emb.encode(["warm up"] * 64, batch_size=32)
    SYNC()
    META["warmup_s"] = round(pc() - t, 3)
    EMPTY()
    LOG("warm-up %.2f s (excluded)" % META["warmup_s"])


def gb(x):
    return round(x / 1024 ** 3, 3)


ABORT = [None]
UKEYS = {}
first_in_process = True
for run in TODO:                                                                        # the repeats not saved yet
    chk = META["preflight"] if first_in_process else proc_check("run%d" % run)
    REP = {"meta": META, "run": run, "tag": A.tag, "cells": CELLS, "expected_sha256": EXP_SHA, "complete": False,
           "setup": SETUP, "first_repeat_in_process": first_in_process, "proc_check": chk,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stores": {}}
    first_in_process = False
    if chk["violations"]:
        REP["FAIL"] = "proc_check before run %d" % run
        save_json(os.path.join(A.out, "FAIL_%s_run%d.json" % (A.tag, run)), REP); LOG("FAIL before run", run); sys.exit(2)
    LOG("READY: repeat %d of %d starts (%d stores: %s)" % (run, A.runs, len(CELLS), ", ".join(K.NAME[c] for c in CELLS)))
    for c in CELLS:
        pq, rows, nq, metric, prot = REG.CELLS[c]
        df = DF[c]
        env = ENV[c]
        T = {"statistics_s": 0.0, "units_s": 0.0, "embed_s": 0.0, "bm25_s": 0.0, "graph_s": 0.0, "units_recut_s": 0.0,
             "units_nonspace_s": 0.0}
        t = pc(); d = SD.derive(c); T["statistics_s"] = pc() - t
        maxlen = K.configure(SL, env)
        assert maxlen == EXP["stores"][c]["emb_maxlen"] == 2048, ("BEAM legs embed at EMB_MAXLEN=2048", c, maxlen)
        emb.max_seq_length = maxlen
        n_units = n_par = n_emb = chars = n_done = 0
        hU, hE, hP = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        if A.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        gpu_peak = 0
        per_row = []
        keys_rows = []
        graph = None
        for ROW in rows:
            CO.clear(); NS.clear()
            SL.re = RE_PROBE
            try:
                t = pc(); rtext, rkey, par, order, ds = SL.units_row(df, ROW); u = pc() - t
            finally:
                SL.re = _RE
            rc = CO[1] if (SL.MSCALE and len(CO) >= 2) else 0.0
            assert not SL.MSCALE or (len(NS) == 3 and all(x[2:] == ("ship_lme.py", _NS_LINE) for x in NS)), \
                ("the non-space probe saw other calls than the 3 of lme_pf29.py:2228-2229", [x[2:] for x in NS])
            ns = (NS[-1][1] - NS[0][0]) if SL.MSCALE else 0.0
            ekey = rkey[:A.emb_limit] if A.emb_limit else rkey                          # CPU smoke test only
            t = pc(); kv = SL.embed_row(emb, ekey); SYNC(); e = pc() - t
            t = pc(); g = SL.graph_row(rtext, rkey, par, order, ds); gr = pc() - t
            t = pc(); bm = SL.bm25_row(rtext); b = pc() - t
            assert kv.shape == (len(ekey), 1024), kv.shape
            T["units_s"] += u; T["units_recut_s"] += rc; T["units_nonspace_s"] += ns; T["embed_s"] += e
            T["graph_s"] += gr; T["bm25_s"] += b
            n_units += len(rtext); n_par += len(set(par)); n_emb += len(rkey); chars += sum(len(x) for x in rkey)
            n_done += len(ekey)
            K.digest_update(hU, ROW, rtext); K.digest_update(hE, ROW, rkey)               # untimed
            K.digest_update(hP, ROW, [str(p) for p in par])
            keys_rows.append((ROW, rkey))
            graph = dict(g)                                                             # one row per store
            per_row.append({"row": ROW, "units": len(rtext), "parents": len(set(par)), "embedded": len(ekey),
                            "units_s": round(u, 4), "recut_s": round(rc, 4), "nonspace_s": round(ns, 4),
                            "nonspace_calls": len(NS), "embed_s": round(e, 4), "graph_s": round(gr, 4),
                            "bm25_s": round(b, 4)})
            del kv, bm, g
            if A.device == "cuda":
                gpu_peak = max(gpu_peak, torch.cuda.max_memory_allocated())
            EMPTY()                                     # as the harness does after every row (cuda form)
        # the store must be the harness's, unit for unit, before any of its times is reported
        X = EXP["stores"][c]
        got = {"units": n_units, "parents": n_par, "texts_embedded": n_emb, "units_sha256": hU.hexdigest(),
               "embedded_sha256": hE.hexdigest(), "parents_sha256": hP.hexdigest(),
               "derive": json.loads(json.dumps(d)), "graph": json.loads(json.dumps(graph))}   # + graph counts
        bad = {k: {"got": got[k], "expected": X[k]} for k in got if got[k] != X[k]}
        if bad:
            REP["FAIL"] = "store %s differs from expected.json" % c
            save_json(os.path.join(A.out, "FAIL_%s_run%d_%s.json" % (A.tag, run, c)),
                      {"meta": META, "run": run, "store": c, "differences": bad})
            LOG("STOP: store %s (%s) does not match expected.json: %s. No time is reported for it. The inputs or the "
                "pinned library versions differ from the recorded ones; nothing after this store was measured."
                % (c, K.NAME[c], json.dumps(bad)[:3000]))
            sys.exit(4)
        UKEYS[c] = keys_rows
        total = T["statistics_s"] + T["units_s"] + T["embed_s"] + T["bm25_s"]
        S = {k: round(v, 4) for k, v in T.items()}
        S.update(total_s=round(total, 4), total_with_graph_s=round(total + T["graph_s"], 4), rows=len(rows),
                 units=n_units, parents=n_par, texts_embedded=n_emb, texts_embedded_now=n_done, chars_embedded=chars,
                 emb_maxlen=maxlen, units_sha256=got["units_sha256"], embedded_sha256=got["embedded_sha256"],
                 parents_sha256=got["parents_sha256"], matches_expected=True, graph=graph,
                 derive=d, leg_env={k: env[k] for k in ("ATOM", "CHUNK", "MSCALE", "AUTOSTRUCT", "AS_SLOT", "EMB_MAXLEN")
                                    if k in env}, gpu_peak_allocated_gb=gb(gpu_peak), per_row=per_row)
        EMPTY()
        fp = free_pct()
        S["free_pct_after"] = fp
        REP["stores"][c] = S
        LOG("run %d %-8s stats %.3f units %.3f (recut %.3f nonspace %.3f) embed %.3f bm25 %.3f | total %.3f | graph %.3f | units %d emb %d | gpu %.1f GB free %s%%"
            % (run, c, T["statistics_s"], T["units_s"], T["units_recut_s"], T["units_nonspace_s"], T["embed_s"],
               T["bm25_s"], total, T["graph_s"], n_units, n_done, gb(gpu_peak), fp))
        if ABORT[0] is None and fp is not None and fp < A.min_free_pct:
            ABORT[0] = "system free %d%% < %d%% after %s" % (fp, A.min_free_pct, c)
        if ABORT[0]:
            REP["FAIL"] = ABORT[0]
            save_json(os.path.join(A.out, "FAIL_%s_run%d.json" % (A.tag, run)), REP); LOG("ABORT", ABORT[0]); sys.exit(3)
    tot = {k: round(sum(s[k] for s in REP["stores"].values()), 3)
           for k in ("statistics_s", "units_s", "units_recut_s", "units_nonspace_s", "embed_s", "bm25_s", "graph_s",
                     "total_s", "total_with_graph_s")}
    REP["totals"] = tot
    REP["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    REP["gpu_after"] = gpu_state()
    REP["complete"] = True
    save_json(RPATH(run), REP)                                                          # after each repeat
    SUMM.write(A.out, want=A.runs, setup=SETUP)
    LOG("RUN %d DONE totals %s" % (run, tot))
    LOG("saved %s and %s" % (RPATH(run), os.path.join(A.out, "summary.md")))


# the fp32 embed pass. A second SentenceTransformer at the same revision and no .half(), which is the precision
# the scored legs embedded in (lme_pf29.py:2276), max_seq_length 2048, over the same units. One pass per store, saved
# as embed_fp32_s, never added to the fp16 build totals.
if DO_FP32:
    chk = proc_check("fp32")
    REPF = {"meta": META, "tag": A.tag, "pass": "fp32", "cells": CELLS, "expected_sha256": EXP_SHA, "complete": False,
            "setup": SETUP, "proc_check": chk, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "after_repeats_in_process": [r for r in TODO], "stores": {}}
    if chk["violations"]:
        REPF["FAIL"] = "proc_check before the fp32 pass"
        save_json(os.path.join(A.out, "FAIL_%s_fp32.json" % A.tag), REPF); LOG("FAIL before the fp32 pass"); sys.exit(2)
    if emb is not None:                                 # the fp16 model is released first
        emb = None
        gc.collect()
        EMPTY()
    t = pc()
    emb32 = SentenceTransformer(BGE, device=A.device, revision=REV)   # lme_pf29.py:2276, no .half()
    emb32.max_seq_length = 2048                                       # EMB_MAXLEN of the legs, as lme_pf29.py:2276 sets it
    SYNC()
    REPF["model_load_s"] = round(pc() - t, 3)
    REPF["model"] = {"name": BGE, "revision": REV, "device": str(emb32.device),
                     "dtype": str(next(emb32.parameters()).dtype), "class": type(emb32[0].auto_model).__name__,
                     "attn_implementation": getattr(emb32[0].auto_model.config, "_attn_implementation", None),
                     "loaded_from": "the local files of the pinned revision (the download is not included)"}
    assert REPF["model"]["dtype"] == "torch.float32", REPF["model"]
    REPF["float32_math"] = {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                            "set_by_this_script": False}
    LOG("fp32 MODEL LOAD %.2f s (excluded) %s %s" % (REPF["model_load_s"], REPF["model"], REPF["float32_math"]))
    t = pc()
    emb32.encode(["warm up"] * 64, batch_size=32)       # the same warm-up as the fp16 model, excluded
    SYNC()
    REPF["warmup_s"] = round(pc() - t, 3)
    EMPTY()
    LOG("fp32 warm-up %.2f s (excluded)" % REPF["warmup_s"])
    for c in CELLS:
        maxlen = K.configure(SL, ENV[c])
        assert maxlen == 2048, (c, maxlen)
        keys_rows = UKEYS.get(c)
        cut_again = keys_rows is None
        if cut_again:                                   # resumed in a new process: the same cut again, untimed
            keys_rows = []
            for ROW in REG.CELLS[c][1]:
                _rt, _rk, _p, _o, _d = SL.units_row(DF[c], ROW)
                keys_rows.append((ROW, _rk))
        hE = hashlib.sha256()
        for ROW, rk in keys_rows:
            K.digest_update(hE, ROW, rk)
        if hE.hexdigest() != EXP["stores"][c]["embedded_sha256"]:
            save_json(os.path.join(A.out, "FAIL_%s_fp32_%s.json" % (A.tag, c)),
                      {"meta": META, "pass": "fp32", "store": c,
                       "differences": {"embedded_sha256": {"got": hE.hexdigest(),
                                                           "expected": EXP["stores"][c]["embedded_sha256"]}}})
            LOG("STOP: the fp32 pass would embed other texts than the harness's for %s. No fp32 time is recorded." % c)
            sys.exit(4)
        emb32.max_seq_length = maxlen
        if A.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        e32, n_now, n_all, peak = 0.0, 0, 0, 0
        for ROW, rk in keys_rows:
            ekey = rk[:A.emb_limit] if A.emb_limit else rk                              # CPU smoke test only
            t = pc(); kv = SL.embed_row(emb32, ekey); SYNC(); e32 += pc() - t
            assert kv.shape == (len(ekey), 1024), kv.shape
            n_now += len(ekey); n_all += len(rk)
            del kv
            if A.device == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated())
            EMPTY()
        REPF["stores"][c] = {"embed_fp32_s": round(e32, 4), "texts_embedded": n_all, "texts_embedded_now": n_now,
                             "emb_maxlen": maxlen, "embedded_sha256": hE.hexdigest(), "matches_expected": True,
                             "units_cut_again_untimed": cut_again, "gpu_peak_allocated_gb": gb(peak)}
        LOG("fp32 %-8s embed %.3f s | texts %d | gpu %.1f GB" % (c, e32, n_now, gb(peak)))
    REPF["totals"] = {"embed_fp32_s": round(sum(s["embed_fp32_s"] for s in REPF["stores"].values()), 3)}
    REPF["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    REPF["gpu_after"] = gpu_state()
    REPF["complete"] = True
    save_json(FPATH, REPF)
    SUMM.write(A.out, want=A.runs, setup=SETUP)
    LOG("fp32 DONE totals %s" % REPF["totals"])
    LOG("saved %s and %s" % (FPATH, os.path.join(A.out, "summary.md")))
LOG("ALL RUNS DONE")
