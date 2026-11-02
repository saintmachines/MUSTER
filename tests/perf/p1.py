#!/usr/bin/env python3
"""P1: bounded CPU measurements with fake models and synthetic text.

Run on the benchmark machine, from a checkout:
  PYTHONHASHSEED=0 python tests/perf/p1.py --case all --out p1.json

Every case runs in a fresh process. Outputs contain relative source names and
source hashes, never machine account names or absolute checkout paths.
"""
import argparse
import cProfile
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def _peak_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 ** 2 if platform.system() == "Darwin" else 1024)


def _rss_mib():
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True,
                            check=True, timeout=10)
    return int(result.stdout.strip()) / 1024


def _distribution(samples):
    ordered = sorted(samples)
    # Nearest rank, with no interpolation between observations.
    import math
    return {"n": len(samples), "p50_ms": statistics.median(samples) * 1000,
            "p99_ms": ordered[max(0, math.ceil(0.99 * len(ordered)) - 1)] * 1000}


def _identity():
    files = sorted((ROOT / "muster").rglob("*.py")) + [ROOT / "muster/presets/paper.json"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    hardware = {}
    if platform.system() == "Darwin":
        for key in ("hw.model", "hw.physicalcpu", "hw.memsize"):
            result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True, timeout=10)
            hardware[key] = result.stdout.strip()
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
            "created_utc": datetime.now(timezone.utc).isoformat(), "hardware": hardware,
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                       "VECLIB_MAXIMUM_THREADS")},
            "versions": {n: importlib.metadata.version(n) for n in ("numpy", "tiktoken", "nltk", "rank-bm25")},
            "source_sha256": hashes, "hash_seed": os.environ.get("PYTHONHASHSEED"),
            "models": "Models.fake(dim=16)", "device": "cpu"}


def add_case(count):
    from muster import Store
    from muster.models import Models

    models = Models.fake()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("add called a retrieval model")

    models.encoder.encode = forbidden
    models.reranker.score = forbidden
    records = [{"text": "Sensor record %d: pressure is stable; the sample stays in the blue cabinet." % i}
               for i in range(count)]
    samples = []
    with tempfile.TemporaryDirectory(prefix="muster-p1-add-") as scratch:
        with Store.open(str(Path(scratch) / "single.muster"), models=models) as store:
            for record in records:
                start = time.perf_counter()
                result = store.add(record)
                samples.append(time.perf_counter() - start)
            assert result.version == count and store.version == count
            assert [r.text for r in store.records()] == [r["text"] for r in records]
        with Store.open(str(Path(scratch) / "batch.muster"), models=models) as store:
            start = time.perf_counter()
            result = store.add(records)
            batch_s = time.perf_counter() - start
            assert result.version == count
            assert [r.text for r in store.records()] == [r["text"] for r in records]
    return dict(case="add", single_record=_distribution(samples), batch_total_s=batch_s,
                batch_per_record_ms=batch_s * 1000 / count, model_calls=len(calls),
                peak_rss_mib=_peak_mib(), sqlite_journal="WAL", sqlite_synchronous="FULL")


def _fixture(target):
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    # Serial values vary; adjacent records share sectors. ASCII keeps the final
    # token-boundary truncation independent of multibyte character handling.
    estimate = max(1, target // 42)
    while True:
        text = "\n\n".join("Sector%03d sensor record %08d: pressure is stable and temperature is normal. "
                           "The sample stays in the blue cabinet; the next inspection is at 09:30."
                           % ((i // 64) % 128, i) for i in range(estimate))
        tokens = enc.encode(text)
        if len(tokens) >= target:
            text = enc.decode(tokens[:target])
            del tokens
            break
        estimate = int(estimate * 1.3) + 1
    assert len(enc.encode(text)) == target
    return text, [{"text": part} for part in text.split("\n\n")]


def rebuild_case(tokens, queries, profile):
    from muster import Store
    from muster.cut import cut
    from muster.derive import derive
    from muster.index import Index
    from muster.models import Models

    text, records = _fixture(tokens)
    plan = derive(text)
    gc.collect()
    before = _rss_mib()
    start = time.perf_counter()
    c = cut(text, plan)
    cut_s = time.perf_counter() - start
    start = time.perf_counter()
    index = Index(c.units, c.keys, c.parent, c.order, has_order=plan.branch == "numbered")
    index_s = time.perf_counter() - start
    start = time.perf_counter()
    _ = index.bm25
    bm25_s = time.perf_counter() - start
    cut_index_s = cut_s + index_s + bm25_s
    units, parents = c.units, c.parent
    del c, index
    gc.collect()
    with tempfile.TemporaryDirectory(prefix="muster-p1-rebuild-") as scratch:
        with Store.open(str(Path(scratch) / "head.muster"), models=Models.fake()) as store:
            store.add(records)
            start = time.perf_counter()
            head = store.head()
            _ = head.index.bm25
            head_s = time.perf_counter() - start
            assert head.text == text and head.units == units and head.parent == parents
            samples = []
            for i in range(queries):
                query = "What is the pressure in Sector%03d for inspection %d?" % (i % 128, i)
                start = time.perf_counter()
                context = store.context(query)
                samples.append(time.perf_counter() - start)
                assert context.text and len(set(context.ids)) == len(context.ids)
            query = "What is the pressure in Sector005 for inspection repeat?"
            cold_start = time.perf_counter()
            first = store.context(query)
            cold_s = time.perf_counter() - cold_start
            repeats = []
            for _ in range(queries):
                start = time.perf_counter()
                repeated = store.context(query)
                repeats.append(time.perf_counter() - start)
                assert repeated.text == first.text
            after = _rss_mib()
            top, search_top = [], []
            if profile:
                profiler = cProfile.Profile()
                profiler.enable()
                cc = cut(text, plan)
                ii = Index(cc.units, cc.keys, cc.parent, cc.order)
                _ = ii.bm25
                profiler.disable()
                stats = pstats.Stats(profiler)
                for (filename, line, name), value in sorted(stats.stats.items(), key=lambda item: -item[1][3])[:15]:
                    top.append({"function": "%s:%d:%s" % (Path(filename).name, line, name),
                                "calls": value[1], "self_s": value[2], "cumulative_s": value[3]})
                profiler = cProfile.Profile()
                profiler.runcall(store.context, query)
                for (filename, line, name), value in sorted(pstats.Stats(profiler).stats.items(),
                                                          key=lambda item: -item[1][3])[:15]:
                    search_top.append({"function": "%s:%d:%s" % (Path(filename).name, line, name),
                                       "calls": value[1], "self_s": value[2], "cumulative_s": value[3]})
            return dict(case="rebuild", o200k_tokens=tokens, records=len(records), units=len(units),
                        parents=len(set(parents)), branch=plan.branch, U=plan.U, cut_s=cut_s, index_s=index_s,
                        bm25_s=bm25_s, cut_index_s=cut_index_s, store_head_with_fake_embed_s=head_s,
                        rss_before_rebuild_mib=before, rss_after_search_mib=after, peak_rss_mib=_peak_mib(),
                        unique_query_search=_distribution(samples), repeat_search=_distribution(repeats),
                        first_repeat_query_ms=cold_s * 1000, context_sha256=hashlib.sha256(first.text.encode()).hexdigest(),
                        profile_top=top, profile_search_top=search_top)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "add", "rebuild"), default="all")
    parser.add_argument("--tokens", type=int, default=100_000)
    parser.add_argument("--adds", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--deadline", type=int, default=600)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(args.tokens, args.adds, args.queries, args.deadline) < 1:
        parser.error("counts and deadline must be positive")
    _emit({"event": "start", "pid": os.getpid(), "case": args.case})
    if args.case == "all":
        cases = [("add", 100_000)] + [("rebuild", n) for n in (100_000, 1_000_000, 10_000_000)]
        results = []
        for case, tokens in cases:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--case", case, "--tokens", str(tokens),
                   "--adds", str(args.adds), "--queries", str(args.queries)]
            if tokens == 100_000 and case == "rebuild":
                cmd.append("--profile")
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.deadline)
            except subprocess.TimeoutExpired:
                _emit({"event": "FAIL", "case": case, "tokens": tokens, "reason": "deadline"})
                return 1
            if proc.returncode:
                _emit({"event": "FAIL", "case": case, "tokens": tokens,
                       "reason": "child failed", "stderr_tail": proc.stderr[-1200:]})
                return 1
            child = [json.loads(line) for line in proc.stdout.splitlines()]
            result = next(value for value in reversed(child) if "measurement" in value)["measurement"]
            results.append(result)
            _emit({"event": "case_done", "measurement": result})
        receipt = {"identity": _identity(), "measurements": results}
    else:
        measured = add_case(args.adds) if args.case == "add" else rebuild_case(args.tokens, args.queries, args.profile)
        receipt = {"identity": _identity(), "measurement": measured}
    if args.out:
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
