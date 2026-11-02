# CPU performance, P1

These are measurements of package overhead with `Models.fake(dim=16)`, not results
for bge-m3, a live reranker, or retrieval quality. The input is synthetic sensor text
with varying record numbers and groups of sector names. It is not a benchmark corpus.

The run used a Mac Studio (`Mac15,14`, arm64, 28 physical CPU cores, 256 GiB RAM),
CPython 3.12.13, NumPy 2.4.4, tiktoken 0.13.0, NLTK 3.10.0, and rank-bm25 0.2.2.
The process had niceness 10. OpenMP, OpenBLAS, and Accelerate thread limits were each
set to 1. No other MUSTER replay was running in the reserved CPU slot. No GPU, weights,
LLM, provider API, or paid service was used. Each size ran in a fresh child process.

## Appending records

10,000 calls to `Store.add`, each with one record, in a file-backed SQLite Store:

| Measurement | Value |
|---|---:|
| Calls | 10000 |
| p50 latency, ms | 0.0372 |
| p99 latency, ms | 0.0627 |
| Retrieval-model calls | 0 |
| Process peak RSS, MiB | 38.266 |
| Batch control: all records in one add, seconds | 0.0542 |
| Batch control: amortized ms per record | 0.00542 |

SQLite used WAL and `synchronous=FULL`. Both the single-record and batch controls
were checked for the expected record count and stored text in order. This timing
measures append transactions; the next search also rebuilds a changed Store head.

## Rebuilding and searching

The input sizes below are exact `o200k_base` token counts. All three derive the record
branch, use native mode and `U=10`, and pack at 512 tokens per unit.

| Input tokens | Records | Units | Parents |
|---:|---:|---:|---:|
| 100000 | 2858 | 205 | 25 |
| 1000000 | 28572 | 2041 | 245 |
| 10000000 | 285715 | 20409 | 2442 |

| Input tokens | Cut, s | Entity index, s | BM25, s | Cut + both indexes, s | Store head with fake embedding, s |
|---:|---:|---:|---:|---:|---:|
| 100000 | 0.0389 | 0.0098 | 0.0478 | 0.0964 | 0.1494 |
| 1000000 | 0.4017 | 0.0962 | 0.2547 | 0.7526 | 1.5389 |
| 10000000 | 4.0156 | 0.9680 | 2.4069 | 7.3904 | 15.1857 |

Cut/index timing starts with text and a derived plan already in memory. It excludes
fixture generation and plan derivation. The Store-head column includes SQLite record
reading, serialization, derivation, cutting, fake embedding, vector-cache writes,
entity indexing, and first BM25 construction. It excludes initial batch insertion.
Each Store head was checked against the independently cut text, units, and parents.

Search starts with that head and BM25 ready. There are 30 distinct-query observations
per size, followed by a separate query whose exact same text is repeated 30 times.
The first search of that repeat query is excluded from its cached distribution.

| Input tokens | Distinct-query p50, ms | Distinct-query p99, ms | Repeat p50, ms | Repeat p99, ms |
|---:|---:|---:|---:|---:|
| 100000 | 5.503 | 7.255 | 1.947 | 1.982 |
| 1000000 | 17.010 | 19.466 | 10.564 | 10.723 |
| 10000000 | 98.239 | 100.732 | 87.676 | 116.399 |

p50 is the median. p99 uses the nearest rank; for 30 observations it is the maximum,
so it is not a reliable estimate of a production tail. Every repeated-query context
was checked byte for byte against its first result. Timings exclude hardware/model
startup and do not predict real-model search latency.

| Input tokens | RSS before rebuild, MiB | RSS after search, MiB | Process peak RSS, MiB |
|---:|---:|---:|---:|
| 100000 | 134.188 | 150.328 | 152.031 |
| 1000000 | 166.922 | 263.078 | 263.078 |
| 10000000 | 665.641 | 1242.219 | 1258.719 |

RSS includes the Python process, generated input, source records, independently cut
units retained for comparison, caches, and the live Store head. Peak RSS also includes
fixture generation; it is not the index's standalone memory footprint. The small-size
process includes a separate profiling pass. Temporary Store files were removed.

## Profile and evidence

The separate 100K profile identifies BPE encoding/counting, BM25 token preparation,
and entity extraction as rebuild costs. Search profiling and the complete function
timings are included in the raw receipt. Profiling is outside the reported timing
intervals. No optimization was applied to the shared library and no speedup is claimed.

- Script: [tests/perf/p1.py](../../tests/perf/p1.py).
- Raw observations, profiles, environment, and source SHA256 values:
  [tests/perf/p1_baseline.json](../../tests/perf/p1_baseline.json).
- These numbers apply to the source manifest in that receipt. A future library change
  needs its own run; the paper's real-model context replay is a separate check.

From a checkout on the benchmark machine:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONHASHSEED=0 nice -n 10 python tests/perf/p1.py \
  --case all --deadline 240 --out p1.json
```

The parent runs cases sequentially and prints `FAIL` if a child times out or exits
with an error. The default case sizes are 100K, 1M, and 10M tokens. CPU availability,
storage, cache state, and synthetic text shape affect these results. Large real-model
rebuilds, other store shapes, and downstream agent outcomes were not measured here.
