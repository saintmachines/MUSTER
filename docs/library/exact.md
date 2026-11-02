# Exact mode and how the library is checked against the paper

`muster.build(text, "<preset>")` builds a store under one of the paper's nine presets (`muster presets`) in **exact
mode**: every step runs the paper's harness statements in the harness's order, with the same container types, so set
iteration, float sums and tie orders are the published ones. It needs `PYTHONHASHSEED=0` at interpreter start (the
library refuses otherwise) and is verified with the package versions in `muster/compat/env.py` on CPython 3.12.13.

Without an explicit plan, non-numbered stores use balanced floors plus immediate sibling candidates (`flrs`).
Published presets keep the policy of the recorded run: for example, `beam.u20` and `locomo.u10` use `flr`.
A plan supplied directly or restored from JSON is used unchanged. A Store opened without a plan derives each new
head from its records; pass an explicit plan to retain a previous policy.

**Native mode** (the default for applications), for the same plan, needs no hash seed and no pinned environment. It
departs from exact mode only where the harness's choice depended on hash or hardware order or where the harness stops:

| # | Native mode | Harness | Can it change a published context? |
|---|---|---|---|
| N1 | entity sets iterated in a canonical order | set order under hash seed 0 | only on IDF ties and last-bit weight sums |
| N2 | exact dense-score ties broken by unit index (stable sort) | argsort with no tie rule | only on exact ties (one on BEAM-100K at U=40) |
| N3 | special-token strings read as plain text | the packer stops with an error | no: no gated store holds one |
| N4 | CR and CRLF become LF | kept | no: no gated store holds one |
| N5 | an oversize record is cut on character boundaries | cut on token boundaries, a split character becomes U+FFFD | only where a cut split a character |
| N6 | stores too small for a parent layer are accepted | refused | no: the published runs passed |
| N7 | R4 renders the selected units | R4 renders units recovered from an intermediate text, which can name another copy of an identical text | yes, on stores with identical-text units (3 of 800 BEAM-100K questions) |

## Checks on the paper's data

All checks run on the experiment machine, CPU only, with no API call. The reranker scores come read-only from the
published runs' caches; a question whose scores are not all cached is reported as cold and not compared.

| Check | Script | Questions or rows | Equal |
|---|---|---|---|
| Cut of every row of the 24 final legs, against the harness's own `load_row` | `tests/studio/s2_cut_vs_harness.py` | 296 rows | 296 |
| ... rows whose unit list names the published run's reranker-cache file (a hash of the exact unit list) | same | 296 rows | 201 (the other 95 have no cache file on the machine) |
| Render chain of the MemoryAgentBench board rebuilt from the published dumps | `tests/studio/g0d_render_chain.py` | 5,439 question reads | 5,439 |
| ... LoCoMo and LoCoMo-Refined U=10 bundles and their R4 renders | same | 4 files | 4 byte-identical |
| BEAM-100K search, U=20: channel heads, pool, selected units, reach counts vs the harness CPU replay | `tests/studio/g1_beam100k.py` | 400 | 400 |
| BEAM-100K search, U=40, same fields | same | 400 | 400 |
| BEAM-100K R4 context (exact chain) vs the harness CPU replay | same | 800 | 800 |
| BEAM-100K R4 context (exact chain) vs the published contexts | same | 800 | 799 (U=40, row 4, question 6: an exact dense-score tie, 0.414321989 for two units, that CPU and CUDA order differently; the harness's own CPU replay differs from the published context there too) |
| SH-QA (flrs, flat), every field and the published context | `tests/studio/replay_legs.py` | 100 | 100 |
| banking77 (flrs, flat), context = the board's recorded hash | same | 100 | 100 |
| CLINC150, cached questions | same | 80 | 80 |
| EventQA (flr, U=40, bgroup then R4), context = published | same | 100 | 100 |
| DetectiveQA (flr, U=40, bgroup then R4), context = published | same | 71 | 71 |
| InfBench-Sum (flr, U=40, bgroup then R4), context = published | same | 100 | 100 |
| LoCoMo U=10 (flr, seg3 then R4), context = published (the reads behind 93.46 / 86.04) | same | 1,540 | 1,540 |
| LoCoMo-Refined U=10, context = published (the reads behind 77.57) | same | 1,382 | 1,382 |
| LoCoMo U=20, context = published | same | 1,540 | 1,540 |
| LoCoMo U=5 (floors 3/1/1 below ten units), context = published | same | 1,540 | 1,540 |
| FactConsolidation SH and MH (attprio_sess, numbered log, newest per slot), every field and the context; reranker scored live on CPU (no published cache) | same | 200 | 200 |
| LoCoMo adapter: store texts rebuilt from the benchmark json | `tests/studio/adapters_vs_parquet.py` | 10 | 10 |

Where a list differed in the MemoryAgentBench cells above, two adjacent units had swapped places: 13 of 551
questions have a different reranker top-100 (banking77 2, CLINC150 2, EventQA 2, DetectiveQA 1, InfBench-Sum 6) and
2 a different pool order (EventQA 1, InfBench-Sum 1, possibly one of its six). Twelve were checked one by one: each
pair's cached reranker scores differ by 1.3e-9 to 3.0e-6, and none is a pair of identical texts. The other three (the
DetectiveQA one and two InfBench-Sum ones) were not printed by the run and are not checked. The published dumps were
scored on another device than the one that wrote the cached scores, so near-equal scores can come out in the other
order. The selected units and the contexts were equal on all of them.
