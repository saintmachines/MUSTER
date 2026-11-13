# Release status

What was checked for this release, and the evidence for each line.

The paper directory preserves the author's submitted arXiv package, local revision v219I, and provides the November 11 draft describing SDK 0.1.3 source-collection Auto. [paper/README.md](paper/README.md)
identifies both. The [project page](https://saintmachines.github.io/MUSTER/)
and [repository](https://github.com/saintmachines/MUSTER) are its release addresses.
The Python distribution is `muster-memory`. Install it with `pip install muster-memory` for structure inspection.
Add `[models]` for local retrieval, `[models,mcp]` for MCP, or the framework extras documented in the library guide.

The hosted CI matrix runs Ubuntu x64 and macOS 15 Intel at Python 3.10, 3.11 and 3.12, with the same tests and failure policy.
Earlier `macos-latest` arm64 jobs were cancelled before execution because hosted runners were unavailable, including the retry.
Studio arm64 validation is a separate result; it must not be reported as a hosted arm64 CI pass. See the actual Actions run
for the current commit's execution status.

The [browser structure inspector](docs/try/README.md) is included with its source, bundled tokenizer and compatibility
report: 1,372 inputs matched SDK 0.1.3 single-store native plans, statistics and token counts, with three additional explicit-refusal checks,
including the post-paper serial-format extension; the report pins the tested SDK source hashes. It previews configuration;
it does not generate answers or replace the paper's retrieval evaluation.

## 0.1.3 collection Auto

The new native `build_collection(stores)` entry point builds independent stores under a source-derived configuration.
The default statistics use the first up to three sources; `sample=None` uses all sources. Every member is checked,
and a list of strings passed to ordinary `build` remains one store of records. Source dates, speakers and session-round
structure determine the applicable record path; configuration takes no benchmark name, question or answer.

The final SDK API replay, with all default arguments, covers all 1,471 non-FC question positions from the SDK paper-settings evaluation. Complete messages match at 1,450 positions; 21 LongMemEval positions change because native rounds now retain previously unpaired source turns. The 200 FC contexts retain frozen selection-trace rendering and numbered-path checks. [Input replay receipt](data/auto_0_1_3/API_INPUT_REPLAY.json).

The current MAB scores combine archived responses for identical inputs with new responses and judgments for the 21 changed LongMemEval inputs. [Per-question score lineage](data/auto_0_1_3/LME_REFRESH.json) and [all fourteen task cells and conditions](docs/data/auto_0_1_3.json) bind the five readers to the final SDK. Original and measured 0.1.2 results remain below.

The [updated manuscript](paper/MUSTER_auto_20261111.pdf) retains the original and historical numeric results. Its current Auto rows update LongMemEval and the resulting AR/Overall aggregates after evaluating the changed inputs. Its source, source ZIP and diffs are listed in the [paper archive](paper/README.md). The [build lock](RELEASE_BUILD.json) records the fixed build tools and exact distribution hashes. The current commit's [CI matrix](https://github.com/saintmachines/MUSTER/actions/workflows/ci.yml) and the [release](https://github.com/saintmachines/MUSTER/releases/tag/v0.1.3) retain their execution records; earlier CI and timing receipts remain specific to their original sources.

Other benchmarks do not gain new reader scores from this release. In particular, the historical LoCoMo and BEAM rows retain their original representations and context budgets. Source-structure checks and browser parity are separate from answer quality.

## Package 0.1.2 correction

Version 0.1.2 fixes the PyPI description's app connection link and adds rendered-description and destination checks before publication. Its retrieval implementation and browser runtime are unchanged from 0.1.1. At that November 8 correction, the working paper contained the original thirty experimental tables and its source diff only updated two statements about released reproduction materials.

## November 10 SDK 0.1.2 evaluation

The completed [cell data](docs/data/native_mab_0_1_2.json) evaluate the released 0.1.2 wheel with 2048-token encoder and reranker limits. With the paper settings, MAB Overall is 60.5339, 62.1394, 67.0231, 69.6659 and 67.0278 for GPT-4o-mini, GPT-4.1-mini, GPT-5-mini, GLM-5.3-Flash and GPT-6 Luna. These are the measured SDK 0.1.2 paper-settings MAB results. There are 14 cells and 1,671 questions per reader; DetectiveQA's three reads make 1,813 raw response slots. The GPT reader group combines 678 fresh answers with 4,335 reused question-reader score units. The GLM/Luna group has 448 fresh and 3,178 reused raw response slots. Reuse requires complete-message/request equality; the two accounting groups retain their different units.

The separately measured 0.1.2 per-store automatic results are 58.5282, 60.60 and 64.74 Overall for GPT-4o-mini, GPT-4.1-mini and GPT-5-mini. Relative to the original implementation, SF rises and LRU falls for all three; Overall rises for GPT-4o-mini and GPT-5-mini and falls for GPT-4.1-mini. These rows are not a new aggregate over the other benchmarks, and no new Infini-style judgment is implied. SDK evaluations use 2,048-token encoder and reranker inputs; the added numbered-fact rules were developed on the reported FC data. A same-numerical-cache replay through actual 0.1.0 and 0.1.2 gives identical non-FC contexts and reconstructed full reader messages on all 1,471 questions under the fixed paper configuration.

The [36-page working draft](paper/MUSTER_native_20261109.pdf), two source diffs and cell CSVs record these conditions. Table 2 also compares parent-priority selection, 58.5092, with the final policy, 60.5339, on the same SDK axis. Independent source checks preserve the original numeric rows, all existing table blocks, figures and algorithm. Clean and marked PDFs have identical page-level word sequences, no undefined references or overflowing boxes, and the eleven-file upload ZIP rebuilds to the same clean PDF. The other benchmarks and six-setting overview retain their original implementation scope. Individual cells include losses. Current commit CI must still be read from its own Actions run; arXiv publication is a separate account/server state.

## SDK 0.1.1 follow-up

The [follow-up data](docs/data/sdk_0_1_1.json) separately record the six FactConsolidation rows, requested reader settings, observed models/providers, scoring view and retry limits. The tested SDK commit identifies the source used for input replay, not a later publication commit. MINT state preserves all 563 contexts and complete reader messages from 0.1.0; this is input identity, not a new quality measurement. The project page keeps the original paper tables separate from these native-SDK results.

## Checks recorded for the original paper release

The historical checks below retain their original source and date. Current commit CI status must be read from its own Actions run.

| Item | State | Evidence |
|---|---|---|
| Harness files copied from the experiment machine (current revision; 9 env switches added after the paper's runs, all off by default); only comments, strings, paths (with the imports they need) and device arguments edited for release | done | `MANIFEST.sha256` pins all 59 files (the import closure of the entry points, two prompts, the leg tables, the BEAM and LoCoMo data builders) |
| Key scan of every shipped file | clean | provider, cloud, repository, Google and Hugging Face key shapes, private-key headers, absolute home paths and e-mail addresses: 0 |
| Standalone detection script (`scripts/derive.py`) | done, smoke-tested on three stores | output pasted in README |
| Harness defaults match the paper's runs (9 env switches added later, all off by default) | verified by diff and by the gate below | the gate reproduces the paper's contexts with the released files |
| Gate stage 0a: detection rule over the 14 stores vs paper Table 9 | passed | `gate/stage0_derive.md` (12 cells exact, 2 branch) |
| Gate stage 0b: offline re-score of the paper's main BEAM table from its judge files | passed | `gate/rescore_beam.py`, `gate/rescore_beam.md` (paper Table 5, 48 of 48 scores equal from 51 judge files) |
| Gate stage 0c: render chain regenerates the paper's rendered contexts byte for byte, BEAM 100K/1M/10M at U=20/40 and LoCoMo at U=20 | passed | `gate/stage0_tier0.md` (14 of 14 files sha256-identical) |
| Gate stage 0d: render chain of the final MemoryAgentBench board (all 14 cells; R4 on EventQA, DetectiveQA and InfBench-Sum; `[CURRENT]` removed on FC-SH and FC-MH) and of LoCoMo and LoCoMo-Refined at U=10 | passed 2026-10-18 | `gate/stage0d_render.md` (48 of 48 board reads with every question's context hash equal, 5,439 of 5,439; 14 of 14 rebuilt files sha256-identical with the experiment's files, 9 of them the files the reads consumed and 5 intermediates; the five in-context-learning reads match per question; LoCoMo 4 of 4 files) |
| Gate stage 1: CPU-only retrieval replay (embedding, BM25, rerank from the warm cache, floors, render) vs the published contexts | BEAM-100K U=20: R4 contexts 400/400 byte-identical (seg3 intermediate 399/400, same units, one block boundary); U=40: 399/400, the one difference an exact dense-score tie between two near-duplicate slash-only units; MemoryAgentBench LongMemEval, an earlier leg with read-time extraction: 300/300; both re-confirmed after the release renames. The board's own LongMemEval leg `lz_lme` is not replayed: the reranker cache holds its `CEML=2048` scores for 3 of the first conversation's 60 questions, so a CPU replay means hours of live reranking (`gate/stage0d_render.md`) | `gate/stage1_beam100k.md` |
| Library (distribution name `muster-memory`, imports as `muster`): derive, build, search, render and `Store`; CLI, MCP and HTTP servers | done | `docs/library/`; `docs/library/exact.md` checks it against the paper's data (the cut of 296 of 296 rows, the board's render chain 5,439 of 5,439 reads); Studio test suite 608 passed, 0 skipped with the submitted paper source (CPU, fake models); fresh installed-wheel record and transport checks 41 passed, 0 skipped. Hosted results are recorded by the CI matrix in `.github/workflows/ci.yml` |
| LoCoMo and LoCoMo-Refined data builders (`harness/make_locomo_pq.py`, `make_locomo_refined_pq.py`, `make_locomo_refined_conv_pq.py`) | done 2026-10-27 | rebuilt in a fresh directory from the public inputs (LoCoMo's `locomo10.json`; LoCoMo-Refined at commit 887091190789; all three sha256-identical with the experiment's copies): 3 of 3 parquets sha256-identical with the experiment's; the two the legs read are in `gate/data.sha256` |
| Model revisions | pinned | `muster/plan.py` pins `BAAI/bge-m3` at `5617a9f6…` and `BAAI/bge-reranker-v2-m3` at `953dc6f6…`; the harness loads both by name, and on the experiment machine the names resolve to these revisions (`harness/README.md`) |
| Offline re-score of the main BEAM table from its judge files | done (gate 0b) | `gate/rescore_beam.py` over 51 judge files |
| Offline re-score of the MemoryAgentBench board from its 48 per-cell result tags | done 2026-10-18 (gate 0b, MemoryAgentBench) | `gate/rescore_mab.md`: 57.52 / 60.92 / 64.71, all 45 values the paper prints for the board (14 cells and Overall at three readers) equal; DetectiveQA the mean of three reads; the result files are on the experiment machine |
| CPU replay at BEAM-1M and 10M | not run (hours to days on CPU) | the render chain at both scales is byte-identical (gate 0c) |
| Build times | measured on one RTX PRO 6000, mean of three runs | `gate/build_time_rtx.md`, driver in `harness/buildtime_rtx/` |
| Paper PDF | prepared for the first arXiv submission, 33 pages | `paper/MUSTER.pdf`, identified in `paper/README.md` |
| Evaluation artifact bundle | release in preparation, separate from this source tree | per-question BEAM answers and judgments and Honcho context logs; offline re-scoring requires these files |

