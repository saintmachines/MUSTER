# Release status

What was checked for this release, and the evidence for each line.

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
| Library (`pip install muster-memory`, imports as `muster`): derive, build, search, render and `Store`; CLI, MCP and HTTP servers | done | `docs/library/`; `docs/library/exact.md` checks it against the paper's data (the cut of 296 of 296 rows, the board's render chain 5,439 of 5,439 reads); test suite 511 passed (CPU, fake models, CI matrix in `.github/workflows/ci.yml`) |
| LoCoMo and LoCoMo-Refined data builders (`harness/make_locomo_pq.py`, `make_locomo_refined_pq.py`, `make_locomo_refined_conv_pq.py`) | done 2026-10-27 | rebuilt in a fresh directory from the public inputs (LoCoMo's `locomo10.json`; LoCoMo-Refined at commit 887091190789; all three sha256-identical with the experiment's copies): 3 of 3 parquets sha256-identical with the experiment's; the two the legs read are in `gate/data.sha256` |
| Model revisions | pinned | `muster/plan.py` pins `BAAI/bge-m3` at `5617a9f6…` and `BAAI/bge-reranker-v2-m3` at `953dc6f6…`; the harness loads both by name, and on the experiment machine the names resolve to these revisions (`harness/README.md`) |
| Offline re-score of the main BEAM table from its judge files | done (gate 0b) | `gate/rescore_beam.py` over 51 judge files |
| Offline re-score of the MemoryAgentBench board from its 48 per-cell result tags | done 2026-10-18 (gate 0b, MemoryAgentBench) | `gate/rescore_mab.md`: 57.52 / 60.92 / 64.71, all 45 values the paper prints for the board (14 cells and Overall at three readers) equal; DetectiveQA the mean of three reads; the result files are on the experiment machine |
| CPU replay at BEAM-1M and 10M | not run (hours to days on CPU) | the render chain at both scales is byte-identical (gate 0c) |
| Build times | measured on one RTX PRO 6000, mean of three runs | `gate/build_time_rtx.md` |
| Paper PDF | the arXiv version, 33 pages | `paper/MUSTER.pdf` |

