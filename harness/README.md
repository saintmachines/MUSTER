# harness/ — the experiment harness behind the paper's numbers

These files preserve the paper's experiment harness, alongside historical utilities identified in the table below
(source hashes in `../MANIFEST.sha256`). They are **research code**: env-switch driven, one process per
benchmark leg, with a spend ledger and a cost cap on every paid call. The cleaned library is the separate
`muster/` package (`../docs/library/`); this directory documents the experiment commands and their provenance.

| File | Role |
|---|---|
| `derive_config.py`, `cells_registry.py` | the pre-registered detection rule applied to the fourteen MemoryAgentBench stores; the cell registry (parquet, rows, protocol chunk) |
| `lme_pf29.py` | the harness: load a store, cut units, embed once (bge-m3), BM25, reach, rerank (bge-reranker-v2-m3), resolve, render; `ROUTER_ONLY=1 DUMP_CTX=1 DRY=1` dumps the rendered context per question with no reader call |
| `core2.py`, `unified_router.py`, `rerankers.py` | retrieval channels, the channel-floor router, the reranker wrappers |
| `src/real_e1_lme.py`, `src/real_e1_conflict_killer.py`, `src/real_e1_conflict.py` | encoder choice (`MIRROR_BGE`), entity index and multi-hop reach, supersession helpers |
| `render_parent.py`, `render_rule.py` | context assembly rules (R2/R3/R4; the paper ships R4: one block per parent, stored order) |
| `make_bundle.py` | packs the byte-identical rendered contexts of a leg into a reader-swap bundle with a sha256 pin |
| `or_client.py`, `or_beam_reader.py`, `or_redo_rows.py` | readers on OpenRouter: one pinned provider per model, `allow_fallbacks=false`, answering provider checked on every response |
| `judge_beam.py`, `beam_score.py`, `beam_competitor_judge.py` | BEAM's official judge (imported unmodified from `mohammadtavakoli78/BEAM` at b2da22e) and its ten-category macro |
| `judge_lme.py`, `judge_infbench.py`, `cellscore.py` | MemoryAgentBench judges (LongMemEval GPT-4o, InfBench-Sum) and the one owner of how a cell is scored |
| `noxc_board.py`, `l2k_r4_board.py` | the MemoryAgentBench board at the three readers, assembled from the per-cell result tags and set against the earlier legs with read-time extraction; `l2k_r4_board.py` (`L2K=1`) reads the final board's legs with DetectiveQA's first read only (57.70 / 60.86 / 64.65). The paper's board, DetectiveQA the mean of three reads (57.52 / 60.92 / 64.71), and the context behind every cell: `../gate/rescore_mab.py`, `../gate/rescore_mab.md`. `board3.py` is the earlier board script kept for the v78-era rows |
| `make_mab_bundle.py` | packs a MemoryAgentBench leg's rendered contexts into a bundle with the `ctxhash` of the paid leg it replays |
| `len_legs.json`, `len_legs.tsv` | the exact environment string of every published retrieval leg (31 rows) and the reader system message; the gate reads the leg it replays from here. The rows are retrieval legs only: for EventQA, DetectiveQA and InfBench-Sum (`lz_evq`, `lz_det`, `lz_summ`) and for the BEAM legs, the paper's contexts are these legs' bundles re-rendered with `render_rule.py R4` (one block per parent, stored order; ATOM, CHUNK and MSCALE of the leg). The recipe for every MemoryAgentBench cell is in `../gate/rescore_mab.md`, and LoCoMo's chain in `../gate/stage0_tier0.md` and `../gate/stage0d_render.md` |
| `locomo_read.py`, `locomo_rule.py`, `locomo_judge.py`, `locomo_score_judge.py`, `locomo_score1.py`, `locomo_score3.py`, `locomo_f1.py`, `locomo_fmt_check.py` | LoCoMo: rule, read legs under the two answer prompts, judge, scoring (single run, mean of three runs, token F1), answer-format check |
| `budget_guard.py`, `ledger_profile.py` | the spend ledger and cap (a resumed leg resumes its spend, not its budget) |
| `time_build.py` | legacy four-stage probe: cl100k units, embedding at max_seq 448, BM25, no parent graph. This measures the historical configuration. The current paper's measured times use different settings, documented in [build_time_rtx.md](../gate/build_time_rtx.md); their exact timing driver is not included here. No generative call. |
| `honcho_beam_cloud.py`, `mem0_*.py`, `mem0store_parquet.py` | the competitor runs on BEAM and the budget-matched cuts of their stores |
| `src/real_e1_reader.py`, `src/mirror_store_swap.py`, `src/metrics.py`, `src/stats.py`, `rerender_nomerge.py`, `mab_scorers.py` | modules the files above import at load time |
| `beam_ser.py`, `beam_merge.py`, `beam/build_parquet.py`, `beam/build_parquet_1m.py`, `beam/build_parquet_10m.py` | data preparation: BEAM conversations to the parquet layout the harness reads (serialization S1), shard merge |
| `make_locomo_pq.py`, `make_locomo_refined_pq.py`, `make_locomo_refined_conv_pq.py` | data preparation: LoCoMo (`locomo/locomo10.json` from `snap-research/locomo` at commit 3eb6f2c5) and LoCoMo-Refined (`locomo_refined/questions.jsonl` and `conversations.jsonl` from `mem-eval-suite/LoCoMo_refined` at commit 887091190789) to the parquet layout the harness reads; `beam/locomo_refined.parquet` is an intermediate of the third |
| `prompts/beam_prompt_C_sys.txt`, `prompts/beam_prompt_B_qtpl.txt` | the answer prompt (C) and the question tail used for every BEAM read |

Paths and devices (2026-10-15): every file resolves its working directory from `MUSTER_HOME` (default: this directory);
`MUSTER_RESULTS` points `make_bundle.py` at a results directory, `BEAM_REPO` at the BEAM clone, `BGE_DEVICE` and
`CE_DEVICE` (default `mps`, as published) pick the encoder and reranker devices, and `torch.mps.empty_cache()` is guarded, so
the harness runs on CPU, CUDA or MPS. The published defaults are unchanged, and `../gate/` holds the proof that the cleaned code regenerates the paper's
contexts byte for byte. API keys are read from the environment only; none is in this tree.

Models: the harness loads `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` by name. On the machine that produced the paper's numbers the two names resolve to revisions
`5617a9f61b028005a4858fdac845db406aefb181` and `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, the revisions the library pins in `muster/plan.py`.
