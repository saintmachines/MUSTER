# Gate, stage 0d — the render chains of the MemoryAgentBench board and of LoCoMo at ten units

Run 2026-10-18 21:56–21:57 PDT on the experiment machine, CPU, pure Python, no model, about 30 seconds: `gate/run_mab_ctx.sh`
with the released `harness/` (all 56 files equal to `MANIFEST.sha256`), writing only under the gate directory, then
`gate/compare_mab_ctx.py`. Inputs are the published context dumps (`DUMP_CTX=1`; the MemoryAgentBench legs ran with `DRY=1`, the LoCoMo legs
with `ROUTER_ONLY=1`, and neither called a reader): the twelve MemoryAgentBench retrieval
legs `lz_<cell>` (env rows in `harness/len_legs.json`), the two FactConsolidation structural legs `mg_fc{sh,mh}_struct`, and
the LoCoMo legs `lcm_u10` and `lcmrefc_u10`.

The chain per cell is the one the paper's reads consumed:

| Cells | Bundle | Then |
|---|---|---|
| SH-QA, MH-QA, LongMemEval, Rec, Banking77, CLINC150, NLU, TREC-C, TREC-F | `make_mab_bundle.py` (U=10, `flrs_4_3_3`) | read as built |
| EventQA, DetectiveQA, InfBench-Sum | `make_mab_bundle.py` (U=40, `flr_4_3_3`) | `render_rule.py R4`: one block per parent, stored order (`ATOM=sentence CHUNK=1024 MSCALE=8192`, the leg's own) |
| FC-SH, FC-MH | `make_mab_bundle.py` (AUTOSTRUCT, `attprio_sess`) | `"[CURRENT] "` removed |
| LoCoMo, LoCoMo-Refined | `make_bundle.py` (U=10) | `render_rule.py R4` (`ATOM=blank CHUNK=512 MSCALE=2048`) |

Two checks: every rebuilt file against the file the reads consumed (whole-file sha256), and every rebuilt context against the
context hash that each paid read of the board recorded for each question. The board has 48 reads: fourteen cells at three
readers (GPT-4o-mini, GPT-4.1-mini, GPT-5-mini), DetectiveQA read three times per reader (`gate/rescore_mab.md`).

| Cell | Questions | Rebuilt files sha256-equal to the experiment's files | Reads with every question's context hash equal |
|---|---|---|---|
| SH-QA | 100 | yes | 3 of 3 |
| MH-QA | 100 | yes | 3 of 3 |
| LongMemEval | 300 | yes | 3 of 3 |
| EventQA | 100 | yes: the R4 file the reads consumed, and the bundle before R4 | 3 of 3 |
| Banking77 | 100 | per question only | 3 of 3 |
| CLINC150 | 100 | per question only | 3 of 3 |
| NLU | 100 | per question only | 3 of 3 |
| TREC-C | 100 | per question only | 3 of 3 |
| TREC-F | 100 | per question only | 3 of 3 |
| Rec | 200 | yes | 3 of 3 |
| InfBench-Sum | 100 | yes: the R4 file the reads consumed, and the bundle before R4 | 3 of 3 |
| DetectiveQA | 71 | yes: the R4 file the reads consumed, and the bundle before R4 | 9 of 9 |
| FC-SH | 100 | yes: the NOTAG file the reads consumed, and the tagged bundle | 3 of 3 |
| FC-MH | 100 | yes: the NOTAG file the reads consumed, and the tagged bundle | 3 of 3 |
| **Total** | **1,671** | **14 of 14 files: 9 the reads consumed, 5 intermediates** | **48 of 48 reads; 5,439 of 5,439 question reads** |

The five in-context-learning cells were read from bundles packed under another leg label, so their whole files differ in the
label field; every context equals the reads' recorded hash. Every rebuilt context also hashes to its own `ctxhash`.

LoCoMo at U=10 (1,540 questions; the reads behind 93.46 and 86.04) and LoCoMo-Refined at U=10 (1,382 questions; the reads
behind 77.57): the bundle and its R4 render are byte-identical to the files the reads consumed, 4 of 4 files.

What this proves: from the published dumps, the released bundle writers, R4 renderer and FactConsolidation transform produce
the exact contexts behind every MemoryAgentBench board cell and the LoCoMo and LoCoMo-Refined rows. What it does not prove: the
retrieval that wrote those dumps. Stage 1 replays retrieval for BEAM-100K and for an earlier LongMemEval leg with read-time
extraction (`gate/stage1_beam100k.md`). A CPU replay of the board's LongMemEval leg `lz_lme` was started and stopped: the
reranker cache holds its scores (`CEML=2048`) for 3 of the first conversation's 60 questions, so a CPU replay means live
reranking at 2,048 tokens for hours. The cache snapshot was checked unchanged afterwards (1,083 of 1,083 files equal to
`ce_cache.SHA256`).
