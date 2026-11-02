# Gate, stage 0b (MemoryAgentBench) — the board re-scored offline from its 48 result tags

Run 2026-10-18 on the experiment machine, reading files only, with the released `harness/cellscore.py` and
`harness/mab_scorers.py`: `MUSTER_HOME=<harness directory with results/> python gate/rescore_mab.py`. The 48 tags are the
board's paid reads: fourteen cells at three readers, DetectiveQA read three times per reader (the paper reports the mean of
the three). The metric is the board's own (`harness/noxc_board.py`); the 29 per-question means that `noxc_board.py` checks
against `harness/cellscore.py`, the one owner of every cell's metric, are equal to it.

## The final configuration, cell by cell

| Cells | Retrieval leg (`harness/len_legs.json`) | Context the reader saw | Reads (r = 4om, 41m, 5m) |
|---|---|---|---|
| SH-QA, MH-QA, LongMemEval, Rec | `lz_<cell>`: U=10, `flrs_4_3_3` | the leg's bundle (`make_mab_bundle.py`) | `gOA_l2k_<r>_<cell>` |
| Banking77, CLINC150, NLU, TREC-C, TREC-F | `lz_<cell>`: U=10, `flrs_4_3_3` | the leg's bundle | `nx_ra_<cell>` (GPT-4o-mini), `gOA_nx_<r>_<cell>` |
| EventQA, InfBench-Sum, DetectiveQA | `lz_<cell>`: U=40, `flr_4_3_3`, 1,024-token sentence units, 8,192-token parents | the bundle re-rendered with `render_rule.py R4`: one block per parent, stored order | `gOA_r4_<r>_<cell>`; DetectiveQA also `gOA_r4rep_<r>_det`, `gOA_r4rep2_<r>_det` |
| FC-SH, FC-MH | `mg_<cell>_struct`: AUTOSTRUCT, `attprio_sess` | the bundle with `"[CURRENT] "` removed | `gOA_<4omini, 41mini, 5mini>_<cell>_struct_notag` |

The twelve `lz_` legs run the encoder and the reranker at 2,048 tokens without read-time extraction (`EMB_MAXLEN=2048
CEML=2048 NO_EXTRACT=1`). Every one of these contexts is rebuilt from the published dumps by the released code, and found in
each read's record on every question, in `gate/stage0d_render.md`.

## Result

| Cell | GPT-4o-mini | GPT-4.1-mini | GPT-5-mini |
|---|---|---|---|
| SH-QA | 92.00 | 90.00 | 93.00 |
| MH-QA | 65.00 | 76.00 | 78.00 |
| LongMemEval | 56.33 | 61.67 | 71.00 |
| EventQA | 82.00 | 82.00 | 82.00 |
| Banking77 | 96.00 | 97.00 | 95.00 |
| CLINC150 | 90.00 | 89.00 | 94.00 |
| NLU | 88.00 | 85.00 | 89.00 |
| TREC-C | 70.00 | 73.00 | 84.00 |
| TREC-F | 65.00 | 49.00 | 59.00 |
| Rec | 13.25 | 15.67 | 11.67 |
| InfBench-Sum | 22.24 | 32.33 | 44.47 |
| DetectiveQA (mean of three reads) | 66.20 | 70.89 | 79.34 |
| FC-SH | 92.00 | 93.00 | 94.00 |
| FC-MH | 37.00 | 42.00 | 42.00 |
| Accurate retrieval | 73.83 | 77.42 | 81.00 |
| Test-time learning | 47.52 | 47.13 | 47.93 |
| Long-range understanding | 44.22 | 51.61 | 61.91 |
| Selective forgetting | 64.50 | 67.50 | 68.00 |
| **Overall** | **57.52** | **60.92** | **64.71** |

All 45 values the paper prints for this board (fourteen cells and Overall at three readers) are equal. DetectiveQA's three
reads are 67.61, 66.20 and 64.79 with GPT-4o-mini, 70.42, 71.83 and 70.42 with GPT-4.1-mini, and 78.87, 83.10 and 76.06 with
GPT-5-mini. `harness/l2k_r4_board.py` assembles the same legs with the first read only and prints 57.70 / 60.86 / 64.65.
