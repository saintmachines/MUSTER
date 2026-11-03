# Gate, stage 0b (BEAM), the main table re-scored offline from its 51 judge files

Download the [evaluation artifacts](../data/eval_v1/README.md), verify the archive, and point `MUSTER_HOME` at its extracted
`MUSTER_eval_v1` directory. The release check reproduces the same 48 values below from the actual downloadable files.

Run 2026-10-28 on the experiment machine, CPU only, pure Python, with no model and no API call. `gate/rescore_beam.py` scores
each per-question judge file `results/bOR_<tag>_bj_gpt-4.1-mini.jsonl` with `harness/beam_score.py`, which aggregates as
BEAM does (rubric-item mean per question, mean over a conversation's questions per category, mean over conversations, then
the ten-category macro, with 400/20, 700/35 and 200/10 questions and conversations at 100K, 1M and 10M). It compares every
score with Table 5 of the paper (`tab:beamscale`), where GLM-5.3-Flash reads each store under our prompt and BEAM's
rubric judge scores with GPT-4.1-mini.

    MUSTER_HOME=<directory with results/> python gate/rescore_beam.py

| Row | 100K re-scored / printed | 1M | 10M | Mean |
|---|---|---|---|---|
| Mem0 2.1.0, full-ver. (mean of its two searches) | 75.542 / 75.5 | 72.250 / 72.3 | 62.324 / 62.3 | 70.039 / 70.0 |
| Mem0 2.1.0, full-ver., 19.5K | 75.519 / 75.5 | 73.054 / 73.1 | 63.948 / 63.9 | 70.840 / 70.8 |
| Honcho | 67.068 / 67.1 | 60.651 / 60.7 | 48.888 / 48.9 | 58.869 / 58.9 |
| Hindsight 0.10.3 | 66.429 / 66.4 | 67.199 / 67.2 | 52.194 / 52.2 | 61.941 / 61.9 |
| EverMemOS | 71.959 / 72.0 | 69.401 / 69.4 | 60.609 / 60.6 | 67.323 / 67.3 |
| EverMemOS, 19.5K | 76.276 / 76.3 | 73.148 / 73.1 | 62.699 / 62.7 | 70.708 / 70.7 |
| HippoRAG-v2, 9.7K | 72.097 / 72.1 | 68.594 / 68.6 | 58.730 / 58.7 | 66.474 / 66.5 |
| HippoRAG-v2, 19.5K | 75.707 / 75.7 | 71.307 / 71.3 | 59.951 / 60.0 | 68.988 / 69.0 |
| True Memory, 9.7K (mean of two reads) | 75.351 / 75.4 | 72.109 / 72.1 | 61.377 / 61.4 | 69.612 / 69.6 |
| True Memory, 19.5K (mean of two reads) | 78.085 / 78.1 | 74.599 / 74.6 | 62.448 / 62.4 | 71.711 / 71.7 |
| MUSTER, U=20 (mean of two reads) | 74.981 / 75.0 | 72.449 / 72.4 | 62.702 / 62.7 | 70.044 / 70.0 |
| MUSTER, U=40 (mean of two reads) | 76.146 / 76.1 | 73.363 / 73.4 | 64.305 / 64.3 | 71.271 / 71.3 |

All 48 printed values are reproduced. Where a row is the mean of two reads or two searches, each cell averages the scores of
the two files, and no read was dropped or selected. The tags behind each row:

| Row | Tags (each ends in `_100k`, `_1m` or `_10m` unless given in full) |
|---|---|
| Mem0 2.1.0, full-ver. | `M0_top200_glmmax` with `M0R_top200_100k` at 100K, `M0_1m_glmmax` with `M0R_top200_1m`, `M0_10m_glmmax` with `M0R_top200_10m` |
| Mem0 2.1.0, full-ver., 19.5K | `M0B_b19500` |
| Honcho | `HC_glmmax`, `HC1m_glmmax`, `HC10m_glmmax` |
| Hindsight 0.10.3 | `pC_glm_hs` |
| EverMemOS, EverMemOS 19.5K | `pC_glm_emososs`, `pC_glm_emosd19500` |
| HippoRAG-v2, 9.7K and 19.5K | `pC_glm_hr2r40b9700`, `pC_glm_hr2r40b19500` |
| True Memory, 9.7K and 19.5K | `pC_glm_tmb9700` and `pC_glm_tmb9700rep`, `pC_glm_tmb19500` and `pC_glm_tmb19500rep` |
| MUSTER, U=20 and U=40 | `r4_u20` and `r4rep_u20`, `r4_u40` and `r4rep_u40` |

This proves that every score of the paper's main BEAM table can be recomputed from the per-question judge files with the
shipped scorer, without a model. It does not regenerate the answers (a reader call) or the judgments (a judge call), which
would be a paid run and is outside this gate. A copy of the script with one printed value changed (Mem0 at 100K, 75.5 to
75.6) exits 1 and names that cell.
