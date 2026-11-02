# Gate, stage 0a — the detection rule over the fourteen MemoryAgentBench stores

Run 2026-10-15 07:09 PDT on the experiment machine, CPU, no model, from the cleaned harness
(`harness/derive_config.py`, `harness/cells_registry.py`; `PYTHONHASHSEED=0`, `MUSTER_HOME=<gate>/harness`,
benchmark parquets read in place). Output compared with the paper's Table 9 (`tab:stores`).

| Cell | Rule branch | Unit | Parent | Units (derive / paper) | Parents (derive / paper) | Match |
|---|---|---|---|---|---|---|
| DetectiveQA (det) | narrative (sentence) | 4096 | 8192 | 25 / 25 | 12 / 12 | yes |
| InfBench-Sum (summ) | narrative (sentence) | 4096 | 16384 | 58 / 58 | 14 / 14 | yes |
| EventQA (evq) | narrative (sentence) | 4096 | 32768 | 131 / 131 | 16 / 16 | yes |
| TREC-fine (trcF) | record (blank) | 512 | 4096 | 202 / 202 | 25 / 25 | yes |
| CLINC150 (clin) | record (blank) | 512 | 4096 | 206 / 206 | 25 / 25 | yes |
| TREC-coarse (trcC) | record (blank) | 512 | 4096 | 207 / 207 | 25 / 25 | yes |
| NLU (nlu) | record (blank) | 512 | 4096 | 208 / 208 | 26 / 26 | yes |
| banking77 (b77) | record (blank) | 512 | 4096 | 210 / 210 | 26 / 26 | yes |
| SH-QA (shqa) | record (blank) | 512 | 4096 | 398 / 398 | 49 / 49 | yes |
| LongMemEval (lme) | outside the rule (rounds / sessions) | 512 | 4096 | 719 / 719 | 89 / 89 | yes |
| MH-QA (mhqa) | record (blank) | 512 | 4096 | 854 / 854 | 106 / 106 | yes |
| Recsys (rec) | record (blank) | 512 | 4096 | 2,933 / 2,933 | 366 / 366 | yes |
| FC-SH (fcsh) | numbered log (AUTOSTRUCT, serial 0.9999) | record | slot | – / 18,333 | – / 12,800 | branch |
| FC-MH (fcmh) | numbered log (AUTOSTRUCT, serial 0.9999) | record | slot | – / 18,333 | – / 12,800 | branch |

Statistics printed by the rule (record share / serial fraction): shqa 0.9751 / 0.0016, mhqa 0.9891 / 0.0000, lme 0.0000 / 0.0000,
evq 0.0000 / 0.0000, b77 1.0000 / 0.0000, clin 1.0000 / 0.0000, nlu 1.0000 / 0.0000, trcC 1.0000 / 0.0000, trcF 1.0000 / 0.0000,
rec 1.0000 / 0.0000, summ 0.5504 / 0.0000, det 0.0000 / 0.0000, fcmh 0.0000 / 0.9999, fcsh 0.0000 / 0.9999. The paper's
separation statement (record stores at 0.9751 or above, the rest at 0.5504 or below) is reproduced exactly.

Twelve cells match the paper's unit and parent counts exactly. The two FactConsolidation cells take the numbered-log branch,
whose unit and slot counts (18,333 / 12,800) are produced by the harness run itself, not by `derive_config.py`; the branch
decision is what the rule outputs, and it matches. The script's own out-of-sample check reports MATCH on `summ`, `fcmh` and
`fcsh` against the published legs and marks `lme` as vacuous, as the paper's Section 3.2 explains (LongMemEval keeps the
benchmark's round/session segmentation).

The released `scripts/derive.py` also applies Algorithm 1's step 8, which the table above does not cover: a prose store
that keeps the 4,096-token unit is searched at 1,024 tokens with four times as many units and 8,192-token parents, the
setting of the recorded prose legs (`harness/len_legs.json`: EventQA, InfBench-Sum and DetectiveQA at 1,024 / 8,192,
U = 40). The table checks the base rule against the paper's Table 9.

Standalone script (`scripts/derive.py`, one store at a time, re-run 2026-10-17 06:58 PDT on the same machine, CPU, no
model; the README and landing-page example is that run, verbatim):
BEAM-100K conversation 0 -> record store, 512-token units, 4,096-token parents (28); FactConsolidation row 7 -> numbered
log; DetectiveQA row 100 -> prose, 1,024-token units, 8,192-token parents (12). On the first row of each
MemoryAgentBench cell it gives the recorded unit and parent sizes on 10 of the 12 cells that record them; the two
FactConsolidation cells, which record none, come out as numbered logs. The other two are the paper's stated exceptions:
LongMemEval is read in the benchmark's own rounds and sessions, and InfBench-Sum row 0 alone is a record store while the
paper decides that cell over three rows (record share 0.5504, prose).
