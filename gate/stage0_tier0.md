# Gate, stage 0c — the render chain regenerates the paper's rendered contexts byte for byte

Run 2026-10-15 07:09–07:12 PDT on the experiment machine, CPU, pure Python, no model: the cleaned `harness/make_bundle.py`
(reading the published `ROUTER_ONLY=1 DUMP_CTX=1 NO_EXTRACT=1` context dumps `results/lz_u{U}_beam{SZ}_s*.jsonl` of the
`lz_u{20,40}_beam{100k,1m,10m}` legs, env rows in `len_legs.tsv`) and `harness/render_rule.py R4` (with
`render_parent.py` and `rerender_nomerge.py`, `ATOM=blank CHUNK=512 MSCALE=4096`, `PYTHONHASHSEED=0`), writing only under
the gate directory. The R1 bundle is what the paper's `bOR_r1_*` reads consumed; the R4 file is what the Table 6 reads
(`bOR_r4_*`, `bOR_r4rep_*`) consumed. Whole-file sha256 of the regenerated files against the oracles:

| Scale | U | Questions | R1 bundle (gate = oracle) | R4 file (gate = oracle) | R4 == R1 per question |
|---|---|---|---|---|---|
| 100K | 20 | 400 | 0d96b3a92bbb = 0d96b3a92bbb | 491ba133a699 = 491ba133a699 | 351 / 400 |
| 100K | 40 | 400 | 1cde43d8cb1d = 1cde43d8cb1d | 4aa925466c0f = 4aa925466c0f | 399 / 400 |
| 1M | 20 | 700 | f551db3b5e01 = f551db3b5e01 | 8af45c481686 = 8af45c481686 | 674 / 700 |
| 1M | 40 | 700 | 333d74618e07 = 333d74618e07 | 5ffa992189ff = 5ffa992189ff | 698 / 700 |
| 10M | 20 | 200 | 0d5cde79f93b = 0d5cde79f93b | 9b8f4c781973 = 9b8f4c781973 | 195 / 200 |
| 10M | 40 | 200 | a36c89b91215 = a36c89b91215 | a36c89b91215 = a36c89b91215 | 200 / 200 |
| LoCoMo | 20 | 1,540 | 372315507d5c = 372315507d5c (R0, `MSCALE=2048`) | 123997b38608 = 123997b38608 | 1,366 / 1,540 |

Fourteen of fourteen files are byte-identical, the twelve BEAM files and the two LoCoMo files (the LoCoMo chain: `results/lcm_u20_s*.jsonl` → `make_bundle.py lcm_u20 beam/locomo.parquet` → R0 bundle → `render_rule.py R4` with `ATOM=blank CHUNK=512 MSCALE=2048`, the configuration the rule derives for LoCoMo, → `bundle_out/locomo/lcm_u20_r4.jsonl`, the file the paper's LoCoMo reads consumed). Full sha256 values are in the gate logs. The last column is the
renderer's own statistic (`same_as_input`): how many questions the one-block-per-parent rule left unchanged relative to the
adaptive render, as expected (at U=40 on 10M every selection already had one block per parent).

What this proves: the unit cutter (o200k, 512-token record units), the parent labelling (4096-token parents), the R4 render
and the bundle writer in the cleaned code are the ones that produced the paper's contexts, byte for byte, at all three
scales. What it does not prove: the retrieval itself (embedding, BM25, reranking, channel floors), which is stage 1
(`stage1_beam100k.md`), run on CPU from the same cleaned code.
