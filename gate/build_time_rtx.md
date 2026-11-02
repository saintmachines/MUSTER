# Build time on one RTX PRO 6000

The build times of the paper's appendix table `tab:beamcost`: MUSTER's write-time build of BEAM conversation 0 at each
scale, three repeats, on one NVIDIA RTX PRO 6000 Blackwell Server Edition (95.0 GB, driver 580.82.07, CUDA 13.0,
torch 2.11.0+cu130, sentence-transformers 5.3.0, transformers 5.5.0, tokenizers 0.22.2, rank-bm25 0.2.2), run on
2026-10-31.

| Store | Source tokens | Units | Parents | Statistics s | Units s | Embed s | BM25 s | Graph s | Build s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BEAM 100K | 132K | 247 | 29 | 0.068 | 0.098 | 0.652 | 0.031 | 0.021 | 0.870 |
| BEAM 1M | 1.05M | 2,031 | 241 | 0.529 | 0.799 | 5.058 | 0.248 | 0.155 | 6.790 |
| BEAM 10M | 11.0M | 22,746 | 2,697 | 5.993 | 9.006 | 56.791 | 2.955 | 2.056 | 76.800 |

Each time is the mean of the three repeats. The build is the five stages together. Model load (1.75 s) and warm-up
(0.39 s) are excluded.

Settings are those of the reported BEAM legs (`harness/len_legs.tsv`): blank-line atoms packed into units of 512
`o200k_base` tokens, parents every 4,096 tokens, one bge-m3 pass at revision `5617a9f` with max_seq 2048, batch 32 and
normalized vectors, then BM25 over lowercased tokens without punctuation. The embedding pass runs in fp16 for these
timings and in fp32 for the reported scores.

Before a store's times were kept, its units, parents, statistics output, graph counts and the sha256 of its unit texts,
embedded texts and parent ids had to equal the values the harness computes for that store on the CPU.

The driver and the steps to run it are in [`harness/buildtime_rtx/`](../harness/buildtime_rtx/README.md).

No generative model is called.
