# Build time on one RTX PRO 6000

This folder is the driver behind [the build times the paper reports](../../gate/build_time_rtx.md). It times MUSTER's
write-time build of BEAM conversation 0 at 100K, 1M and 10M on one NVIDIA RTX PRO 6000 Blackwell Server Edition, with the
settings of the reported BEAM legs (`../len_legs.tsv` lines 19 to 24). It calls no generative model and needs no key.

| Stage | What runs |
|---|---|
| Statistics | `derive()`: the two rule statistics and the store's tokens, then the rule |
| Units | `o200k_base` units of 512 tokens, parents every 4,096 tokens, and the harness's two checks of the cut |
| Embed | one bge-m3 pass at revision `5617a9f`, max_seq 2048, batch 32, fp16 |
| BM25 | the index over the unit texts, lowercased tokens without punctuation |
| Graph | the entity index and the slot maps |

The build time is the five stages together, the row `Build with graph s` in `summary.md`. Model load and warm-up are
timed apart and excluded.

The stages run the harness's build path, cut for timing, with the encoder set as above. `ship_lme.py` and `ship_derive.py`
hold the lines of `../lme_pf29.py` and `../derive_config.py` verbatim, and `verify_copies.py` compares them line by line.
Before a store's time is kept, its units, parents, statistics, graph counts and the sha256 of its unit texts, embedded texts
and parent ids must equal `expected.json`, the values the harness computes for that store on the CPU. A difference stops
the run.

## Run

The card is available on Google Colab's G4 runtime. From `harness/`:

```sh
pip install -r buildtime_rtx/requirements.txt
python -c "import shutil; from huggingface_hub import hf_hub_download as d; shutil.copy(d('Mohammadta/BEAM', 'data/100K-00000-of-00001.parquet', repo_type='dataset'), 'beam/100K.parquet')"
python beam/build_parquet.py; python beam/build_parquet_1m.py; python beam/build_parquet_10m.py   # the 1M and 10M builders fetch their own files
python -B buildtime_rtx/make_stores.py                                                            # stores and bge-m3
python -B buildtime_rtx/verify_copies.py --src .
BM25LC=1 PYTHONHASHSEED=0 python -B buildtime_rtx/units_check.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BM25LC=1 PYTHONHASHSEED=0 \
  python -B buildtime_rtx/time_beam_build_rtx.py --out ../results/buildtime_rtx
```

`units_check.py` needs no GPU and ends with `3 of 3 stores equal expected.json`. The timed run takes three repeats, then one
fp32 embed pass per store saved in its own file and never added to a build time (`--no-fp32` skips it). It writes
`beam_buildtime_rtx_repeatN.json` and `summary.md` after each repeat, and `--resume` skips what is already saved with the same
setup.

The timed run refuses without CUDA, stops on any other GPU unless `--force` is given, and refuses an environment with a
variable named like a key (`API_KEY`, `_TOKEN`, `SECRET`).

To try the path on a CPU, never as a measurement:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BM25LC=1 PYTHONHASHSEED=0 \
  python -B buildtime_rtx/time_beam_build_rtx.py --device cpu --emb-limit 16 --tag smoke --out ../results/buildtime_smoke
```
