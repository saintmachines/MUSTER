# Verify the library against the paper yourself

Two checks run anywhere, with public data. Each compares the library's output with a table of hashes in `gate/`
(no benchmark text is in the repository).

## 1. Get the data at the pinned versions

```bash
mkdir -p data_root/data data_root/data_all/data data_root/beam
# MemoryAgentBench: Hugging Face ai-hyz/MemoryAgentBench at revision 7ea066982b140a19337e17e60d45d4076e042faf
huggingface-cli download ai-hyz/MemoryAgentBench --repo-type dataset --revision 7ea066982b140a19337e17e60d45d4076e042faf \
    --local-dir mab_hf
cp mab_hf/data/Accurate_Retrieval-00000-of-00001.parquet data_root/data/
cp mab_hf/data/{Test_Time_Learning,Long_Range_Understanding,Conflict_Resolution}-00000-of-00001.parquet data_root/data_all/data/
# BEAM: scripts/fetch_beam_repo.sh, then harness/beam/build_parquet.py (100K), build_parquet_1m.py, build_parquet_10m.py
# LoCoMo: snap-research/locomo data/locomo10.json; the store texts are muster.adapters.locomo(...) of each conversation
```

`gate/data.sha256` holds the hash of each file the paper's runs read; the four MemoryAgentBench files download
byte-identical. The checks below do not depend on parquet bytes (which change with the parquet writer's version):
each row's store text is checked against its hash in `gate/cut_table.json` first, and a row whose text differs is
reported and skipped, never compared.

## 2. The cut: no model, CPU only

```bash
pip install ".[models,parquet]"          # in the repository folder
muster verify cuts --data data_root
```

For every row of the paper's 24 final legs it compares the number of units and the hashes of the unit list and of the
parent labels with `gate/cut_table.json`. The unit-list hash is the one the paper's harness used to name its reranker
cache files, so an equal hash means the library cut the store into exactly the units the published runs scored.

## 3. The contexts: the paper's searches, rerun

```bash
PYTHONHASHSEED=0 muster verify contexts --data data_root --legs lz_shqa,lcm_u10 [--cache DIR]
```

This builds each store under its preset in exact mode, runs every question, and compares the context hash with
`gate/ctx_table.jsonl` (10,273 questions over the 24 legs). Exact mode needs `PYTHONHASHSEED=0`; the published equality
was measured on CPU with the versions in `muster/compat/env.py`.

With the published reranker scores (`--cache`, the cache files the paper's runs wrote), the comparison is exact on
CPU. Without them the reranker runs live: on another device, two candidates whose scores differ in the sixth decimal
can change places, and a context that holds one of them can differ. [exact.md](exact.md) lists what was measured on
the experiment machine and every known difference with its cause.
