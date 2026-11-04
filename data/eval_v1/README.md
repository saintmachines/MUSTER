# Evaluation artifact download

The per-question BEAM answers and judgments and the Honcho query/context artifacts promised with the code are packaged as a release download.

- [GitHub Release: paper-artifacts-v1](https://github.com/saintmachines/MUSTER/releases/tag/paper-artifacts-v1)
- [MUSTER_eval_v1.tar.gz](https://github.com/saintmachines/MUSTER/releases/download/paper-artifacts-v1/MUSTER_eval_v1.tar.gz)
- [SHA256SUMS](https://github.com/saintmachines/MUSTER/releases/download/paper-artifacts-v1/SHA256SUMS)

The release is staged as a draft during preparation. The links become publicly accessible when the repository and this release are published. A Hugging Face mirror will be linked after it is uploaded and verified.

| Group | Files | Rows | Uncompressed bytes |
|---|---:|---:|---:|
| BEAM answers | 51 | 22,100 | 108,294,974 |
| BEAM judgments | 51 | 22,100 | 38,900,825 |
| Honcho artifacts | 6 | 2,600 | 216,270,040 |

The two Honcho formats contain the same 1,300 query/context pairs; the repeated BEAM runs are separate run tags. Rows are not pooled into one accuracy figure. The [manifest](MANIFEST.tsv) identifies all 108 files by SHA-256, bytes and rows.

Archive: **77,486,652 bytes**, SHA-256:

```
617c19d23c02a8bc958fbc5dba7d6e41fae11b8c672a5a96df5b562200bc77f7
```

## Verify and re-score

Download the archive and `SHA256SUMS` into the same directory. On macOS:

```bash
shasum -a 256 -c SHA256SUMS
tar -xzf MUSTER_eval_v1.tar.gz
python MUSTER_eval_v1/verify_eval_artifacts.py MUSTER_eval_v1
```

On Linux, `sha256sum -c SHA256SUMS` verifies the same digest. Then, from the code repository:

```bash
MUSTER_HOME=/absolute/path/to/MUSTER_eval_v1 python gate/rescore_beam.py
```

This recomputes all 48 printed values of the main BEAM table from the 51 saved judgment files, with no model/API calls. The archive contains `VERIFIED.json` and `OFFLINE_BEAM_GATE.txt`. It also includes a [copy of the offline verifier](verify_eval_artifacts.py); it verifies all hashes/counts and joins answer/judgment files by `(row, qi)`.

## Honcho provenance

The saved JSONL contains returned context messages. Its companion JSON joins these messages with questions from the frozen question bundles, checked by identifiers for all 1,300 pairs. The harness called `honcho-ai 2.5.0`, `session.context(summary=True, peer_target="user", search_query=question)` after ingestion drained. These files provide the query/context artifacts, not HTTP traffic captures, intermediate retries, or historical ingestion responses. The public harness retains the ingestion configuration and its no-network dry mode.

MUSTER's original annotations and verifier code use Apache-2.0. BEAM source content retains the CC BY-SA 4.0 license declared by the [BEAM](https://huggingface.co/datasets/Mohammadta/BEAM) and [BEAM-10M](https://huggingface.co/datasets/Mohammadta/BEAM-10M) dataset cards. `LICENSE_BEAM` identifies the dataset creators, source and license links and preserves the original BEAM software MIT notice separately. This bundle contains evaluation artifacts rather than newly collected conversations; other benchmarks and model weights are not included.
