# Evaluation artifact supplement v2

This supplements the [base artifact](../eval_v1/README.md). Download both for the combined coverage. Original saved answers, judgments and returned context messages are preserved byte for byte. Multiple formats for one query are not different model calls.

- [GitHub Release: paper-artifacts-v2](https://github.com/saintmachines/MUSTER/releases/tag/paper-artifacts-v2)
- [MUSTER_eval_v2.tar.gz](https://github.com/saintmachines/MUSTER/releases/download/paper-artifacts-v2/MUSTER_eval_v2.tar.gz)
- [SHA256SUMS](https://github.com/saintmachines/MUSTER/releases/download/paper-artifacts-v2/SHA256SUMS)

| Component | Coverage |
|---|---|
| BEAM | 268 saved answer/judgment files, 108,600 physical JSONL rows. Offline re-scoring verifies 196 selected manuscript values and 18 additional printed values. |
| Honcho | 4,793 query/context records across MemoryAgentBench, LoCoMo, LoCoMo-Refined and a separate uncapped FC variant. Raw and reader files are paired by question ID. |
| Ingestion | Source-based reconstruction code, frozen input locators, parameters and batch hashes. |

The base archive's 108 raw files are not duplicated here. Repeated runs remain separate run tags. Honcho requests reconstructed from code and frozen questions are marked as reconstructed; saved messages are not historical HTTP captures. Ingestion reconstruction makes no service calls and does not invent historical ingestion responses.

## Verify and re-score

Download this archive and its `SHA256SUMS` into a directory separate from the base archive. On macOS:

```bash
shasum -a 256 -c SHA256SUMS
tar -xzf MUSTER_eval_v2.tar.gz
python MUSTER_eval_v2/verify_manifest.py
python MUSTER_eval_v2/BEAM/rescore_verify.py --data MUSTER_eval_v2/BEAM
```

On Linux, use `sha256sum -c SHA256SUMS` for the first command. The BEAM re-scorer uses only the Python standard library and makes no model/API calls. Each component has its own README, manifest and provenance.

The BEAM coverage report explicitly lists two aggregate or operational comparisons whose exact run cohorts remain unattributed. Their constituent files are not selected by matching scores. The submitted paper, predictions and scores are unchanged.

## Licenses

MUSTER-produced code and annotations use Apache-2.0. BEAM-origin content retains CC BY-SA 4.0 with attribution in `LICENSE_BEAM` and its separate upstream software notice. LoCoMo and LoCoMo-Refined content retains CC BY-NC 4.0. MemoryAgentBench and other upstream notices are included in the relevant component. Dataset terms are separate from the memory library's license.
