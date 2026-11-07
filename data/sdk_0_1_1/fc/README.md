# Offline FC answer receipts for MUSTER 0.1.1

Run `python3 verify_fc.py` with Python 3.10 or later. The verifier uses only the Python standard library and makes no network requests.
It checks the SHA256 manifest, 4,800 unique rows, QID/gold alignment, every stored paper-cut score, and all six table counts and paired p-values.

The data cover FC 600 and 262K 200, two arms, three readers. `official` names the gpt-4o-mini-2024-07-18 followup reader;
`glm` is GLM-5.3-Flash on Novita with effort max and `luna` is GPT-6 Luna on Azure with effort high.
`frozen010` is the saved baseline; `v15` is the 0.1.1 value walk in hop order at U10.
The saved FC baseline reproduces current 0.1.0 on 599/600 contexts; the differing QID is factconsolidation_sh_32k_no23.
It is correct on both arms for every reader. Removing it preserves all discordant counts and p-values. The registered 600 denominator remains unchanged.

## Fields and limits

`gold_aliases` retains the gold values used by the frozen scorer. `first10_o200k` is the exact stored visible-answer cut used by the paper table.
`paper_cut_correct` applies the included fixed `subem` to that text. The full-answer registered gate is a separate view and is not replaced by this export.
The exporter removed complete think blocks with the preserved scorer pattern, then used o200k_base to take the first 10 visible tokens.
The stdlib verifier recomputes scoring on those exported cuts; it does not independently tokenize the omitted pre-cut source or regenerate model answers.
If tiktoken and its o200k cache are already available locally, `python3 verify_fc.py --check-tokenization` also recomputes the cuts from every retained full visible answer.
That optional check disables network connections and does not install or download anything.

`visible_answer` is present only for `answer_scope=full_visible`. It is null when the original visible answer exceeds 2,000 characters or matches a sensitive-data filter;
in those rows only the first 10-token text and `raw_answer_sha256` are retained. Scope counts are {"full_visible": 4800}.
`raw_answer_sha256` fingerprints the original saved raw visible-channel answer before think removal. It is not a hash of a reasoning log.
`input_sha256` and `context_sha256` fingerprint the audited input messages and context. They do not reconstruct inputs that are omitted here.

No original questions, source contexts, reasoning logs, cost records, account values, or excluded comparison arms are included.
The six-row summary is computed solely from this whitelist; the larger evidence ledger separately records source-input and protocol checks.
Initial GLM output cap was 31,200. Saved FC baseline retries used 131,072; candidate and both 262K arms used 65,536, seeds 42 then 43.
These followup GLM settings use effort max; the original-paper MAB reader used high. The 262K GLM paired p is about 0.06143.

## Upstream notices

MemoryAgentBench source revision 455306dcabc3842526eb83cd4e225e5d486c5c5d:
https://github.com/HUST-AI-HYZ/MemoryAgentBench . Its original MIT notice is retained byte for byte in LICENSE-MemoryAgentBench.txt;
CITATION.bib is retained from that repository's README. Original benchmark questions and contexts are not redistributed.
LICENSE-MUSTER.txt retains the MUSTER source notice. SOURCE_METADATA.json identifies the frozen scorer and release candidate.
