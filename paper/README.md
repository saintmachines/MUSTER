# MUSTER paper

[Read the updated draft (PDF)](MUSTER_native_20261109.pdf) · [Project page](https://saintmachines.github.io/MUSTER/) · [Code](https://github.com/saintmachines/MUSTER)

**MUSTER: Top-Ranked, Self-Configuring Agent Memory That Knows a Log from a Novel and Retrieves Each by Its Structure**

## November 10: current SDK results

The updated draft reports library 0.1.2 with the paper settings as the current implementation's MAB result for five readers. The abstract, Table 1, Table 2 and complete-cell tables use the same measured sources. Automatic settings are a separate 4o-mini result. Original implementation rows remain available for comparison; 25 tables, the figure assets and algorithm are preserved. One figure caption now makes the SDK's additional walk reranking explicit. Other benchmarks, the 71.45 six-setting overview and build timings retain their original implementation scope. Individual task cells are not universally improved.

- [35-page clean PDF](MUSTER_native_20261109.pdf) · [Changes in blue](MUSTER_native_20261109_marked.pdf)
- [Diff against the submitted v219I](MUSTER_native_20261109_vs_submitted.diff) · [Diff against the November 7 draft](MUSTER_native_20261109_vs_current.diff)
- [Verified source ZIP](MUSTER_native_20261109_source.zip) · [Clean TeX](MUSTER_native_20261109.tex) · [Marked TeX](MUSTER_native_20261109_marked.tex)
- [42 GPT task cells](native_mab_cells.csv) · [28 GLM/Luna task cells](frontier_native_mab_cells.csv) · [14 automatic task cells](auto4omini_mab_cells.csv)
- [Representative SDK rows](representative_sdk_mab_rows.csv) · [28 SDK policy-control cells](sdk_policy_comparison_cells.csv) · [Conditions and provenance](../docs/data/native_mab_0_1_2.json)

Clean PDF SHA-256: `b8b02cd7db115d6be578deb552b9be401bfae4b0e1f4f60ecb4e49e4ec0ea338`.
Source ZIP SHA-256: `784483e4317f167fd02dae32cb6106de683e467c838261652f92c1383066fae4`.

The eleven-file source ZIP rebuilds to the clean PDF byte for byte with the recorded TeX Live 2025 image. It contains five ancillary CSVs and their README. This working draft is separate from the archived submitted manuscript below; repository availability does not establish arXiv server publication.

## Submitted manuscript

[The author's submitted manuscript](MUSTER.pdf), local revision v219I, retains the reported experiments and retrieval protocol
clarifications. Its first page links to the code repository and project page.

PDF SHA-256: `c959e71d7a0e40f335dfa1d86451e7c454e67c4a0b6f80fbfb4c81853cb42c7d`.

For citation metadata, see [CITATION.cff](../CITATION.cff).

## November 7 working draft and source diff

The [working draft](MUSTER_followup_20261107.pdf) updates two statements about the released reproduction materials.
Its experimental tables and figures are the submitted manuscript's originals.

- [Changes marked in blue](MUSTER_followup_20261107_DIFF.pdf)
- [Source diff against v219I](MUSTER_followup_20261107_vs_v219I.diff)
- [Clean TeX source bundle](MUSTER_followup_20261107_source.zip) · [Marked source bundle](MUSTER_followup_20261107_DIFF_source.zip)
Both working PDFs have 33 pages. The submitted PDF above remains the original manuscript; the working
drafts are distributed separately. The source bundles contain the five files used in the verified build.
