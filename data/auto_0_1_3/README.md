# SDK 0.1.3 collection Auto evidence

The current release uses these two receipts together:

- `API_INPUT_REPLAY.json`: final SDK source hashes, the complete input inventory, 1,450 matching and 21 changed non-FC reader messages, and the separate FC frozen-trace scope.
- `LME_REFRESH.json`: all 300 LongMemEval judgments per reader, combining 279 archived identical-input positions with evaluation of the 21 changed positions under the recorded contracts.

The 70 task-reader cells and their aggregate values are in `../../docs/data/auto_0_1_3.json` and `../../paper/collection_auto_mab_cells.csv`.

`API_INPUT_IDENTITY.json` is an earlier development-candidate receipt retained unchanged for history. Its different source hashes do not validate the final release. Input agreement alone is not an answer-quality measurement.
