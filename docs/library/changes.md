# Changes in 0.1.1

## Native numbered logs

For dot-serial logs such as `12. body`, native reads can group revisions by subject and relation, follow named values
to related subjects, and show the reached lines in chain order. The default read budget remains ten lines. These
rules were measured on MemoryAgentBench FactConsolidation. Frozen paper presets and explicit exact plans retain
their existing slot and ordering rules.

Logs numbered in their own style, such as `[step 3] body`, retain the 0.1.0 slot and reading order. All 563 MINT state
contexts and their reader messages were independently reproduced byte for byte against 0.1.0. This preserves the
existing MINT result; it is not a new MINT accuracy improvement.

Serial spelling determines the scope. Dot-serial event logs can omit earlier events from the selected reader context
when those events share a slot, including events needed by history questions. The original source text and records
remain stored. Fact tables with `[step N]` do not gain the new slot or value-walk rules. Both serial forms still use
newest-per-slot selection, so the older slot rule can also omit earlier occurrences it groups together. See
[native and exact behavior](exact.md) for the full boundary.

Long dot serials preserve numeric ordering without depending on Python's decimal-to-integer digit limit. Serials
longer than 600 digits use an internal sortable key and render in canonical decimal form, including Unicode decimal
digits and leading zeros.

## Application input and errors

- Empty-word stores, including emoji and punctuation, can be read.
- Invalid UTF-8 text is refused with `RecordError` or `QueryError` at the public input boundary. `QueryError` is both
  a `MusterError` and a `ValueError`.
- Chat content containing text parts keeps those parts as text. Unsupported parts are refused. One mapping or
  `Record` is accepted as one record; unordered sets are refused.
- `U`, and `units` over HTTP, accept positive integers, including NumPy integers. `None` and zero retain the plan's
  default. Booleans, fractions, numeric strings, infinity and negative counts raise `PlanError`; HTTP returns 400.
- Record metadata must be a mapping. `Store.add` validates persistence before writing. Unknown presets raise
  `PlanError`.
- The HTTP service returns an error response for malformed requests. Command-line usage errors print a concise
  message.

## Structure and browser inspection

Long whitespace runs avoid quadratic pattern matching, and the numbered-log slot graph is computed lazily instead
of materializing a quadratic table.

The browser structure inspector matches SDK whitespace and serial recognition. Slow inspections can be cancelled;
editing the input cancels outstanding work and hides stale output. Input above the existing 200,000 UTF-16 code-unit
limit is refused without truncation. SDK and browser plans, statistics and token counts matched on 1,277 fixed
inputs, with desktop and mobile Chromium checks. This checks configuration preview, not retrieval or answer quality.
