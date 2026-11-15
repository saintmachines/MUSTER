# Fine retrieval and original-record restoration in 1.0.0

MUSTER uses the source's structure to decide when finer retrieval is appropriate. Deriving a plan requires no language model, query, answer label or benchmark name.

At the default reference scale P=4,096, a native record store normally uses 512-token units and U10. The additional record-branch test opens only when the store is larger than U×P (40,960 tokens) and at least 90% of its record tokens retain their source records at the finer 128-token scale. The refined plan uses 128-token units and U40, and derives its parent groups again. Other reference scales use the corresponding quarter step.

Typed input uses each serialized original `Record` when measuring fine coverage. Plain text uses its blank-line records. The decision is made at the same store or collection boundary as the original derivation; a collection's statistics are not a count of its messages. Existing numbered-log and rounds decisions are evaluated first.

A refined selection can restore the original records it touches when they fit the token budget of the selected units. It preserves source order and does not request another embedding or language-model pass. `Selection.restored` reports the unit-token budget, tokens used and records restored.

The selected-unit budget and final rendered-context size are different quantities. U40 limits the selected units; it does not force a 5,120-token fill. Headers, grouping and tokenization affect the final reader text. Measure the rendered text when comparing reader budgets.

Declared record-order calls retain one source occurrence per record unless their documented derivation permits refinement. Explicit plans, closed refinement paths, numbered logs and frozen exact-mode paper presets preserve their existing behavior. The new rule is not applied by changing a benchmark's input into a different source representation.

The published answer-quality tables for 0.1.2 and 0.1.3 keep their SDK labels, original result files and evaluated source snapshots. Source replay, package tests and evidence-recall diagnostics do not establish 1.0.0 generated-answer accuracy. The browser structure inspector remains a separately labeled 0.1.3 demonstration until its own Python/JavaScript parity checks cover the new rule.
