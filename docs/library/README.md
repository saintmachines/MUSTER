# MUSTER memory library

Long-term memory for agents that keeps every word. It reads its own text to decide how to cut, group and search it,
and it makes no LLM call to build or grow the memory. This package is the method of the MUSTER paper as a library:
the paper's runs are reproduced by it, question by question (see [Verification](#verification)).

## Install

See the [0.1.1 changes](https://github.com/saintmachines/MUSTER/blob/main/docs/library/changes.md) for input validation,
native numbered-log reading and their compatibility limits.

Install `muster-memory` from PyPI. The basic install provides structure inspection.
The commands below add local retrieval models and the optional MCP server.

```bash
pip install "muster-memory[models]"        # build, add and search (bge-m3 and bge-reranker-v2-m3, run locally)
pip install "muster-memory[models,mcp]"    # also the MCP server
muster fetch                                  # optional: tokenizer and sentence-splitter files for offline use
```

`pip install muster-memory` alone gives `muster derive` and `muster.derive()`, which need only `tiktoken`. The
package imports as `muster`; `pip install muster` (without `-memory`) installs an unrelated 2017 project.
The paper's harness and the tests run from a clone of the repository.

```bash
git clone https://github.com/saintmachines/MUSTER.git
cd MUSTER
pip install ".[models,mcp]"
```

## Try it on a file

```bash
muster query chat.jsonl "Where does Caroline live now?" --explain     # one JSON record per line, or a .txt file
```

## Use it in three lines

```python
import muster

mem = muster.build([
    {"speaker": "Caroline", "time": "2023-05-08", "text": "I moved to Lisbon last month for the new job."},
    {"speaker": "Melanie",  "time": "2023-05-08", "text": "How do you like the city so far?"},
    # ... thousands more
])
print(mem.context("Where does Caroline live now?").text)    # what your LLM reads, verbatim and grouped
```

`muster.build` derives the plan from the text: a numbered log (most lines start with `12. `), a record store (most of
the text sits in short blank-line records, like a chat) or prose (books, documents). `print(muster.explain(mem.plan))`
shows the decision and why. Nothing is tuned per dataset.

The numbered branch also recognizes serials written in the log's own style (`Step 12: ...`, `#12 ...`, `(12) ...`).
This form is detected when 95% of the lines after the first carry a serial one above the previous line's and
the median line is at most 20 tokens, and the reader sees each line as written. The check was added after the paper and
its two thresholds were set on public benchmark logs; the paper's presets and `derive_collection` do not use it.
The extra native slot rule and value walk described below apply only to the dot form.

## A memory that grows: `Store`

```python
store = muster.Store.open("agent.muster")                     # one SQLite file
store.add({"speaker": "user", "text": "My daughter's name is Ana.", "time": "2026-10-20T09:12"})
store.add(messages, request_id="turn-41")                     # a retried request is applied once
ctx = store.context("What is my daughter's name?")
```

`add` writes records in one transaction and calls no model, so it is safe on a hot path. The next search builds the
new version exactly as a fresh build of all records would be (tested), embedding only texts it has not seen before.
Cached unit vectors and reranker scores survive restarts; new queries still run retrieval models.

## Connect it

Start with the [app connection guide](apps.md) to choose Python, MCP or HTTP,
wire memory around your existing model call, and try copyable client requests.

**MCP** (Claude Desktop, Claude Code, Cursor, any MCP client):

```bash
muster mcp --store agent.muster
```
```json
{"mcpServers": {"muster": {"command": "muster", "args": ["mcp", "--store", "/path/to/agent.muster"]}}}
```
Tools: `remember`, `remember_many`, `recall`, `search_memory`, `memory_info`.

**HTTP** (any app, service or robot):

```bash
muster http --store agent.muster --port 8765
curl -s localhost:8765/v1/add -d '{"records":[{"speaker":"robot","text":"Red mug on the kitchen table","time":"2026-10-20T10:02"}]}'
curl -s localhost:8765/v1/context -d '{"query":"Where did I last see the red mug?"}'
```

**Python agents**: `Store.context(q).text` is a string; put it in the prompt. `ctx.messages(question)` builds a chat
message list; `ctx.blocks` gives the groups with the ids of the passages in each.

Examples: [examples/][examples] (agent loop, robot observation log, MCP client configuration, ROS 2 node).
Reference: [api.md][api]. Checking it against the paper: [verify.md][verify], [exact.md][exact].

## How it works, in one paragraph

Units are the store's own records, packed up to a token budget (512 tokens for record stores, 1,024 for prose) and
never split mid-record. Parents group about eight neighbouring units. A query is matched three ways: meaning (bge-m3),
keywords (BM25) and an entity graph over parents. The read budget (`U` units, 10 by default, 40 for prose) is split
between the three by reserved floors (4/3/3 per ten), so each signal always contributes; candidates are ordered by a
cross-encoder. Selected units are shown verbatim, one block per parent, in stored order. On numbered logs the newest
version of a slot replaces older ones. In native mode a slot is the line's subject and relation without its value, once
two subjects share that relation and a changed subject holds at most 4 lines on average, so `... the sport of cricket`
retires `... the sport of basketball`. Other lines keep the paper's slot, the sentence with its capitalised names masked.
This rule was added after the paper, on FactConsolidation, the one public benchmark family that reaches it, and reads
logs whose lines start with a dot serial (`12. `). A log that states a single subject's facts, says `Alice is a nurse`,
or numbers its lines in its own spelling (`[step 3]`) keeps the paper's slot.

In native mode a read of a numbered log also follows what the lines name. It starts from the subjects the question names
and moves to the lines about whatever those lines name, so `Ada plays the sport of cricket` leads to `cricket was created
in the country of England`, and `The author of Emma is Austen` leads to `Austen's child is ...`. The reranker orders each
hop, and the read takes the best line of every hop, then the second best, up to four lines deep, before the slot graph.
A subject held by more than three slots and by more than 5% of them leads nowhere. The lines it reached come first, in
the order it reached them, and the others follow by serial, so `[348]` can stand above `[179]`. This reads words, not
meaning, and a log whose lines name no subject reads as before, in serial order. It was also added after the paper, on
FactConsolidation, and reads dot-serial logs only; exact mode does not walk.

## Two modes

- **native** (default): everything above, deterministic for given inputs and versions, with no hash seed and no pinned
  environment. Use this in applications.
- **exact**: the paper's harness, step for step, under one of nine presets (`muster presets`), for example
  `muster.build(text, "beam.u20")`. Set sets are iterated as the harness iterated them, so it must run under
  `PYTHONHASHSEED=0` with the gate's package versions. Use this to check a paper number.

With the same plan, native mode departs from exact mode only in the nine places [exact.md][exact] lists, among them canonical
iteration over entity sets and a stable sort for exact dense-score ties.

Automatically derived non-numbered plans use the balanced method with immediate sibling candidates (`flrs`).
The frozen paper presets retain each published run's policy, including `flr` where it was recorded; explicit or
JSON-restored plans are not migrated. See [exact.md][exact] when choosing between automatic planning and replay.

## Verification

The library is checked against the paper's harness and its published outputs on the paper's data, CPU only, with no
API call ([exact.md][exact] has every check, its script and its count). In short:

| Check | Equal |
|---|---|
| Cut of every row of the 24 final legs vs the harness | 296 of 296 rows |
| Render chain of the MemoryAgentBench board from the published dumps | 5,439 of 5,439 questions |
| BEAM-100K search and R4 context vs the harness replay | 800 of 800 questions |
| Other cells replayed, context = the published one | 6,753 of 6,753 questions |

## Optional declared record input

An experimental [declared record mode](https://github.com/saintmachines/MUSTER/blob/main/docs/library/declared-records.md) connects explicit
source boundaries and order metadata to the SDK, CLI and an opt-in HTTP endpoint.
It keeps all versions and does not change the default or paper replay paths.

For a producer-declared record array, `record_format="serial"` reversibly reads prefixes such as
`Step 12: ...`, `#12 ...`, `(12) ...`, `[12] ...`, or `12. ...` using the leading-serial lexer (the integer must end
within the first 32 code points). Pass `declared_order=SourceOrder(...)` to `muster.build`, or use `"format": "serial"`
in the declared-record CLI/HTTP envelope. Prefixes and metadata remain in `mem.ordered_records.roundtrip()`; the body
is what retrieval and the reader receive. This explicit codec does not require the automatic detector's 95%/20-token
conditions. Gaps, resets and repeated markers do not sort or delete records: declared source order and all versions
are preserved, with supersession off. The existing `step` and `integer_dot` codecs remain strict and unchanged.

For a record array or `Store`, automatic serial detection reads only the start of each record's `text`/`content`.
An ID, time, or speaker field never supplies a serial. A literal body beginning `user: Step 1: ...` is not rewritten.
Numbered records keep their serialized metadata in retrieval and reader text, while slots and revision order come
from the body. Each record remains one unit. Multiline bodies are not split into invented records. Unmarked records
keep their own occurrence and cannot enable structural replacement. Paper replay and valid, fully marked plain-text
logs are unchanged. In native lead logs, an unmarked or out-of-bound line keeps its original position and has no
inferred revision clock, and structural replacement is disabled.

For structured records the continuity denominator is the original record pairs, including empty records as missing
markers, and the length median counts each entire record body, including continuation lines. It does not scan for
a serial in arbitrary words later in the body. Raw strings retain the existing line-based rule and thresholds.
An unmarked record has no inferred revision clock. All-marked numbered inputs retain numeric serial display order,
while an array containing unmarked records is displayed in original array order with structural replacement off.

[examples]: https://github.com/saintmachines/MUSTER/tree/main/examples
[api]: https://github.com/saintmachines/MUSTER/blob/main/docs/library/api.md
[verify]: https://github.com/saintmachines/MUSTER/blob/main/docs/library/verify.md
[exact]: https://github.com/saintmachines/MUSTER/blob/main/docs/library/exact.md
