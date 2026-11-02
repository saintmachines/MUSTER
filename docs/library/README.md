# MUSTER memory library

Long-term memory for agents that keeps every word. It reads its own text to decide how to cut, group and search it,
and it makes no LLM call to build or grow the memory. This package is the method of the MUSTER paper as a library:
the paper's runs are reproduced by it, question by question (see [Verification](#verification)).

## Install

```bash
pip install "muster-memory[models]"          # build, add and search (bge-m3 and bge-reranker-v2-m3, run locally)
pip install "muster-memory[models,mcp]"      # also the MCP server
muster fetch                                  # optional: tokenizer and sentence-splitter files for offline use
```

`pip install muster-memory` alone gives `muster derive` and `muster.derive()`, which need only `tiktoken`. The
package imports as `muster`; `pip install muster` (without `-memory`) installs an unrelated 2017 project.
Until the first release, install from the repository folder: `pip install ".[models,mcp]"`.

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

## A memory that grows: `Store`

```python
store = muster.Store.open("agent.muster")                     # one SQLite file
store.add({"speaker": "user", "text": "My daughter's name is Ana.", "time": "2026-10-20T09:12"})
store.add(messages, request_id="turn-41")                     # a retried request is applied once
ctx = store.context("What is my daughter's name?")
```

`add` writes records in one transaction and calls no model, so it is safe on a hot path. The next search builds the
new version exactly as a fresh build of all records would be (tested), embedding only texts it has not seen before.
Scores and vectors are cached by text, so a restart costs nothing.

## Connect it

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

Examples: [examples/](../../examples/) (agent loop, robot observation log, MCP client configuration, ROS 2 node).
Reference: [api.md](api.md). Checking it against the paper: [verify.md](verify.md), [exact.md](exact.md).

## How it works, in one paragraph

Units are the store's own records, packed up to a token budget (512 tokens for record stores, 1,024 for prose) and
never split mid-record. Parents group about eight neighbouring units. A query is matched three ways: meaning (bge-m3),
keywords (BM25) and an entity graph over parents. The read budget (`U` units, 10 by default, 40 for prose) is split
between the three by reserved floors (4/3/3 per ten), so each signal always contributes; candidates are ordered by a
cross-encoder. Selected units are shown verbatim, one block per parent, in stored order. On numbered logs the newest
version of a slot replaces older ones.

## Two modes

- **native** (default): everything above, deterministic for given inputs and versions, with no hash seed and no pinned
  environment. Use this in applications.
- **exact**: the paper's harness, step for step, under one of nine presets (`muster presets`), for example
  `muster.build(text, "beam.u20")`. Set sets are iterated as the harness iterated them, so it must run under
  `PYTHONHASHSEED=0` with the gate's package versions. Use this to check a paper number.

With the same plan, native mode departs from exact mode only where the harness's choice depended on hash or hardware order: canonical
iteration over entity sets, and a stable sort for exact dense-score ties.

Automatically derived non-numbered plans use the balanced method with immediate sibling candidates (`flrs`).
The frozen paper presets retain each published run's policy, including `flr` where it was recorded; explicit or
JSON-restored plans are not migrated. See [exact.md](exact.md) when choosing between automatic planning and replay.

## Verification

The library is checked against the paper's harness and its published outputs on the paper's data, CPU only, with no
API call ([exact.md](exact.md) has every check, its script and its count). In short:

| Check | Equal |
|---|---|
| Cut of every row of the 24 final legs vs the harness | 296 of 296 rows |
| Render chain of the MemoryAgentBench board from the published dumps | 5,439 of 5,439 questions |
| BEAM-100K search and R4 context vs the harness replay | 800 of 800 questions |
| Other cells replayed, context = the published one | 6,753 of 6,753 questions |
