# API reference

## Building and reading a memory

`muster.build(store, plan=None, *, models=None, embed=True, template=DEFAULT_TEMPLATE, normalize_newlines=True) -> Memory`
: `store` is a string, or records: `muster.Record`, mappings (`text` or `content`, `id`, `time` or `timestamp`,
  `speaker` or `role`, `session`, other keys go to `meta`) or plain strings. `plan=None` derives the plan from the
  text; a string names a preset (exact mode); a `Plan` is used as given. `embed=False` defers the embedding pass.
  One mapping or `Record` alone is a list of one. A `content` list of chat parts keeps its text parts, joined by a new
  line; any other part is refused, so pass what should be remembered as text. Text holding half of a surrogate pair
  (usually an emoji cut by a stream) raises `RecordError` at the call that brought it in.

`Memory.search(query, *, U=None, models=None, cache=None, scores=None) -> Selection`
: `U` overrides the plan's read budget: a positive whole number (an int or a numpy integer), or `None` or 0 for the plan's
  own; a bool, a fraction or a numeric string raises `PlanError`. A query that is empty, not text or not valid UTF-8 raises
  `QueryError`. `cache` is a published reranker-cache file's contents (reproduction);
  `scores` a `ScoreCache` (text-keyed, persistent). Exact plans require `PYTHONHASHSEED=0` and the plan's model lengths.

`Memory.render(selection, *, style=None) -> Context`
: `style` is `r4`, `flat` or `numbered`; the default is the plan's. Rendering a selection made on another version of
  the memory raises `StaleSelection`.

`Memory.context(query, **kw) -> Context` is `render(search(query, **kw))`.
`Memory.unit_records(i) -> list[Record]` names the records unit `i` was cut from.
`muster.search`, `muster.render`, `muster.context` are the same calls as functions.

`Context`: `text`, `style`, `ids`, `blocks` (`Block(index, parent, ids, text)`), `ctxhash` (sha1 of the text, 12 hex),
`tok` (o200k tokens), `messages(question, system=None, template="{context}\n\nQuestion: {question}")`.
`muster.compose(*contexts, sep="\n\n")` joins contexts of separately read stores and renumbers their blocks.

`Selection`: `ids` (what render receives), `pool`, `heads` (`ce100`, `sparse30`, `grank30`, `ce_rank`, `flr`, `flrs`),
`floors` (dense, keyword, graph), `n_reach`, `n_reach_sess`, `ce_hits`, `ce_misses`, `U`, `policy`, `mode`.

## Deciding the plan

`muster.derive(store, *, reference_chunk=4096, template=..., normalize_newlines=True) -> Plan`
: the rule of scripts/derive.py on one store. `muster.explain(plan) -> str` describes it and lists the decision steps.

`muster.derive_collection(stores, *, reference_chunk=4096, sample=3, step8=True, mode="native") -> Plan`
: one plan for a benchmark cell from its first `sample` stores (harness/derive_config.py, then step 8).

`Plan` fields: `branch` (numbered, record, prose, rounds, empty), `atom`, `chunk`, `mscale`, `U`, `policy` (flr, flrs,
attprio_sess), `render`, `mode` (native, exact), `frozen`, `supersede`, `reference_chunk`, `encoder`, `reranker`
(`ModelSpec(name, revision, max_length, batch_size)`), `compat`, `stats`, `source`, `trace`; `plan_id`, `to_json()`.

`muster.presets.names()`, `muster.presets.get(name_or_leg_tag)`, `muster.presets.legs()`, `muster.presets.PAPER`.

## A memory that grows

`muster.Store.open(path, *, models=None, plan=None, template=None, readonly=False)`
: one SQLite file (`:memory:` for a scratch store). A second writer on the same file raises `StoreLocked`.
  With `readonly=True` a missing file is an error and SQL writes are refused; new vectors stay in memory. Keep the
  folder writable, since SQLite holds its `-wal` and `-shm` index files beside the store.

`Store.add(records, *, request_id=None) -> AddResult(version, duplicate, n_records)`
: appends in one transaction with no model call; a repeated `request_id` is acknowledged once and appends nothing.

`Store.head() -> Memory`, `Store.search(query, **kw)`, `Store.context(query, **kw)`, `Store.records()`,
`Store.version`, `Store.close()`; usable as a context manager. The head is rebuilt on the first read after an add and
equals a fresh build of all records.

## Models

`muster.Models(device="auto", *, encoder=None, reranker=None, encoder_spec=ENCODER, reranker_spec=RERANKER,
cache_dir=None, local_files_only=False)`
: loads bge-m3 and bge-reranker-v2-m3 at pinned revisions on first use (cuda, else mps, else cpu). Any object with
  `encode(texts, batch_size, normalize_embeddings)` or `score(query, docs)` can be injected.
  `Models.fake(dim=16, quantize=None)` gives deterministic hash models for tests.

## Records and adapters

`muster.Record(text, id=None, time=None, speaker=None, session=None, slot=None, order=None, status="accepted", ...)`;
`muster.serialize(records, template=..., normalize=True) -> (text, spans)`; `muster.numbered(texts, start=1)`.
`muster.adapters.chat(messages, session=None)`, `muster.adapters.locomo(conversation)`,
`muster.adapters.numbered_lines(lines)`.

## Command line

| Command | What it does |
|---|---|
| `muster derive PATH [--json]` | the native plan of a store, including the library's additional serial spellings |
| `muster query PATH "question" [--units N] [--explain]` | build once from a file and print the context |
| `muster add --store S FILE.jsonl [--request-id ID]` | append records |
| `muster context --store S "question" [--units N]` | print the context from a store |
| `muster mcp --store S [--transport stdio]` | MCP server: remember, remember_many, recall, search_memory, memory_info |
| `muster http --store S [--host H] [--port P]` | JSON API: POST /v1/add, /v1/context, /v1/search; GET /v1/info |
| `muster verify cuts --data DIR` | check the cut of every final-leg row against gate/cut_table.json |
| `muster presets` | the paper's presets and the legs each stands for |
| `muster fetch [--models]` | tokenizer and sentence-splitter files (and both models) for offline use |

## Errors

All are `muster.MusterError`: `PlanError`, `RecordError`, `QueryError`, `StoreRefused` (exact mode refuses what the harness refuses,
naming the harness line), `EnvironmentRefused`, `AssetMissing`, `StaleSelection`, `PlanViolation`, `StoreLocked`.
