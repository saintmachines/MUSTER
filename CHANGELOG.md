# Changelog

## 0.1.3 — source-derived configuration

- `muster.build(source)` builds one memory store. The new native `muster.build_collection(stores)` builds independent stores under a shared source-derived plan and returns a `MemoryCollection`.
- Collection statistics use the first up to three stores by default; `sample=None` uses all stores. Every member is checked for compatible source boundaries, and stores are cut and searched separately. Ordinary `build(list_of_strings)` remains one store of records.
- Independent records use sibling candidates and flat passages. Prose and dated, speaker-tagged conversation records use the main retrieval channels and parent-grouped context. Complete serialized session-round inputs preserve their source rounds and sessions.
- MUSTER derives retrieval and rendering automatically from source structure before queries. Configuration takes no benchmark name or gold answer.
- Explicit plans, JSON-restored plans and frozen exact presets retain their recorded behavior. Numbered-log and declared-record boundaries remain separately documented.
- Native session rounds also retain turns without a matching user round and render them in their source positions. Exact presets retain their historical round behavior.
- The final collection API replay covers 1,471 non-FC inputs: 1,450 complete reader inputs match the SDK paper-settings evaluation, and 21 LongMemEval inputs change after the source-turn preservation fix. Changed inputs require fresh answers and judgments under the recorded contracts; archived results are reused only for identical inputs. The 200 FC contexts retain their frozen selection-trace and numbered-path checks. Original and SDK 0.1.2 results remain available alongside the new score lineage.
- Flat contexts retain source blocks for HTTP and Python clients while preserving reader text bytes. The browser inspector is rebuilt and checked against the current single-store Python API, including dated speaker records and JSON sessions. Python session literals require the SDK.

## 0.1.2

Corrected the PyPI description's application-guide link and added rendered-description and destination checks.
Its retrieval implementation and browser runtime were unchanged from 0.1.1. Later SDK evaluations record their
paper-settings and per-store automatic conditions separately.

## 0.1.1

Added the native dot-numbered subject–relation slots and value walk, plus input validation and browser inspection
improvements. See the [detailed compatibility notes](docs/library/changes.md).
