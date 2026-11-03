# Declared record boundaries and source order (experimental opt-in)

Use this mode when your producer provides a record array and declares its order.
It keeps each body as one retrieval unit and retains source IDs, original fields,
prefixes and source positions separately. It never infers that a later record
replaces an earlier fact. Dates are not sorted; deletion requests are stored as
text. This is an additional input contract, not the paper's exact replay mode.

```python
import muster

order = muster.SourceOrder(
    source_sha256="a" * 64,  # replace with the original source file's SHA-256
    source_locator="records.json#/records",
    kind="structured_array",  # or an explicitly documented append sequence
    declaration="The producer emits this array in append order.",
    namespace="my-source-v1",
)
records = [{"id": "a", "content": "Ann moved to Paris."},
           {"id": "b", "content": "Ann moved to Seoul."}]
memory = muster.build(records, declared_order=order)
print(memory.context("Where has Ann lived?").text)
assert memory.ordered_records.roundtrip() == records
```

`record_format="content"` preserves `content`/`text` bodies verbatim, including
CRLF and multiline text. Explicit `"step"` (`[step 2] body`) or `"integer_dot"`
(`2. body`) formats separate the declared producer prefix from the body without
inserting a new numbered prefix. Prefixes can be reconstructed exactly and all
records are kept even when serials reset or have gaps. Duplicate supplied IDs,
conflicting content/text fields, missing provenance and a single unstructured
text blob are refused. Generated IDs are occurrence IDs within a source version.
The source hash and declaration are supplied by the caller; this API cannot
independently prove an external producer's order contract. It also records a
canonical hash of the records actually received.

For CLI and HTTP, use the same JSON envelope:

```json
{
  "records": [{"id": "a", "content": "Ann moved to Paris."}],
  "order": {
    "source_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "source_locator": "records.json#/records",
    "kind": "structured_array",
    "declaration": "Producer declares append order.",
    "namespace": "my-source-v1"
  },
  "format": "content"
}
```

```sh
muster query records.json "Where has Ann lived?" --declared-records
muster http --store agent.muster --allow-declared-records
```

`POST /v1/declared-context` accepts
`{"store": <envelope>, "query": "...", "units": 10}`. This endpoint is disabled
by default. When enabled, it builds a stateless memory and returns source record
IDs and provenance with the context; it does not append the envelope to the
persistent store. Python can build the same envelope with
`muster.build_declared(envelope)`.

With no new option, existing SDK/CLI/HTTP behavior is unchanged. Exact presets
reject this mode. Custom `template` values are rejected instead of silently
ignored. Only each record's body is sent to the reader; other fields remain in
`roundtrip()`. To show dates or speakers to the reader, include them in the
producer-rendered body. The mode deliberately uses declared record boundaries rather
than the normal 512-token packing rule; large records can therefore yield large
reader contexts. Set a suitable record granularity and inspect actual context
tokens before model calls. The posthoc 0.95-counter/20-token diagnostic is not
part of this feature. Input and wiring tests do not establish improved answer
quality or reproduce a new paper result.
