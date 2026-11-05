# Connect MUSTER to your app

MUSTER supplies stored passages to your existing answer model. Python, MCP and
HTTP are three interfaces to the same memory; choose the one your app already
uses. Structure inspection is a separate, smaller starting point.

| Your next step | Use | Install |
|---|---|---|
| Add memory inside a Python app | `Store.add` and `Store.context` | `pip install "muster-memory[models]"` |
| Let Claude, Cursor or another MCP host use memory tools | MCP server and the host's configuration | `pip install "muster-memory[models,mcp]"` |
| Call memory from a JavaScript backend, service or robot | HTTP server and JSON requests | `pip install "muster-memory[models]"` |
| See which structure the rule detects | Browser inspector or `muster derive` | `pip install muster-memory` for the CLI |

The retrieval models run locally. Their weights are downloaded on first use;
the first search can therefore take longer. `muster fetch --models` prepares
the weights and tokenizer files in advance. A GPU is optional. The base install
provides inspection; it does not include the retrieval models. Your app chooses
and configures its answer model separately.

## Python: put memory around the model call you already have

Retrieve before answering, then save the completed exchange. Keep one Store open
for the lifetime of the app's memory owner and close it when that owner stops.
Use a separate file for each user whose memory should stay separate.

```python
import muster

with muster.Store.open("user-42.muster") as store:
    question = "Where is the spare cable?"
    messages = store.context(question).messages(
        question, system="Use relevant memory when answering."
    )
    answer = call_model(messages)  # your app's existing function; returns text
    store.add([
        {"speaker": "user", "text": question},
        {"speaker": "assistant", "text": answer},
    ])
```

This is an integration template: `call_model` is your existing model call, not a
function supplied by MUSTER. A [reusable callback example](../../examples/chat_app.py)
puts the same flow in `answer_with_memory`. Its `request_id` prevents duplicate
stored turns on retries; your app still owns retries and response caching for
the answer model. Use a new `request_id` for each new turn. If you also log
failed requests, record those in your app's input handler.

For a runnable, no-key wiring check from a repository checkout:

```bash
python -m pip install "muster-memory[models]"
python examples/chat_app.py --demo
```

That demo uses fake retrieval models and a fixed reply. It checks the connection
to the reader and persistence across reopening, not answer quality. The normal
`Store.open` example above uses the real local retrieval models.

## MCP: register the server with your host

Install `muster-memory[models,mcp]` in the Python environment your host will run.
For hosts that use `mcpServers`, print a configuration with the current Python
executable and Store path already filled in:

```bash
python - <<'EOF'
import json, os, sys
store = os.path.abspath("mcp-demo.muster")
server = {"command": sys.executable, "args": ["-m", "muster.cli", "mcp", "--store", store]}
print(json.dumps({"mcpServers": {"muster": server}}, indent=2))
EOF
```

Run this in the environment where you installed MUSTER, then paste the printed
JSON into your host's configuration and restart or reconnect the host. Keep that
Python environment available. Its absolute path avoids relying on the GUI app
finding `muster` on PATH. The host starts the stdio server; do not also launch a
second writer for the same file. The configuration location depends on the host.

The server exposes `remember`, `remember_many`, `recall`, `search_memory` and
`memory_info`. To check the connection, ask the host to save a small fact with
`remember`, retrieve it with `recall`, then reconnect and retrieve it again.
`remember_many` accepts `request_id` for retry-safe storage. Providing tools does
not force a model to call them; your app or host decides when to save and recall.

## HTTP: run one server, then call it from your backend

In terminal 1, after installing `muster-memory[models]`:

```bash
muster http --store http-demo.muster --port 8765
```

The server stays in the foreground. In terminal 2:

```bash
curl --fail --silent --show-error http://127.0.0.1:8765/health
curl --fail --silent --show-error http://127.0.0.1:8765/v1/add \
  -H 'Content-Type: application/json' \
  -d '{"records":[{"speaker":"user","text":"The spare cable is in the blue cabinet."}],"request_id":"cable-1"}'
curl --fail --silent --show-error http://127.0.0.1:8765/v1/context \
  -H 'Content-Type: application/json' \
  -d '{"query":"Where is the spare cable?"}'
```

`/v1/context` returns JSON whose `text` field contains the passages to pass to your
app's answer model. It does not generate the final answer. Repeating the same
`/v1/add` request acknowledges it without adding the record twice. Restarting
the server with the same file retains the records.

A Node.js backend can make the same request without a client SDK:

```javascript
const response = await fetch("http://127.0.0.1:8765/v1/context", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "Where is the spare cable?" })
});
if (!response.ok) throw new Error(`MUSTER HTTP ${response.status}`);
const { text } = await response.json();
// Add text as memory context to the model request your backend already makes.
console.log(text);
```

The default address is local to the server machine. For a web frontend, call your
own backend, which calls MUSTER; the built-in server is not a cross-origin browser
API. One service process owns a Store file; several clients can call that
service. The Python, MCP and HTTP demos use different files so they can be tried
independently.

## Inspect: check the rule before loading retrieval models

The [browser inspector](https://saintmachines.github.io/MUSTER/try/) runs locally
in your browser and needs no API key. It shows detected structure and settings,
not a working retrieval service or chatbot.

For a copyable CLI example, create a real file first:

```bash
cat > sample-memory.txt <<'EOF'
1. The spare cable is in the blue cabinet.
2. The meeting starts at 09:30.
3. The charging dock is beside the west wall.
EOF
muster derive sample-memory.txt
```

Replace the sample file with your own text once this runs. The command prints the
selected branch and configuration; it does not save a Store or answer questions.

## Agent frameworks

The existing adapters attach tools to your agent; they do not replace its model
client or application loop. Install the extra for the framework you use, then
keep its Store open while the agent runs.

- [OpenAI Agents SDK](integrations.md#openai-agents-sdk): `muster-memory[models,agents]`
- [LangChain](integrations.md#langchain): `muster-memory[models,langchain]`
- [LangGraph](integrations.md#langgraph): `muster-memory[models,langgraph]`

Those examples distinguish model-free tool tests from real retrieval. For the
underlying Store API, see [api.md](api.md).
