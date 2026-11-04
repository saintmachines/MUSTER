# Use MUSTER in an agent application

MUSTER supplies local `remember` and `recall` tools over an open `Store`.
`remember` appends a record and returns its version, duplicate status, and record count.
`recall` returns exactly `Store.context(query).text`, including the chosen renderer's headers.
The application chooses its models, opens and closes the Store, and decides which user or robot owns it.
Bind a separate Store for each identity that needs separate memory.

Importing `muster.integrations` loads no SDK. Install only the framework you use.

```sh
pip install "muster-memory[models,agents]"
# Or: muster-memory[models,langchain], muster-memory[models,langgraph]
```

For a model-free local example use `Models.fake()` as below. Real retrieval uses the models
configured on the Store. Tool construction and `remember` make no provider calls;
`recall` may load and run the Store's retrieval models.

Use `models` for local retrieval alongside the `agents`, `langchain` or `langgraph` extra for your framework.
For the MCP server, install `muster-memory[models,mcp]`. HTTP needs no extra beyond `models`.

## OpenAI Agents SDK

```python
from muster import Store
from muster.models import Models
from muster.integrations import openai_tools

with Store.open("agent.muster", models=Models.fake()) as store:
    tools = openai_tools(store, read_budget=10)
    # Supply tools to agents.Agent(..., tools=tools).
    # Keep this Store open until every agent/tool call finishes.
```

These are SDK `FunctionTool` objects named `muster_remember` and `muster_recall`.
They execute blocking Store work in a thread, including rebuilding after an append.
The factory creates no Agent or Runner and does not configure credentials or tracing.
An application that runs an OpenAI model separately needs its own provider setup.
Errors propagate to the caller. `include_remember=False` exposes recall alone.

## LangChain

```python
from muster import Store
from muster.models import Models
from muster.integrations import langchain_tools

with Store.open("agent.muster", models=Models.fake()) as store:
    remember, recall = langchain_tools(store, read_budget=10)
    print(remember.invoke({"text": "The spare cable is in the blue cabinet.",
                           "request_id": "write-1"}))
    print(recall.invoke({"query": "Where is the spare cable?"}))
```

The tools support both `invoke` and `ainvoke`. Reusing a `request_id` acknowledges a retry
without appending the record twice. Omit it for unrelated writes. `read_budget` belongs
to application configuration and is absent from the tool's model-facing query schema.
Omitting it uses the plan's own budget.

## LangGraph

```python
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from muster import Store
from muster.models import Models
from muster.integrations import langgraph_node

with Store.open("robot.muster", models=Models.fake()) as store:
    store.add({"text": "The charging dock is beside the west wall."})
    graph = StateGraph(MessagesState)
    graph.add_node("memory", langgraph_node(store, include_remember=False))
    graph.add_edge(START, "memory")
    graph.add_edge("memory", END)
    result = graph.compile().invoke({"messages": [AIMessage(content="", tool_calls=[{
        "name": "muster_recall", "args": {"query": "Where is the charging dock?"}, "id": "read-1"
    }])]})
    print(result["messages"][-1].content)
```

`langgraph_tools` also returns the underlying tools for an existing graph's ToolNode.
The adapter uses append/search semantics; it is not a LangGraph key/value `BaseStore`.
Do not schedule remember and a dependent recall in the same parallel tool batch: wait for
the add result before reading the new version. Stored passages are context data, and the
application should keep its instructions separate from that text.

## Verification scope

The integration tests invoke the installed SDK tools directly and execute a compiled
LangGraph graph with `Models.fake()`. They block outbound socket connections during tool
calls and require no key, provider, or weight download. This checks Store connectivity,
idempotent writes, growth, and rendered output; it does not measure an agent's decisions.

The direct tool calls were tested with these installed versions on CPython 3.12.13:

| Framework package | Tested version |
|---|---|
| openai-agents | 0.24.0 |
| langchain-core | 1.6.12 |
| langgraph | 1.2.16 |

Run `python -m pytest -q tests/ci/test_integrations.py` from a checkout to replay the
checks. Optional SDK tests skip when their dependency is absent; CI sets
`MUSTER_REQUIRE_INTEGRATIONS=1` so a missing dependency fails there.

Official references: [OpenAI agent definitions](https://developers.openai.com/api/docs/guides/agents/define-agents),
[LangChain tools](https://docs.langchain.com/oss/python/langchain/tools),
[LangGraph workflows](https://docs.langchain.com/oss/python/langgraph/workflows-agents).
