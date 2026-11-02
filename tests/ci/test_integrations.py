"""Framework invocation tests: real SDKs, fake MUSTER models, no agent/provider execution."""
import asyncio
import importlib.util
import json
import os
import socket
import subprocess
import sys

import pytest

from muster import Store
from muster.models import Models


@pytest.fixture
def store(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network attempted during a framework tool invocation")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    with Store.open(str(tmp_path / "tools.muster"), models=Models.fake()) as value:
        yield value


def test_namespace_is_lazy():
    code = """
import sys
import muster.integrations
blocked = ('agents', 'openai', 'anthropic', 'requests', 'langchain_core', 'langgraph', 'torch')
assert not any(name in sys.modules for name in blocked)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def _sdk(name):
    if importlib.util.find_spec(name) is None:
        if os.environ.get("MUSTER_REQUIRE_INTEGRATIONS") == "1":
            pytest.fail("required integration dependency is missing: " + name)
        pytest.skip("optional integration dependency is not installed: " + name)


def test_openai_sdk_tool_invocation_and_idempotent_add(store):
    _sdk("agents")
    from agents.tool_context import ToolContext
    from muster.integrations import openai_tools

    remember, recall = openai_tools(store, read_budget=5)
    schema = remember.params_json_schema
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"text", "speaker", "time", "request_id"}
    assert set(recall.params_json_schema["properties"]) == {"query"}

    async def run():
        args = json.dumps(dict(text="The spare cable is in the blue cabinet.", speaker="user", time="",
                               request_id="write-1"))
        write_ctx = ToolContext(context=None, tool_name=remember.name, tool_call_id="write-1", tool_arguments=args)
        first = await remember.on_invoke_tool(write_ctx, args)
        again = await remember.on_invoke_tool(write_ctx, args)
        read_args = json.dumps({"query": "Where is the spare cable?"})
        read_ctx = ToolContext(context=None, tool_name=recall.name, tool_call_id="read-1", tool_arguments=read_args)
        output = await recall.on_invoke_tool(read_ctx, read_args)
        return first, again, output

    first, again, output = asyncio.run(run())
    assert first == {"version": 1, "duplicate": False, "n_records": 1}
    assert again == {"version": 1, "duplicate": True, "n_records": 0}
    assert store.version == 1
    assert output == store.context("Where is the spare cable?", U=5).text
    assert "blue cabinet" in output
    assert [t.name for t in openai_tools(store, include_remember=False)] == ["muster_recall"]


def test_langchain_sync_async_growth_and_read_budget(store):
    _sdk("langchain_core")
    from muster.integrations import langchain_tools

    remember, recall = langchain_tools(store, read_budget=5)
    first = remember.invoke({"text": "The pickup location is Oslo.", "request_id": "one"})
    assert first["version"] == 1
    old = recall.invoke({"query": "pickup location"})
    assert "Oslo" in old

    async def run():
        await remember.ainvoke({"text": "The pickup time is 09:30.", "request_id": "two"})
        return await recall.ainvoke({"query": "pickup time"})

    output = asyncio.run(run())
    assert "09:30" in output
    assert output == store.context("pickup time", U=5).text
    assert store.version == 2
    assert [t.name for t in langchain_tools(store, include_remember=False)] == ["muster_recall"]


def test_langgraph_compiled_graph_calls_real_tool_node(store):
    _sdk("langgraph")
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, MessagesState, StateGraph
    from muster.integrations import langgraph_node

    graph = StateGraph(MessagesState)
    graph.add_node("memory", langgraph_node(store))
    graph.add_edge(START, "memory")
    graph.add_edge("memory", END)
    app = graph.compile()
    remembered = app.invoke({"messages": [AIMessage(content="", tool_calls=[{
        "name": "muster_remember", "args": {"text": "The sensor interval is 12 seconds."}, "id": "add-1"}])]})
    assert json.loads(remembered["messages"][-1].content)["n_records"] == 1
    recalled = app.invoke({"messages": [AIMessage(content="", tool_calls=[{
        "name": "muster_recall", "args": {"query": "sensor interval"}, "id": "read-1"}])]})
    message = recalled["messages"][-1]
    assert message.tool_call_id == "read-1"
    assert message.content == store.context("sensor interval").text
    assert "12 seconds" in message.content


@pytest.mark.parametrize("budget", [0, -1, True, 2.5])
def test_invalid_budget_is_rejected_without_importing_an_sdk(store, budget):
    from muster.integrations import openai_tools

    with pytest.raises(ValueError, match="read_budget"):
        openai_tools(store, read_budget=budget)


def test_optional_dependency_failure_is_actionable(store, monkeypatch):
    import builtins
    from muster.integrations import openai_tools

    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "agents":
            raise ModuleNotFoundError("optional SDK is absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"muster-memory\[agents\]"):
        openai_tools(store)
