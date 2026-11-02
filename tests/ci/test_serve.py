"""The MCP server and the HTTP API, end to end over a real transport, with fake models."""
import json
import os
import sys
import threading
import urllib.request

import pytest

import muster
from muster.models import Models
from muster.serve import Service, http_server

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _post(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def test_http_api(tmp_path):
    st = muster.Store.open(str(tmp_path / "h.muster"), models=Models.fake(32))
    srv = http_server(Service(st), "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = "http://127.0.0.1:%d" % srv.server_address[1]
    try:
        r = _post(base + "/v1/add", {"records": [{"speaker": "Ann", "text": "Ann's sister Mia lives in Porto.",
                                                   "time": "2026-04-03"},
                                                  {"speaker": "Ben", "text": "Ben bought a blue bicycle."}],
                                     "request_id": "r1"})
        assert r == {"version": 2, "duplicate": False, "n_records": 2}
        assert _post(base + "/v1/add", {"records": [{"text": "x"}], "request_id": "r1"})["duplicate"] is True
        c = _post(base + "/v1/context", {"query": "Where does Mia live?"})
        assert "Porto" in c["text"] and c["blocks"] and c["version"] == 2
        s = _post(base + "/v1/search", {"query": "bicycle", "units": 1})
        assert len(s["ids"]) == 1
        info = json.loads(urllib.request.urlopen(base + "/v1/info").read())
        assert info["units"] >= 1 and info["mode"] == "native"
    finally:
        srv.shutdown()
        st.close()


def test_mcp_stdio(tmp_path):
    pytest.importorskip("mcp")
    import asyncio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    store = str(tmp_path / "m.muster")
    params = StdioServerParameters(command=sys.executable, args=["-m", "muster.cli", "mcp", "--store", store,
                                                                 "--fake-models"],
                                   env=dict(os.environ, PYTHONPATH=ROOT), cwd=ROOT)

    async def run():
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = sorted(t.name for t in (await s.list_tools()).tools)
                assert names == ["memory_info", "recall", "remember", "remember_many", "search_memory"]
                await s.call_tool("remember", {"text": "The spare key is under the blue flowerpot.", "speaker": "Ann"})
                await s.call_tool("remember", {"text": "Ben's flight lands at 6 pm on Friday.", "speaker": "Ben"})
                out = await s.call_tool("recall", {"query": "Where is the spare key?"})
                text = "".join(getattr(c, "text", "") for c in out.content)
                assert "flowerpot" in text
    asyncio.run(run())
