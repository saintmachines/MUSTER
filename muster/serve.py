"""Connect MUSTER to agents and devices: an MCP server (Claude, Cursor, any MCP client) and a small HTTP JSON API
(any app, service or robot that can POST). Both wrap one Store file.

  muster mcp  --store agent.muster                 # stdio MCP server
  muster http --store agent.muster --port 8765     # POST /v1/add, /v1/context, /v1/search; GET /v1/info
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .store import Store

INSTRUCTIONS = ("Long-term memory that keeps every word. Call `remember` for anything worth keeping (one fact, "
                "message or observation per call, with who said it and when if known). Call `recall` with a question "
                "before answering anything that depends on the past; it returns the stored passages, grouped by the "
                "part of the history they come from.")


def _blocks(ctx, mem):
    out = []
    for b in ctx.blocks:
        seen, recs = set(), []
        for i in b.ids:
            for r in mem.unit_records(i):
                if id(r) not in seen:             # records hold a dict (meta), so they are not hashable
                    seen.add(id(r))
                    recs.append({"id": r.id, "time": r.time, "speaker": r.speaker})
        out.append({"index": b.index, "parent": str(b.parent), "ids": list(b.ids), "text": b.text, "records": recs})
    return out


class Service:
    """The operations both servers expose, over one Store (thread-safe)."""

    def __init__(self, store: Store, *, allow_declared_records=False):
        self.store = store
        self.allow_declared_records = allow_declared_records
        self.lock = threading.Lock()

    def add(self, records, request_id=None):
        with self.lock:
            r = self.store.add(records, request_id=request_id)
        return {"version": r.version, "duplicate": r.duplicate, "n_records": r.n_records}

    def context(self, query, units=0):
        with self.lock:
            mem = self.store.head()
            ctx = self.store.context(query, U=units or None)
        return {"text": ctx.text, "ctxhash": ctx.ctxhash, "style": ctx.style, "ids": list(ctx.ids),
                "blocks": _blocks(ctx, mem), "version": mem.version}

    def search(self, query, units=0):
        with self.lock:
            mem = self.store.head()
            sel = self.store.search(query, U=units or None)
            ctx = mem.render(sel)
        return {"ids": list(sel.ids), "units": [mem.units[i] for i in sel.ids], "parents": [str(mem.parent[i]) for i in sel.ids],
                "blocks": _blocks(ctx, mem), "version": mem.version, "U": sel.U, "floors": list(sel.floors)}

    def info(self):
        with self.lock:
            mem = self.store.head()
        p = mem.plan
        return {"version": mem.version, "units": len(mem), "parents": len(set(mem.parent)), "branch": p.branch,
                "atom": p.atom, "chunk": p.chunk, "mscale": p.mscale, "U": p.U, "policy": p.policy, "render": p.render,
                "mode": p.mode}

    def declared_context(self, envelope, query, units=0):
        """One opt-in, stateless record-preserving context; does not append to Store."""
        if not self.allow_declared_records:
            raise ValueError("declared records endpoint is disabled; enable it explicitly")
        from .source_order import build_declared
        with self.lock:
            mem = build_declared(envelope, models=self.store._models())
            ctx = mem.context(query, U=units or None)
        return {"text": ctx.text, "ctxhash": ctx.ctxhash, "style": ctx.style,
                "ids": list(ctx.ids), "blocks": _blocks(ctx, mem),
                "record_order": mem.ordered_records.diagnostics(), "stateless": True}


def mcp_server(service: Service, name="muster"):
    try:
        from mcp.server.mcpserver import MCPServer          # mcp >= 2
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as MCPServer   # mcp 1.x
        except ImportError:
            raise SystemExit("the MCP server needs the MCP SDK: pip install 'muster-memory[mcp]'")
    srv = MCPServer(name, instructions=INSTRUCTIONS)

    @srv.tool(description="Store one thing to remember: a fact, a message or an observation. Optional: who said or saw "
                          "it (speaker), when (time, ISO 8601), and a session or episode id.")
    def remember(text: str, speaker: str = "", time: str = "", session: str = "") -> dict:
        rec = {"text": text}
        for k, v in (("speaker", speaker), ("time", time), ("session", session)):
            if v:
                rec[k] = v
        return service.add([rec])

    @srv.tool(description="Store several records at once. Each record: {text, speaker?, time?, session?, id?}. "
                          "A request_id makes a retried call safe: it is applied once.")
    def remember_many(records: list, request_id: str = "") -> dict:
        return service.add(records, request_id=request_id or None)

    @srv.tool(description="Recall what memory holds about a question: the stored passages, verbatim, grouped by where "
                          "they come from. `units` overrides how many passages to read (default: the memory's own).")
    def recall(query: str, units: int = 0) -> str:
        return service.context(query, units)["text"]

    @srv.tool(description="Like recall, but structured: passage ids, texts and their groups.")
    def search_memory(query: str, units: int = 0) -> dict:
        return service.search(query, units)

    @srv.tool(description="How this memory is organised: size, how it is cut and grouped, how much a recall reads.")
    def memory_info() -> dict:
        return service.info()

    return srv


def http_server(service: Service, host="127.0.0.1", port=8765):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                return self._send(200, {"ok": True})
            if self.path == "/v1/info":
                return self._send(200, service.info())
            self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/v1/add":
                    recs = req.get("records") if "records" in req else [req]
                    return self._send(200, service.add(recs, request_id=req.get("request_id")))
                if self.path == "/v1/context":
                    return self._send(200, service.context(req["query"], int(req.get("units") or 0)))
                if self.path == "/v1/search":
                    return self._send(200, service.search(req["query"], int(req.get("units") or 0)))
                if self.path == "/v1/declared-context":
                    return self._send(200, service.declared_context(req["store"], req["query"],
                                                                  int(req.get("units") or 0)))
                return self._send(404, {"error": "not found"})
            except (KeyError, ValueError, TypeError) as e:
                return self._send(400, {"error": "%s: %s" % (type(e).__name__, e)})
    return ThreadingHTTPServer((host, port), H)
