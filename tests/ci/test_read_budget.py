"""A read's unit count is a positive whole number, an int or a numpy integer. None or 0 reads the plan's own count. A bool,
a fraction, a numeric string, a negative number or infinity is refused in the SDK and gets 400 over HTTP; they read as 1,
1, 2 or failed with 500 before (found by Codex on candidate v12)."""
import importlib
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

import muster
from muster.errors import MusterError

FAKE = muster.Models.fake(16)
BAD = [1.9, 2.0, True, False, "2", float("inf"), float("nan"), -1, [2]]


@pytest.fixture(scope="module")
def mem():
    return muster.build("Alice likes tea.\n\nBob likes coffee.\n\nCarol likes juice.", models=FAKE)


@pytest.mark.parametrize("op", ["search", "context"])
@pytest.mark.parametrize("U", BAD, ids=repr)
def test_a_count_that_is_not_a_positive_whole_number_is_refused(mem, op, U):
    with pytest.raises(MusterError):
        getattr(mem, op)("tea", U=U)


@pytest.mark.parametrize("op", ["search", "context"])
def test_numpy_integers_are_counts(mem, op):
    a, b = getattr(mem, op)("tea", U=np.int64(2)), getattr(mem, op)("tea", U=2)
    assert a.ids == b.ids
    if op == "search":
        assert a.U == b.U == 2


@pytest.mark.parametrize("op", ["search", "context"])
@pytest.mark.parametrize("U", [None, 0, np.int64(0)], ids=repr)
def test_none_and_zero_read_the_plans_own_count(mem, op, U):
    a, b = getattr(mem, op)("tea", U=U), getattr(mem, op)("tea")
    assert a.ids == b.ids
    if op == "search":
        assert a.U == b.U == mem.plan.U


def test_a_huge_count_reads_like_a_large_one(mem):
    assert mem.search("tea", U=10 ** 30).ids == mem.search("tea", U=1000).ids
    assert mem.context("tea", U=10 ** 30).text == mem.context("tea", U=1000).text


class _Store:
    """The Store calls Service makes, answered by one Memory."""
    def __init__(self, memory):
        self.memory = memory

    def head(self):
        return self.memory

    def context(self, query, **kw):
        return self.memory.context(query, **kw)

    def search(self, query, **kw):
        return self.memory.search(query, **kw)


@pytest.fixture
def post(mem, monkeypatch):
    serve = importlib.import_module("muster.serve")  # the module; `muster.serve` may name something else
    monkeypatch.setattr(serve, "ThreadingHTTPServer", lambda address, handler: SimpleNamespace(RequestHandlerClass=handler))
    handler = serve.http_server(serve.Service(_Store(mem))).RequestHandlerClass

    def call(path, raw):
        raw = raw if isinstance(raw, bytes) else json.dumps(raw).encode()
        request = object.__new__(handler)
        request.path, request.headers, request.rfile = path, {"Content-Length": str(len(raw))}, io.BytesIO(raw)
        out = []
        request._send = lambda status, payload: out.append((status, payload))
        request.do_POST()
        assert len(out) == 1
        return out[0]
    return call


@pytest.mark.parametrize("path", ["/v1/search", "/v1/context"])
@pytest.mark.parametrize("units", [b"1.9", b"2.0", b"true", b'"2"', b"1e309", b"-1", b"[2]", b"NaN"])
def test_http_refuses_a_count_that_is_not_a_positive_whole_number(post, path, units):
    status, payload = post(path, b'{"query": "tea", "units": ' + units + b"}")
    assert status == 400 and "error" in payload


@pytest.mark.parametrize("path", ["/v1/search", "/v1/context"])
def test_http_reads_whole_numbers_and_the_default(post, path):
    base = post(path, {"query": "tea"})
    assert base[0] == 200
    assert post(path, {"query": "tea", "units": 0}) == base
    assert post(path, {"query": "tea", "units": None}) == base
    status, payload = post(path, {"query": "tea", "units": 2})
    assert status == 200 and "error" not in payload
    status, payload = post(path, {"query": "tea", "units": 10 ** 18})
    assert status == 200 and "error" not in payload
