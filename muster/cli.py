"""The `muster` command.

  muster derive PATH [--json]               the plan muster.derive picks (same output as scripts/derive.py, which keeps
                                            the paper's rule, except a log numbered under its own spelling and a serial
                                            log that shows no updates, which is read without replacing anything)
  muster query PATH "question"              build a memory from a file once and print what a reader gets
  muster add --store S.muster FILE.jsonl    append records (one JSON object per line: text, speaker, time, ...)
  muster context --store S.muster "query"   what a reader gets for a query
  muster mcp --store S.muster               serve the store to MCP clients over stdio
  muster http --store S.muster [--port N]   serve the store as a JSON API
  muster verify cuts --data DIR             check the cut of every final-leg row against gate/cut_table.json
  muster verify contexts --data DIR         rerun the paper's searches (exact mode) and compare every context hash
  muster presets                            the paper's presets and the legs each stands for
  muster fetch                              download the tokenizer and sentence-splitter files once, for offline use
"""
import argparse
import json
import sqlite3
import sys

from . import bpe
from .derive import MIN_PARENTS, MIN_UNIT, NUMBERED_TAU, PROSE_STEP, RECORD_STEP, RECORD_TAU, derive
from .errors import MusterError, RecordError
from .io import read_store
from .stats import LEAD_LINE_TOKENS, lead_serial, store_stats

P = 4096


def _decision(serial, record, total, lead=False):
    """The dict scripts/derive.py prints, from the same three numbers (`lead`: numbered under its own spelling)."""
    if serial >= NUMBERED_TAU or lead:
        return dict(store="numbered log", unit_tokens=None, units=None, parent_tokens=None, parents=None,
                    order="newest per slot wins", retrieval="mostly by slot",
                    channel_floors=dict(meaning=2, keyword=0, parent=8, per_selected_units=10))
    if record >= RECORD_TAU:
        store, unit = "record store", max(P // RECORD_STEP, MIN_UNIT)
    else:
        store, unit = "prose", (P // PROSE_STEP if P > MIN_UNIT else P)
    units = total // unit
    mult = next((m for m in (8, 4, 2) if units // m >= MIN_PARENTS), None)
    return dict(store=store, unit_tokens=unit, units=units, parent_tokens=unit * mult if mult else None,
                parents=units // mult if mult else 0, order="none: all versions kept", retrieval="balanced, 3 channels",
                channel_floors=dict(meaning=4, keyword=3, parent=3, per_selected_units=10))


def _show(path, total, d):
    print("MUSTER  %s  %s tokens" % (path, format(total, ",")))
    if d["store"] == "numbered log":
        rows = [("store", "numbered log"), ("units", "one per record"), ("parents", "one per slot"),
                ("order", d["order"]), ("retrieval", d["retrieval"])]
        if "leading_serial_share" in d:
            rows.append(("serials", "each line's own prefix, shown as written"))
    else:
        if not d["units"]:
            unit = "whole store (under one unit)"
        elif d["store"] == "record store":
            unit = "512 tokens, records whole"
        else:
            unit = "%s tokens, 4x as many" % format(d["unit_tokens"], ",")
        par = "%s tokens, %d of them" % (format(d["parent_tokens"], ","), d["parents"]) if d["parent_tokens"] else "none (too short)"
        rows = [("store", d["store"]), ("units", unit), ("parents", par), ("order", d["order"]),
                ("retrieval", d["retrieval"])]
    for k, v in rows:
        print("  %-10s %s" % (k, v))


def cmd_derive(a):
    try:
        text = read_store(a.path, a.column, a.row)
    except FileNotFoundError:
        sys.exit("no such file: %s" % a.path)
    except UnicodeDecodeError:
        sys.exit("%s is not UTF-8 text; save it as UTF-8 and run again" % a.path)
    except MusterError as e:
        sys.exit(str(e))
    st = store_stats(text, P)
    if st.tokens == 0:
        sys.exit("%s is empty: nothing to configure" % a.path)
    share, med = lead_serial(text) if st.serial < NUMBERED_TAU else (0.0, None)
    form = "dot" if st.serial >= NUMBERED_TAU else ("lead" if med is not None and med <= LEAD_LINE_TOKENS else None)
    numbered = derive(text).branch == "numbered"   # the library's own decision, so the two cannot disagree
    lead = numbered and form == "lead"
    d = _decision(st.serial if numbered else 0.0, st.record_share, st.tokens, lead=lead)
    if lead:
        d = dict(leading_serial_share=round(share, 4), median_line_tokens=med, **d)
    if a.json:
        print(json.dumps(dict(path=a.path, tokens=st.tokens, serial_fraction=round(st.serial, 4),
                              record_share=round(st.record_share, 4), **d), indent=1))
    else:
        _show(a.path, st.tokens, d)


def cmd_verify(a):
    import os
    from .compat.verify import verify_cuts
    root = a.repo or os.getcwd()
    legs = set(a.legs.split(",")) if a.legs else None
    if a.what == "cuts":
        ok = verify_cuts(a.data, os.path.join(root, "gate", "cut_table.json"), legs=legs)
    else:
        from .compat.verify import verify_contexts
        ok = verify_contexts(a.data, os.path.join(root, "gate", "ctx_table.jsonl"),
                             os.path.join(root, "gate", "cut_table.json"), legs=legs,
                             cache_dirs=[d for d in (a.cache or "").split(":") if d], nq=a.nq or None)
    sys.exit(0 if ok else 1)


def cmd_presets(a):
    from . import presets
    by = {}
    for tag, leg in presets.legs().items():
        by.setdefault(leg["preset"], []).append(tag)
    for name in presets.names():
        p = presets.get(name)
        shape = ("%s/%s/%s" % (p.atom, p.chunk, p.mscale)) if p.atom else p.branch
        print("%-11s %-9s %-16s U=%-3d %-13s %-8s %s" % (name, p.branch, shape, p.U, p.policy, p.render,
                                                        ", ".join(sorted(by.get(name, [])))))


def cmd_fetch(a):
    for name, dest, what in bpe.fetch(quiet=True):
        print("%-12s %-8s %s" % (name, what, dest))
    from .cut import _ensure_punkt
    _ensure_punkt(exact=False)
    print("punkt_tab    ready")
    if a.models:
        from huggingface_hub import snapshot_download
        from .plan import ENCODER, RERANKER
        for spec in (ENCODER, RERANKER):
            path = snapshot_download(spec.name, revision=spec.revision)
            print("%-12s ready    %s" % (spec.name.split("/")[-1], path))


def _models(a):
    if getattr(a, "fake_models", False):
        from .models import Models
        return Models.fake(32)
    return None


def _json_records(lines, name):
    out = []
    for n, x in enumerate(lines, 1):
        if x.strip():
            try:
                out.append(json.loads(x))
            except json.JSONDecodeError as exc:
                raise RecordError("%s line %d is not JSON; this command reads one JSON record per line" % (name, n)) from exc
    return out


def cmd_add(a):
    from .store import Store
    if a.file == "-":
        recs = _json_records(sys.stdin, "standard input")
    else:
        with open(a.file, encoding="utf-8") as source:
            recs = _json_records(source, a.file)
    with Store.open(a.store, models=_models(a)) as st:
        r = st.add(recs, request_id=a.request_id)
    print(json.dumps({"version": r.version, "duplicate": r.duplicate, "n_records": r.n_records}))


def cmd_context(a):
    from .store import Store
    with Store.open(a.store, models=_models(a), readonly=True) as st:
        ctx = st.context(a.query, U=a.units or None)
    print(ctx.text)


def cmd_query(a):
    from .memory import build
    from .records import as_record
    if a.declared_records:
        from .source_order import build_declared
        with open(a.path, encoding="utf-8") as source:
            envelope = json.load(source)
        mem = build_declared(envelope, models=_models(a))
    elif a.path.endswith(".jsonl") and not a.column_given:
        with open(a.path, encoding="utf-8") as source:
            store = [as_record(x) for x in _json_records(source, a.path)]
        mem = build(store, models=_models(a))
    else:
        store = read_store(a.path, a.column, a.row)
        mem = build(store, models=_models(a))
    if a.explain:
        from .derive import explain
        print(explain(mem.plan), file=sys.stderr)
    print(mem.context(a.question, U=a.units or None).text)


def cmd_mcp(a):
    from .serve import Service, mcp_server
    from .store import Store
    mcp_server(Service(Store.open(a.store, models=_models(a)))).run(transport=a.transport)


def cmd_http(a):
    from .serve import Service, http_server
    from .store import Store
    srv = http_server(Service(Store.open(a.store, models=_models(a)),
                              allow_declared_records=a.allow_declared_records), a.host, a.port)
    print("muster http on %s:%d (POST /v1/add /v1/context /v1/search, GET /v1/info)" % (a.host, a.port), flush=True)
    srv.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="muster", description="MUSTER memory: derive, presets, fetch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("derive", help="the plan of a store, from its own text")
    d.add_argument("path")
    d.add_argument("--column", default="context")
    d.add_argument("--row", type=int, default=0)
    d.add_argument("--json", action="store_true")
    d.set_defaults(fn=cmd_derive)
    q = sub.add_parser("query", help="build a memory from a file and print the context for a question")
    q.add_argument("path")
    q.add_argument("question")
    q.add_argument("--units", type=int, default=0)
    q.add_argument("--column", default=None)
    q.add_argument("--row", type=int, default=0)
    q.add_argument("--explain", action="store_true", help="print the plan and why, to stderr")
    q.add_argument("--declared-records", action="store_true",
                   help="native opt-in: JSON envelope with explicit record boundaries and source-order provenance")
    q.add_argument("--fake-models", action="store_true", help=argparse.SUPPRESS)
    q.set_defaults(fn=cmd_query)
    for name, fn, extra in (("add", cmd_add, ("file",)), ("context", cmd_context, ("query",)),
                            ("mcp", cmd_mcp, ()), ("http", cmd_http, ())):
        p = sub.add_parser(name)
        p.add_argument("--store", required=True)
        p.add_argument("--fake-models", action="store_true", help=argparse.SUPPRESS)
        for x in extra:
            p.add_argument(x)
        if name == "add":
            p.add_argument("--request-id", default=None)
        if name == "context":
            p.add_argument("--units", type=int, default=0)
        if name == "mcp":
            p.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
        if name == "http":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8765)
            p.add_argument("--allow-declared-records", action="store_true",
                           help="enable stateless /v1/declared-context for explicit source-order envelopes")
        p.set_defaults(fn=fn)
    v = sub.add_parser("verify", help="check the library against the paper's tables (no benchmark text needed)")
    v.add_argument("what", choices=["cuts", "contexts"])
    v.add_argument("--cache", default="", help="contexts: colon-separated directories of published reranker caches")
    v.add_argument("--nq", type=int, default=0, help="contexts: at most this many questions per store")
    v.add_argument("--data", required=True, help="directory with the benchmark files at the legs' paths")
    v.add_argument("--repo", default=None, help="repository root holding gate/ (default: the current directory)")
    v.add_argument("--legs", default="", help="comma-separated leg tags (default: all)")
    v.set_defaults(fn=cmd_verify)
    sub.add_parser("presets", help="the paper's presets").set_defaults(fn=cmd_presets)
    f = sub.add_parser("fetch", help="download tokenizer and sentence-splitter files (and, with --models, both models)")
    f.add_argument("--models", action="store_true", help="also download bge-m3 and bge-reranker-v2-m3 at the pinned revisions")
    f.set_defaults(fn=cmd_fetch)
    a = ap.parse_args(argv)
    if a.cmd == "query":
        a.column_given = a.column is not None
        a.column = a.column or "context"
    try:
        a.fn(a)
    except MusterError as e:
        sys.exit(str(e))
    except FileNotFoundError as e:
        sys.exit("no such file: %s" % e.filename)
    except UnicodeDecodeError:
        sys.exit("%s is not UTF-8 text; save it as UTF-8 and run again" % (getattr(a, "path", None) or getattr(a, "file", "the input")))
    except sqlite3.Error as e:
        sys.exit("%s: %s" % (getattr(a, "store", "the store"), e))
    except (OSError, ValueError) as e:
        sys.exit(str(e) or type(e).__name__)


if __name__ == "__main__":
    main()
