#!/usr/bin/env python3
"""MUSTER detection: point it at a memory store and it sets its own strategy.

    python scripts/derive.py path/to/store.txt

No model or key, nothing to tune, and the store never leaves the machine. Two statistics of the stored text decide everything, by the paper's rule
(Algorithm 1, Part I), with the thresholds fixed before any experiment (harness/derive_config.py):

  numbered log  at least 80% of lines open with a serial ("12. ...")
                units = records, parents = the slot each record updates, the newest version of a slot wins,
                retrieval mostly by slot (floors meaning 2, keyword 0, parent 8 per 10 selected units)
  record store  at least 90% of the text sits in records of at most 4,096 tokens
                units = 512 tokens with records kept whole, parents = 8 units
  prose         everything else
                units = 1,024 tokens: the 4,096-token chunk searched at a quarter of its size with four times as many
                units, so the token budget is unchanged; parents = 8 units
  Parents shrink to 4 or 2 units if fewer than ten would remain, else there is no parent layer. Off the numbered-log
  branch retrieval is balanced (floors meaning 4, keyword 3, parent 3 per 10 selected units; they scale with the number
  of units read, so prose read at 40 units uses 16, 12, 12). Tokens are cl100k_base.

These are the settings of the paper's runs (harness/len_legs.json: record stores 512 / 4,096, prose 1,024 / 8,192). The
paper's tables differ in two places, both stated in the paper: LongMemEval is read in the benchmark's own rounds and
sessions, and each MemoryAgentBench cell is decided over three sampled rows, where this script decides one store.

Formats: .txt; .jsonl and .parquet take --column (default "context"); .parquet also --row (default 0) and needs pandas
and pyarrow. --json prints the decision for a pipeline.
"""
import argparse, json, re, sys

NUM = re.compile(r"^\s*\d+\.\s+\S")    # a line that opens with "<integer>. <body>"
NUMBERED_TAU, RECORD_TAU, MIN_PARENTS = 0.80, 0.90, 10
P = 4096                     # reference chunk: MemoryAgentBench's protocol size
RECORD_STEP, MIN_UNIT = 8, 512   # record stores: P / 8, never below 512
PROSE_STEP = 4                   # prose: searched at P / 4 with four times the units (Algorithm 1, step 8)
MAX_LINES = 20000


def load(path, column, row):
    try:
        if path.endswith(".parquet"):
            try:
                import pandas as pd
            except ImportError:
                sys.exit("reading .parquet needs pandas and pyarrow: pip install pandas pyarrow")
            df = pd.read_parquet(path)
            if column not in df.columns:
                sys.exit("%s has no column %r; pass --column NAME" % (path, column))
            if not 0 <= row < len(df):
                sys.exit("%s has %d rows; pass --row between 0 and %d" % (path, len(df), len(df) - 1))
            return str(df[column].iloc[row])
        if path.endswith(".jsonl"):
            parts = []
            for n, line in enumerate(open(path, encoding="utf-8"), 1):
                if line.strip():
                    try:
                        parts.append(str(json.loads(line)[column]))
                    except json.JSONDecodeError:
                        sys.exit("%s line %d is not JSON" % (path, n))
                    except (KeyError, TypeError):
                        sys.exit("%s line %d has no field %r; pass --column NAME" % (path, n, column))
            return "\n\n".join(parts)
        return open(path, encoding="utf-8").read()
    except FileNotFoundError:
        sys.exit("no such file: %s" % path)
    except UnicodeDecodeError:
        sys.exit("%s is not UTF-8 text; save it as UTF-8 and run again" % path)


def statistics(text, enc):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    total = fit = 0
    for rec in text.split("\n\n"):
        if rec.strip():
            t = len(enc.encode(rec, disallowed_special=()))
            total += t
            fit += t if t <= P else 0
    lines = [l for l in text.split("\n") if l.strip()][:MAX_LINES]
    serial = sum(1 for l in lines if NUM.match(l)) / float(len(lines)) if lines else 0.0
    return serial, (fit / float(total) if total else 0.0), total


def derive(serial, record, total):
    if serial >= NUMBERED_TAU:
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


def show(path, total, d):
    print("MUSTER  %s  %s tokens" % (path, format(total, ",")))
    if d["store"] == "numbered log":
        rows = [("store", "numbered log"), ("units", "one per record"), ("parents", "one per slot"),
                ("order", d["order"]), ("retrieval", d["retrieval"])]
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


def main():
    ap = argparse.ArgumentParser(prog="derive.py", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description="Set MUSTER's strategy for a memory store from its own text, with no model.",
                                 epilog="example:  python scripts/derive.py path/to/store.txt\n\n"
                                        "numbered log  -> one unit per record, newest version of each slot wins\n"
                                        "record store  -> 512-token units, records kept whole\n"
                                        "prose         -> 1,024-token units, four times as many, same token budget\n"
                                        "The full rule and its two stated exceptions are written at the top of this file.")
    ap.add_argument("path", help="the store: a .txt file, or .jsonl / .parquet")
    ap.add_argument("--column", default="context", help=argparse.SUPPRESS)
    ap.add_argument("--row", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--json", action="store_true", help="print the decision as JSON")
    a = ap.parse_args()
    try:
        import tiktoken
    except ImportError:
        sys.exit("needs tiktoken: pip install tiktoken")
    enc = tiktoken.get_encoding("cl100k_base")
    serial, record, total = statistics(load(a.path, a.column, a.row), enc)
    if total == 0:
        sys.exit("%s is empty: nothing to configure" % a.path)
    d = derive(serial, record, total)
    if a.json:
        print(json.dumps(dict(path=a.path, tokens=total, serial_fraction=round(serial, 4), record_share=round(record, 4), **d), indent=1))
    else:
        show(a.path, total, d)


if __name__ == "__main__":
    main()
