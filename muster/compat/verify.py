"""Check the library against the paper without the paper's machine: tables of hashes, no benchmark text.

  muster verify cuts --data DIR      every row of the 24 final legs: unit count, unit-list hash, parent-label hash

DIR holds the benchmark files at the paths the legs name (data/..., data_all/data/..., beam/...), each checked against
gate/data.sha256 first. The table (gate/cut_table.json) was written by tools/make_cut_table.py on the experiment machine
from the same library code, after the cut had been checked row by row against the harness (tests/studio/).
"""
import hashlib
import json
import os

from .. import presets
from ..cut import cut
from .ce_cache import ce_sig


def parent_hash(parent):
    h = hashlib.sha1()
    for p in parent:
        h.update(("%s:%s\n" % (type(p).__name__, p)).encode())
    return h.hexdigest()[:16]


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_signature(text, plan):
    c = cut(text, plan)
    return {"units": len(c.units), "unit_sig": ce_sig(list(c.units)), "parent_sig": parent_hash(c.parent),
            "parents": len(set(c.parent)), "text_sha256": text_hash(text)}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_data_hashes(path):
    out = {}
    for line in open(path):
        if line.strip() and not line.startswith("#"):
            sha, rel = line.split()
            out[rel] = sha
    return out


def verify_cuts(data_dir, table_path, hashes_path=None, *, legs=None, out=print):
    """Each row's store text is checked against the table's text hash before it is cut, so a file rebuilt with
    another parquet writer still verifies, and a wrong file is reported, never compared."""
    import pandas as pd
    table = json.load(open(table_path))
    ok = bad = skipped = 0
    for tag, rows in sorted(table["legs"].items()):
        if legs and tag not in legs:
            continue
        leg = presets.legs()[tag]
        path = os.path.join(data_dir, leg["parq"])
        if not os.path.exists(path):
            out("SKIP %-16s %s not found" % (tag, leg["parq"]))
            skipped += len(rows)
            continue
        df = pd.read_parquet(path)
        plan = presets.get(tag)
        n_wrong = 0
        for r, want in rows.items():
            text = str(df["context"].iloc[int(r)]) if int(r) < len(df) else None
            if text is None or ("text_sha256" in want and text_hash(text) != want["text_sha256"]):
                n_wrong += 1
                skipped += 1
                continue
            got = row_signature(text, plan)
            if "text_sha256" not in want:
                got.pop("text_sha256")
            same = got == want
            ok += same
            bad += not same
            if not same:
                out("DIFF %-16s row %s: %s != %s" % (tag, r, got, want))
        out("%-16s %d rows checked%s" % (tag, len(rows) - n_wrong,
                                          (", %d skipped: store text differs from the published one" % n_wrong) if n_wrong else ""))
    out("cut verification: %d equal, %d different, %d skipped (missing or different data)" % (ok, bad, skipped))
    return bad == 0 and ok > 0


def verify_contexts(data_dir, ctx_table_path, cut_table_path, *, legs=None, cache_dirs=(), models=None, nq=None,
                    out=print):
    """Rebuild each leg's stores under its preset, search every question in exact mode and compare the context hash
    with gate/ctx_table.jsonl. With the published reranker caches (cache_dirs) the comparison is exact on CPU;
    without them, scores are computed live and near-equal scores may come out in another order."""
    import pandas as pd
    from ..memory import build
    from ..models import Models
    from .ce_cache import LegacyCECache
    want = {}
    for line in open(ctx_table_path):
        r = json.loads(line)
        want.setdefault(r["leg"], {})[(r["row"], r["qi"])] = r["ctxhash"]
    texts = {tag: {int(r): v.get("text_sha256") for r, v in rows.items()}
             for tag, rows in json.load(open(cut_table_path))["legs"].items()}
    total = equal = 0
    for tag in sorted(want):
        if legs and tag not in legs:
            continue
        leg = presets.legs()[tag]
        path = os.path.join(data_dir, leg["parq"])
        if not os.path.exists(path):
            out("SKIP %-16s %s not found" % (tag, leg["parq"]))
            continue
        plan = presets.get(tag)
        m = models or Models(device="cpu", encoder_spec=plan.encoder, reranker_spec=plan.reranker)
        df = pd.read_parquet(path)
        t = e = 0
        for row in leg["rows"]:
            keys = sorted(k for k in want[tag] if k[0] == row)[:nq] if nq else sorted(k for k in want[tag] if k[0] == row)
            if not keys:
                continue
            text = str(df["context"].iloc[row])
            if texts.get(tag, {}).get(row) and text_hash(text) != texts[tag][row]:
                out("SKIP %-16s row %d: store text differs from the published one" % (tag, row))
                continue
            mem = build(text, plan, models=m)
            cache = {}
            for d in cache_dirs:
                c = LegacyCECache(d, knobs=plan.compat["knobs"])
                if c.has(list(mem.units)):
                    cache = c.load(list(mem.units))
                    break
            qs = list(df["questions"].iloc[row])
            for k in keys:
                ctx = mem.context(str(qs[k[1]]), cache=cache, models=m)
                t += 1
                e += ctx.ctxhash == want[tag][k]
        total += t
        equal += e
        out("%-16s %5d questions, %5d contexts equal" % (tag, t, e))
    out("context verification: %d of %d equal" % (equal, total))
    return total > 0 and equal == total
