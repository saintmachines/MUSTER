"""Bundles: one JSON line per question with the context a published read consumed.

Library versions of harness/make_mab_bundle.py, harness/make_bundle.py, harness/render_rule.py (R4) and the
FactConsolidation tag strip of gate/run_mab_ctx.sh. They write the same bytes, so a regenerated file can be compared
with a published one by its sha256.
"""
import collections
import glob
import hashlib
import json
import os

from .recover import index80, recover_ids
from ..render import per_parent_groups


def _h(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _gold(a):
    return [str(x) for x in a] if hasattr(a, "__len__") and not isinstance(a, str) else [str(a)]


def read_dump(path, *, on_duplicate="error"):
    """{(row, qi): record} of a DRY dump. A repeated (row, qi) is an error unless on_duplicate="last"."""
    recs, dup = {}, 0
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        k = (r["row"], r["qi"])
        dup += k in recs
        recs[k] = r
    if dup and on_duplicate == "error":
        raise ValueError("%s: %d duplicated (row, qi)" % (path, dup))
    return recs, dup


def bundle_mab(dump_path, *, tag, arm, df, nq, out, dsname, view="prune_annot"):
    """make_mab_bundle.py: the scored arm's context per question, checked against its recorded hash."""
    V = arm + "|" + view
    recs, _ = read_dump(dump_path)
    if len(recs) != nq:
        raise ValueError("%s: partial export %d/%d" % (tag, len(recs), nq))
    n = 0
    with open(out, "w") as fo:
        for k in sorted(recs):
            r = recs[k]
            row, qi = k
            ctx = r["ctx|" + V]
            h = _h(ctx)
            if h != r["ctxhash|" + V]:
                raise ValueError("%s %s: context does not hash to its recorded ctxhash" % (tag, k))
            m = df["metadata"].iloc[row]
            qs = list(df["questions"].iloc[row])
            ans = df["answers"].iloc[row]
            ids = list(m["question_ids"]) if m.get("question_ids") is not None else []
            lines = [x for x in ctx.split("\n") if x.strip()]
            c = ctx.count("[CURRENT]")
            fo.write(json.dumps(dict(row=row, qi=qi, qt=r["qt"], dataset=dsname, question=str(qs[qi]), gold=_gold(ans[qi]),
                                     gold_str=r["gold"], question_id=str(ids[qi]) if qi < len(ids) else "", arm=V,
                                     src_tag=tag, ctxhash=h, tok=(r.get("tok_by_arm") or {}).get(V), n_lines=len(lines),
                                     n_current=c, n_units=r.get("n_units"), context=ctx), ensure_ascii=False) + "\n")
            n += 1
    return n


def bundle_beam(dump_paths, *, df, out, arm="flr_4_3_3|prune_annot"):
    """make_bundle.py: every question of every conversation, in parquet order. Later shards win a repeated key,
    as the original did; the repeats are returned so a caller can report them."""
    recs, dups = {}, []
    for f in dump_paths:
        for line in open(f):
            r = json.loads(line)
            k = (r["row"], r["qi"])
            if k in recs:
                dups.append((k, recs[k]["ctxhash|" + arm] == r["ctxhash|" + arm]))
            recs[k] = r
    nq = sum(len(df.iloc[i]["questions"]) for i in range(len(df)))
    if len(recs) != nq:
        raise ValueError("partial export: %d/%d" % (len(recs), nq))
    n = 0
    with open(out, "w") as fo:
        for row in range(len(df)):
            m = dict(df.iloc[row]["metadata"])
            qs = list(df.iloc[row]["questions"])
            ans = df.iloc[row]["answers"]
            for qi in range(len(qs)):
                r = recs[(row, qi)]
                ctx = r["ctx|" + arm]
                if _h(ctx) != r["ctxhash|" + arm]:
                    raise ValueError("%s: context does not hash to its recorded ctxhash" % ((row, qi),))
                fo.write(json.dumps(dict(row=row, qi=qi, conv_id=str(m["conv_id"]), category=str(m["question_types"][qi]),
                                         cat_index=int(m["cat_index"][qi]), question=str(qs[qi]), gold=_gold(ans[qi]),
                                         rubric=[str(x) for x in m["rubric"][qi]], ctxhash=r["ctxhash|" + arm],
                                         n_units=r.get("n_units"), tok=(r.get("tok_by_arm") or {}).get(arm),
                                         context=ctx), ensure_ascii=False) + "\n")
                n += 1
    return n, dups


def r4_bundle(bundle_in, out, units_of_row):
    """render_rule.py R4: recover the ids each context holds, render one block per parent. `units_of_row(row)` returns
    (units, parent) of that row's store under the leg's cut."""
    stat = collections.Counter()
    cache = {}
    with open(out, "w") as fo:
        for line in open(bundle_in):
            r = json.loads(line)
            row = r["row"]
            if row not in cache:
                u, p = units_of_row(row)
                cache[row] = (u, p, index80(u))
            units, parent, idx = cache[row]
            ids = recover_ids(r["context"], units, parent, idx)
            groups = per_parent_groups(ids, parent)
            new = "\n".join("Memory %d:\n%s" % (n + 1, "\n".join(units[i] for i in b)) for n, b in enumerate(groups))
            stat["same_as_input"] += new == r["context"]
            r["context"] = new
            r["ctxhash"] = _h(new)
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            stat["n"] += 1
    return dict(stat)


def notag(bundle_in, out):
    """FactConsolidation: drop the "[CURRENT] " tag the structural render writes before the newest record of a slot."""
    rows = [json.loads(line) for line in open(bundle_in)]
    n_tag = 0
    for d in rows:
        n_tag += d["context"].count("[CURRENT] ")
        d["context"] = d["context"].replace("[CURRENT] ", "")
        if "[CURRENT]" in d["context"]:
            raise ValueError("a [CURRENT] tag without its trailing space survived the strip")
        d["ctxhash"] = _h(d["context"])
    with open(out, "w") as fo:
        for d in rows:
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    return len(rows), n_tag


def shards(results_dir, tag):
    return sorted(glob.glob(os.path.join(results_dir, "%s_s*.jsonl" % tag)) or
                  glob.glob(os.path.join(results_dir, "%s_r*.jsonl" % tag)))
