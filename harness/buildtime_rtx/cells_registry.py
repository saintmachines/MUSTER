"""THE CANONICAL CELL REGISTRY. One source of truth for (cell -> parquet, row, nq, metric).

WHY THIS FILE EXISTS. The same defect occurred FOUR times in this campaign:
    1  PROFILES["factconsolidation"]      structure reachable only by dataset name
    2  queue_shard.sh TTL rows 2/3/4      cell identity from a hand-typed tag
    3  _HAS_ORDER = DSNAME in PROFILES    resolver enabled by registry membership
    4  queue_p24rest.sh                   the row assignments of (2) copied AFTER (2) was fixed
Each was fixed where it appeared, so the next hand-written file re-created it. (4) is the proof:
the rows were corrected in one queue and typed wrong again in the next one the same night.

The class only closes if rows are GENERATED, never typed. This module owns them; `legs()` emits
queue lines; `validate()` checks any existing queue against the registry; and cellscore.score()
refuses a file whose recorded row disagrees. A hand-typed row now fails at three points.

The row->dataset mapping is verified against the parquet's own metadata["qa_pair_ids"],
corroborated by class counts (TTL row 4 has SIX distinct labels = trec_coarse) and by each ICL
name's shot count equalling its blank-line record count exactly (7050, 8296, 6600, 6400).
"""
AR = "data/Accurate_Retrieval-00000-of-00001.parquet"
FCP = "data_all/data/Conflict_Resolution-00000-of-00001.parquet"
LRU = "data_all/data/Long_Range_Understanding-00000-of-00001.parquet"
TTL = "data_all/data/Test_Time_Learning-00000-of-00001.parquet"
BEAM = "beam/beam100k.parquet"   # BEAM 100K (ICLR 2026), serialization S1 in beam/build_parquet.py

# cell -> (parquet, rows, n_questions, metric, protocol_chunk)
CELLS = {
    "shqa": (AR,  [0],                 100, "substring", 512),
    "mhqa": (AR,  [1],                 100, "substring", 512),
    "evq":  (AR,  [2],                 100, "substring", 4096),
    "lme":  (AR,  [17, 18, 19, 20, 21], 300, "judge_lme", 512),
    "fcmh": (FCP, [3],                 100, "substring", 512),
    "fcsh": (FCP, [7],                 100, "substring", 512),
    # the six size configs the board does not use. Same dataset family, 6k/32k/64k against the
    # board's 262k -- a 45x range over which the structure detector must keep firing.
    "fcmh6k":  (FCP, [0],              100, "substring", 512),
    "fcmh32k": (FCP, [1],              100, "substring", 512),
    "fcmh64k": (FCP, [2],              100, "substring", 512),
    "fcsh6k":  (FCP, [4],              100, "substring", 512),
    "fcsh32k": (FCP, [5],              100, "substring", 512),
    "fcsh64k": (FCP, [6],              100, "substring", 512),
    "det":  (LRU, list(range(100, 110)), 71, "substring", 4096),
    "summ": (LRU, list(range(100)),    100, "judge_infbench", 4096),
    "rec":  (TTL, [0],                 200, "recsys_recall@5", 4096),
    "b77":  (TTL, [1],                 100, "substring", 4096),
    "clin": (TTL, [2],                 100, "substring", 4096),
    "nlu":  (TTL, [3],                 100, "substring", 4096),
    "trcC": (TTL, [4],                 100, "substring", 4096),
    "trcF": (TTL, [5],                 100, "substring", 4096),
    # BEAM 100K: 20 conversations, 400 probing questions, scored OUTSIDE the harness by the
    # benchmark authors' own nugget judge (mohammadtavakoli78/BEAM src/evaluation). prot=4096 is
    # our protocol chunk; BEAM has none of its own (its RAG baseline uses variable-length turn pairs).
    "beam": (BEAM, list(range(20)),    400, "judge_beam", 4096),
}
# the inverse, for checking a file against what it claims to be
ROW_OF = {}
for _c, (_p, _r, _n, _m, _k) in CELLS.items():
    for _x in _r:
        ROW_OF.setdefault((_p, _x), _c)


def legs(prefix, cells=None, env=""):
    """Emit queue leg definitions. Rows come from here, never from a keyboard."""
    out = []
    for c in (cells or CELLS):
        pq, rows, nq, _, _ = CELLS[c]
        out.append('"%s_%s|%s|%s|%s|%d"' % (prefix, c, env, pq, ",".join(map(str, rows)), nq))
    return out


def validate(path):
    """Check a hand-written queue against the registry. Returns the list of disagreements."""
    import re
    bad = []
    txt = open(path).read()
    for m in re.finditer(r'"([A-Za-z0-9_]+)\|[^"]*\|\$?(\w+)\|([\d,]+)\|(\d+)"', txt):
        tag, pqvar, rows, nq = m.groups()
        cell = tag.split("_", 1)[1].rsplit("_", 1)[0] if "_" in tag else tag
        if cell not in CELLS:
            continue
        want = ",".join(map(str, CELLS[cell][1]))
        if rows != want:
            bad.append("%s: rows=%s but %s lives on %s" % (tag, rows, cell, want))
        if int(nq) != CELLS[cell][2]:
            bad.append("%s: nq=%s but %s has %d questions" % (tag, nq, cell, CELLS[cell][2]))
    return bad


if __name__ == "__main__":
    import sys, glob, os
    print("REGISTRY: %d cells" % len(CELLS))
    for c, (p, r, n, m, k) in CELLS.items():
        print("  %-6s rows=%-22s nq=%-4d chunk=%-5d %s" % (c, r if len(r) < 6 else "%s..(%d)" % (r[:3], len(r)), n, k, m))
    print()
    os.chdir(os.environ.get("MUSTER_HOME", os.path.dirname(os.path.abspath(__file__))))
    for q in sorted(glob.glob("queue_*.sh")):
        b = validate(q)
        if b:
            print("  %-22s %d DISAGREEMENTS" % (q, len(b)))
            for x in b[:4]:
                print("      %s" % x)
