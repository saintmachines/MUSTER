"""The paper's harness as a test oracle: selected functions of harness/lme_pf29.py, executed in a namespace of our own.

Nothing here is imported by the muster package. The functions are taken from the harness file at test time by name,
so a test always compares the library with the code that produced the paper's numbers, never with a copy.
"""
import ast
import os
import re
import string
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(ROOT, "harness")
PF29 = os.path.join(HARNESS, "lme_pf29.py")

CUT_FUNCS = ["norm", "norm_dataset", "slotkey", "sessions_from_context", "rounds_of", "gold_rounds_per_question",
             "_hd_chunk_starts", "detect_parents", "_atom_split", "_atom_choose", "chunk_official", "_fc_profile",
             "load_row"]
CUT_CONSTS = ["DATASET_MAPPING", "HDET_CANDIDATES", "_ATOM_JOIN"]


def _sources(path, names):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            out[node.name] = ast.get_source_segment(src, node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in names:
            out[node.targets[0].id] = ast.get_source_segment(src, node)
    missing = [n for n in names if n not in out]
    if missing:
        raise RuntimeError("oracle: %s not found in %s" % (missing, path))
    return [out[n] for n in names]


def cut_namespace(*, ATOM="sentence", CHUNK=4096, MSCALE=0, AUTOSTRUCT=0, AS_SLOT="1", HDETECT=0):
    """A namespace holding the harness's cutting code under the given leg settings."""
    import tiktoken
    if HARNESS not in sys.path:
        sys.path.insert(0, HARNESS)          # load_row's AUTOSTRUCT branch imports ledger_profile from here
    os.environ["AS_SLOT"] = str(AS_SLOT)
    ns = dict(os=os, re=re, ast=ast, string=string, ENC=tiktoken.encoding_for_model("gpt-4o-mini"),
              ATOM=ATOM, FIT_Q=0.98, ATOM1=0, CHUNK=CHUNK, MSCALE=MSCALE, AUTOSTRUCT=AUTOSTRUCT, HYBRID=0,
              HDETECT=HDETECT, LPROFILE=0, PROFILE=0, PROFILE_INFO={}, HDET_INFO={})
    for s in _sources(PF29, CUT_CONSTS):
        exec(s, ns)
    for s in _sources(PF29, CUT_FUNCS):
        exec(s, ns)
    ns["PROFILES"] = {"factconsolidation": ns["_fc_profile"]}
    return ns


def harness_cut(df, row, **settings):
    """(units, keys, parents, order) exactly as harness load_row returns them."""
    import contextlib
    import io
    ns = cut_namespace(**settings)
    with contextlib.redirect_stdout(io.StringIO()):
        r = ns["load_row"](df, row)
    return r[0], r[1], r[2], r[3]
