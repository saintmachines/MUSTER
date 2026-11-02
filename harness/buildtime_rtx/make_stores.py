#!/usr/bin/env python3
"""Writes data/ of this folder and fetches the model: the three store texts cut from the reported legs' parquets, the two
tiktoken encodings, and the bge-m3 files at the pinned revision. Each is checked against stores.json or expected.json, and any
difference stops here. Needs the network once and no key. Nothing here is timed.
The parquets come from ../beam/build_parquet.py, build_parquet_1m.py and build_parquet_10m.py (in MUSTER_HOME/beam, default
../beam). --no-model leaves the model out, enough for units_check.py.
Usage: python -B make_stores.py [--beam DIR] [--data DIR] [--no-model]"""
import argparse, hashlib, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--beam", default=os.path.join(os.environ.get("MUSTER_HOME", os.path.dirname(HERE)), "beam"))
ap.add_argument("--data", default=os.path.join(HERE, "data"))
ap.add_argument("--no-model", action="store_true")
A = ap.parse_args()
S = json.load(open(os.path.join(HERE, "stores.json")))
EXP = json.load(open(os.path.join(HERE, "expected.json")))
CACHE = os.path.join(os.path.abspath(A.data), "tiktoken_cache")
os.makedirs(CACHE, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = CACHE
import pandas as pd
import tiktoken


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


for c, s in S["stores"].items():
    pq = os.path.join(A.beam, os.path.basename(s["parquet"]["path"]))
    text = str(pd.read_parquet(pq, columns=["context"])["context"].iloc[s["parquet"]["row"]])   # as load_row reads it
    raw = text.encode("utf-8")
    assert hashlib.sha256(raw).hexdigest() == s["sha256"] and len(raw) == s["bytes"] and len(text) == s["chars"], ("store differs", c, pq)
    with open(os.path.join(A.data, s["file"]), "wb") as fh:
        fh.write(raw)
    print("%-9s %-22s %d bytes, sha256 equal" % (c, s["file"], len(raw)), flush=True)

for enc in ("cl100k_base", "o200k_base"):
    tiktoken.get_encoding(enc)
for name, rec in S["tiktoken_cache"].items():
    p = os.path.join(CACHE, name)
    assert sha256(p) == rec["sha256"] and os.path.getsize(p) == rec["bytes"], ("tiktoken file differs", name)
print("tiktoken encodings: %d files, sha256 equal" % len(S["tiktoken_cache"]), flush=True)

if not A.no_model:
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"   # a public model, fetched without any token
    from huggingface_hub import snapshot_download
    M = EXP["model"]
    snap = snapshot_download(M["id"], revision=M["revision"], allow_patterns=sorted(M["files"]))
    for rel, rec in M["files"].items():
        p = os.path.join(snap, rel)
        assert sha256(p) == rec["sha256"] and os.path.getsize(p) == rec["bytes"], ("model file differs", rel)
    print("%s at %s: %d files, sha256 equal" % (M["id"], M["revision"][:7], len(M["files"])), flush=True)
print("data ready")
