"""Shared constants and the per-store configuration of the BEAM conversation-0 build-time measurement (no torch, no
network). The stores are conversation 0 of BEAM 100K, 1M and 10M, the row 0 that the reported legs lz_u20_beam{100k,1m,10m}
and lz_u40_beam{100k,1m,10m} (len_legs.tsv lines 19 to 24) index. stores.json holds, per store, the text file and its
sha256, the metadata field load_row reads (source) and the leg's env (len_legs.json). make_stores.py writes the text files
from the legs' parquets."""
import hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BEAMBT_DATA") or os.path.join(HERE, "data")   # written by make_stores.py

# the order of the build table
ORDER = ["beam100k", "beam1m", "beam10m"]
NAME = {"beam100k": "BEAM 100K", "beam1m": "BEAM 1M", "beam10m": "BEAM 10M"}


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def stores():
    """stores.json of this folder."""
    return json.load(open(os.path.join(HERE, "stores.json")))


def store_text(st, S=None):
    """The store as load_row reads it, str(df["context"].iloc[0]) (lme_pf29.py:2215). The file holds exactly those
    characters as UTF-8, so the text is the file's bytes decoded as UTF-8, nothing else (a text-mode read would also
    translate line endings; the stores have none to translate, but the bytes are decoded as they are)."""
    S = S or stores()
    s = S["stores"][st]
    raw = open(os.path.join(DATA, s["file"]), "rb").read()
    assert hashlib.sha256(raw).hexdigest() == s["sha256"] and len(raw) == s["bytes"], ("store bytes differ", st)
    txt = raw.decode("utf-8")
    assert len(txt) == s["chars"], ("store characters differ", st)
    return txt


def store_df(st, S=None):
    """The one-row DataFrame that units_row (load_row) and derive's stats read for conversation 0: column context
    (the text above) and column metadata (the parquet row's metadata fields the build path reads, source and conv_id;
    norm_dataset maps source beam_10m to "beam", lme_pf29.py:2026-2027 and 1294-1305)."""
    import pandas as pd
    S = S or stores()
    return pd.DataFrame({"context": [store_text(st, S)], "metadata": [dict(S["stores"][st]["metadata"])]})


def register(REG, S=None):
    """The three stores as cells of cells_registry.CELLS, in memory only: (key, [0], 20 questions, "judge_beam",
    the protocol chunk of the registry's BEAM cell, CELLS["beam"][4] = 4096). derive() reads them from there
    (derive_config.py:68). The key names the store's DataFrame in ship_derive._DF."""
    S = S or stores()
    prot = REG.CELLS["beam"][4]
    for st in ORDER:
        s = S["stores"][st]
        assert s["prot"] == prot, (st, s["prot"], prot)
        REG.CELLS[st] = ("store:" + st, [0], 20, "judge_beam", prot)
    return {st: REG.CELLS[st] for st in ORDER}


def shipped_env(S=None):
    """{store: env} of the store's reported leg (lz_u20_beam*; the lz_u40 leg differs only in U, which the build path
    never reads), from stores.json."""
    S = S or stores()
    env = {}
    for st in ORDER:
        e = dict(S["stores"][st]["env"])
        assert e.get("ATOM") == "blank", ("BEAM legs pack blank-line atoms; no sentence splitter is needed", st)
        env[st] = e
    return env


def digest_update(h, row, texts):
    """Fold one row of a store into a sha256: b'ROW', the row index and the text count (8 bytes each,
    little endian), then per text in order its UTF-8 length (8 bytes, little endian) and its UTF-8 bytes."""
    h.update(b"ROW" + int(row).to_bytes(8, "little") + len(texts).to_bytes(8, "little"))
    for t in texts:
        b = str(t).encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)


def configure(S, env):
    """Set ship_lme's module switches to the leg's env exactly as lme_pf29.py reads them at import (lines 832, 836,
    849-850, 858, 864, 1161; AS_SLOT is read at call time, line 2081). Returns the encoder's max_seq_length
    (EMB_MAXLEN, line 2276, default 448)."""
    S.CHUNK = int(env.get("CHUNK", "4096"))
    S.MSCALE = int(env.get("MSCALE", "0") or 0)
    S.HYBRID = int(env.get("HYBRID", "0") or 0)
    S.AUTOSTRUCT = int(env.get("AUTOSTRUCT", "0") or 0)
    S.FORCE_ORDER = int(env.get("FORCE_ORDER", "0") or 0)
    S.ATOM = env.get("ATOM", "sentence")
    S.FIT_Q = float(env.get("FIT_Q", "0.98"))
    S.ATOM1 = int(env.get("ATOM1", "0") or 0)
    S.HDETECT = int(env.get("HDETECT", "0") or 0)
    os.environ["AS_SLOT"] = env.get("AS_SLOT", "1")
    os.environ["HUB"] = env.get("HUB", "rel")
    assert not S.MSCALE or S.MSCALE > S.CHUNK
    assert S.ATOM in ("sentence", "blank", "newline", "auto")
    assert int(env.get("PROFILE", "0") or 0) == 0 and int(env.get("LPROFILE", "0") or 0) == 0
    assert int(env.get("HBM", "0") or 0) == 0 and int(env.get("SPARSEX", "0") or 0) == 0 and int(env.get("SPH", "0") or 0) == 0
    assert env.get("BM25LC") == "1" and S.BM25LC and not S.BM25STEM
    assert not env.get("MIRROR_BGE") and not env.get("EMB_TRUNCDIM") and not env.get("EMB_ATTN")
    return int(env.get("EMB_MAXLEN", "448") or 448)
