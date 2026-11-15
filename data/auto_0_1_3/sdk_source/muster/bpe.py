"""Byte-pair encodings.

cl100k_base counts tokens for the detection statistics; o200k_base cuts the store and counts unit sizes, as the
paper's harness does (harness/lme_pf29.py:259, scripts/derive.py:134).

The encodings come from tiktoken. If `muster fetch` has put the BPE files in muster's asset directory, they are read
from there, each checked against the hash tiktoken itself pins, so a run never depends on tiktoken's download cache
(which lives under the system temp directory). Otherwise tiktoken loads them in its usual way.
"""
import base64
import hashlib
import os

from .errors import AssetMissing

NAMES = ("cl100k_base", "o200k_base")
_ENC = {}


def asset_dir():
    """Where `muster fetch` puts tokenizer and sentence-splitter files ($MUSTER_ASSETS, else a per-user data dir)."""
    d = os.environ.get("MUSTER_ASSETS")
    if d:
        return d
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "muster")


def _bpe_path(name):
    return os.path.join(asset_dir(), "bpe", name + ".tiktoken")


def _constructor(name, ranks_for):
    """Run tiktoken's own constructor for `name` with its file loader replaced by `ranks_for(url, sha256)`.

    The pattern string and the special tokens therefore come from the installed tiktoken, not from a copy."""
    import tiktoken_ext.openai_public as op
    if name not in op.ENCODING_CONSTRUCTORS:
        raise AssetMissing("tiktoken has no encoding named %r" % name)
    orig = op.load_tiktoken_bpe
    op.load_tiktoken_bpe = lambda url, expected_hash=None: ranks_for(url, expected_hash)
    try:
        return op.ENCODING_CONSTRUCTORS[name]()
    finally:
        op.load_tiktoken_bpe = orig


def source(name):
    """(url, sha256) of the BPE file tiktoken's constructor names."""
    seen = {}

    def ranks_for(url, sha):
        seen["url"], seen["sha256"] = url, sha
        return {}
    _constructor(name, ranks_for)
    return seen["url"], seen["sha256"]


def _from_file(name, path):
    import tiktoken
    data = open(path, "rb").read()

    def ranks_for(url, sha):
        if sha and hashlib.sha256(data).hexdigest() != sha:
            raise AssetMissing("%s does not match the sha256 tiktoken pins (%s); run `muster fetch` again" % (path, sha))
        return {base64.b64decode(tok): int(rank) for tok, rank in (ln.split() for ln in data.splitlines() if ln)}
    return tiktoken.Encoding(**_constructor(name, ranks_for))


def encoding(name):
    """The tiktoken Encoding `name`, from muster's asset directory when it is there."""
    enc = _ENC.get(name)
    if enc is None:
        p = _bpe_path(name)
        if os.path.exists(p):
            enc = _from_file(name, p)
        else:
            import tiktoken
            enc = tiktoken.get_encoding(name)
        _ENC[name] = enc
    return enc


def o200k():
    return encoding("o200k_base")


def cl100k():
    return encoding("cl100k_base")


# The three ways the harness encodes. They differ only on text that contains special-token strings.
def count(enc, text):
    """Token count with special-token strings read as plain text (lme_pf29.py:1127-1128, derive.py:72)."""
    return len(enc.encode(text, disallowed_special=()))


def encode_pack(enc, text):
    """What the packer encodes with (lme_pf29.py:1273): only <|endoftext|> may appear; any other special string raises."""
    return enc.encode(text, allowed_special={"<|endoftext|>"})


def encode_label(enc, text):
    """tiktoken's default mode, used for parent labels (lme_pf29.py:2237): any special-token string raises."""
    return enc.encode(text)


def fetch(names=NAMES, *, quiet=False):
    """Download the BPE files into the asset directory and check each against tiktoken's pinned hash."""
    import urllib.request
    out = []
    for name in names:
        url, sha = source(name)
        dest = _bpe_path(name)
        if os.path.exists(dest) and hashlib.sha256(open(dest, "rb").read()).hexdigest() == sha:
            out.append((name, dest, "present"))
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        data = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != sha:
            raise AssetMissing("downloaded %s does not match its pinned sha256" % url)
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
        _ENC.pop(name, None)
        out.append((name, dest, "fetched"))
        if not quiet:
            print("fetched %s -> %s" % (name, dest))
    return out
