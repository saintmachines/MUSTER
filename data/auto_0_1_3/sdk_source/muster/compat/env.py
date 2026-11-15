"""The environment exact mode needs, and a record of the one a run used.

Exact mode iterates sets the way the paper's harness did, which is reproducible only with string hashing fixed
(PYTHONHASHSEED=0 at interpreter start; `python -I` and `-E` ignore the variable). The published equality was
measured with the versions in EXACT_VERSIONS on CPython 3.12.13.
"""
import importlib.metadata
import platform
import sys

from ..errors import EnvironmentRefused

EXACT_PYTHON = "3.12.13"
EXACT_VERSIONS = {"tiktoken": "0.13.0", "numpy": "2.4.4", "torch": "2.11.0", "sentence-transformers": "5.3.0",
                  "transformers": "5.5.0", "tokenizers": "0.22.2", "nltk": "3.10.0", "rank-bm25": "0.2.2"}


def versions():
    out = {"python": platform.python_version(), "hash_randomization": sys.flags.hash_randomization}
    for k in EXACT_VERSIONS:
        try:
            out[k] = importlib.metadata.version(k)
        except importlib.metadata.PackageNotFoundError:
            out[k] = None
    return out


def check_environment(*, strict_versions=False):
    """Raise EnvironmentRefused if exact mode cannot reproduce the paper here; return the mismatches otherwise."""
    if sys.flags.hash_randomization:
        raise EnvironmentRefused("exact mode needs string hashing fixed: start Python as "
                                 "`PYTHONHASHSEED=0 python ...` (not with -I or -E)")
    v = versions()
    bad = {k: (v.get(k), want) for k, want in EXACT_VERSIONS.items() if v.get(k) != want}
    if v["python"] != EXACT_PYTHON:
        bad["python"] = (v["python"], EXACT_PYTHON)
    if bad and strict_versions:
        raise EnvironmentRefused("exact mode was verified with other versions: %s" % bad)
    return bad
