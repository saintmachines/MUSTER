"""The paper's settings as frozen exact plans, generated from the released leg tables (tools/gen_presets.py)."""
import json
import os

from ..plan import Plan

_DATA = None


def _data():
    global _DATA
    if _DATA is None:
        _DATA = json.load(open(os.path.join(os.path.dirname(__file__), "paper.json")))
    return _DATA


def names():
    return tuple(sorted(_data()["presets"]))


def legs():
    """{leg tag: {preset, parq, rows, nq, ds, arm}} for the 24 final legs."""
    return dict(_data()["legs"])


def get(name_or_leg: str) -> Plan:
    """A preset by name ("beam.u20") or by the tag of a leg it stands for ("lz_u20_beam100k")."""
    d = _data()
    name = name_or_leg
    if name not in d["presets"]:
        leg = d["legs"].get(name_or_leg)
        if leg is None:
            raise KeyError("no preset or leg named %r; presets: %s" % (name_or_leg, ", ".join(names())))
        name = leg["preset"]
    return Plan.from_dict(d["presets"][name])


class _Paper(dict):
    def __missing__(self, k):
        raise KeyError(k)


PAPER = _Paper({n: get(n) for n in names()})
