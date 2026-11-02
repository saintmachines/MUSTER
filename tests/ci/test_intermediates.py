"""C3 (renders): the lifted bgroup and seg3 intermediates against the harness's own functions, on random inputs with
shared parents, shared entities, hubs and duplicate texts; and the exact R4 chain against render_rule's recovery."""
import random

import pytest

from muster.compat.intermediates import rank_pos, render_bgroup, render_seg3
from muster.compat.recover import recover_ids
from oracle import PF29, _sources


def _harness():
    import math
    ns = dict(math=math, HMARK="off", _RG={})
    for s in _sources(PF29, ["_rank_pos", "_render_bgroup", "_render_hseg"]):
        exec(s, ns)
    return ns


def _case(seed):
    rnd = random.Random(seed)
    n = rnd.randint(5, 120)
    names = ["Ann", "Ben", "Kyoto", "Atlas", "Nova", "Paris", "Mia", "Bo", "Lisbon", "Dock"]
    parent = []
    p = 0
    for i in range(n):
        if rnd.random() < 0.15:
            p += 1
        parent.append("m%d" % p)
    ents = [set(rnd.sample(names, rnd.randint(0, 4))) for _ in range(n)]
    hubs = set(rnd.sample(names, rnd.randint(0, 2)))
    # many duplicates; every unit longer than the 80-character recovery key, as every unit of a real store is
    texts = [("unit %d %s " % (i % max(3, n // 3), " ".join(sorted(ents[i])))) * 12 for i in range(n)]
    fed = sorted(rnd.sample(range(n), rnd.randint(1, min(n, 45))))
    order = list(range(n))
    rnd.shuffle(order)
    return n, parent, ents, hubs, texts, fed, order


@pytest.mark.parametrize("seed", range(300))
def test_bgroup_and_seg3_equal_harness(seed):
    H = _harness()
    n, parent, ents, hubs, texts, fed, order = _case(seed)
    txt = lambda i: texts[i]
    pos = rank_pos(order)
    assert pos == H["_rank_pos"](order)
    for cap in (10, 4, 20):
        assert render_bgroup(fed, txt, pos, len(pos) + 1, parent, ents, hubs, cap) == \
            H["_render_bgroup"](fed, txt, pos, len(pos) + 1, parent, ents, hubs, cap)
    for cap in (-10, -4, 10, 6):
        assert render_seg3(fed, txt, parent, ents, hubs, cap) == \
            H["_render_hseg"](fed, txt, pos, len(pos) + 1, parent, ents, hubs, cap, doc_first=True)


@pytest.mark.parametrize("seed", range(100))
def test_recovery_returns_a_valid_id_set(seed):
    # render_rule.py's recovery keys units by their first 80 characters, so it needs units longer than that
    n, parent, ents, hubs, texts, fed, order = _case(seed)
    inter = render_seg3(fed, lambda i: texts[i], parent, ents, hubs, -10)
    ids = recover_ids(inter, texts, parent)
    assert len(ids) == len(fed) and len(set(ids)) == len(ids)
    assert sorted(texts[i] for i in ids) == sorted(texts[i] for i in fed)     # same texts, possibly other copies
