"""C3 (cutting): the library's cut against the harness's own load_row, on synthetic stores with the edge cases."""
import random

import pytest

from muster.cut import cut
from muster.plan import Plan
from oracle import harness_cut

pd = pytest.importorskip("pandas")


def _df(context, source):
    return pd.DataFrame({"context": [context], "metadata": [{"source": source}],
                         "questions": [["q?"]], "answers": [[["a"]]]})


def _plan(atom, chunk, mscale, branch=None):
    return Plan(branch=branch or ("record" if atom == "blank" else "prose"), atom=atom, chunk=chunk, mscale=mscale,
                U=10, policy="flr", render="r4", mode="exact")


def _texts():
    rnd = random.Random(11)
    w = "the river Paris Maria engine report seven stone light delta Dr. Smith U.S. e.g. 3.5 kg".split()
    sent = lambda n: " ".join(rnd.choice(w) for _ in range(n)).capitalize() + rnd.choice([".", "!", "?", "..."])
    out = []
    out.append("\n\n".join("%d.%d user: %s" % (i, j, sent(rnd.randint(5, 60))) for i in range(30) for j in range(8)))
    out.append(" ".join(sent(rnd.randint(4, 50)) for _ in range(2500)))
    out.append("\n \n".join(sent(30) for _ in range(400)))                 # separator with a space inside
    out.append("\n\n".join([sent(20)] * 200))                              # duplicate texts across parents
    out.append("\n\n".join(sent(rnd.randint(1, 3000)) for _ in range(40)))   # oversize records
    out.append("CHAPTER I\n\n" + "\n\n".join(sent(80) for _ in range(300)))
    out.append(("Ünïcødé ✓ — 東京 " + sent(40) + "\n\n") * 150)
    return out


@pytest.mark.parametrize("atom,chunk,mscale", [("blank", 512, 4096), ("sentence", 1024, 8192), ("blank", 512, 2048),
                                               ("sentence", 512, 4096)])
@pytest.mark.parametrize("k", range(7))
def test_cut_equals_harness(atom, chunk, mscale, k):
    text = _texts()[k]
    units, keys, par, order = harness_cut(_df(text, "ruler_qa1_197K"), 0, ATOM=atom, CHUNK=chunk, MSCALE=mscale)
    c = cut(text, _plan(atom, chunk, mscale))
    assert list(c.units) == list(units)
    assert list(c.parent) == list(par)
    assert list(c.order) == list(order)


def test_numbered_equals_harness():
    rnd = random.Random(3)
    ents = ["Paris", "Maria Lopez", "The Beatles", "Dr. No", "United States", "Apple"]
    lines = ["%d. %s %s %s." % (i, rnd.choice(ents), rnd.choice(["is located in", "was created by", "plays for"]),
                                rnd.choice(ents)) for i in range(1, 600)]
    lines.insert(40, "an unnumbered line")
    text = "\n".join(lines)
    units, keys, par, order = harness_cut(_df(text, "factconsolidation_sh_262k"), 0, AUTOSTRUCT=1, AS_SLOT=5)
    c = cut(text, Plan(branch="numbered", atom=None, chunk=None, mscale=0, U=10, policy="attprio_sess",
                       render="numbered", mode="exact"))
    assert list(c.units) == list(units) and list(c.parent) == list(par) and list(c.order) == list(order)


def test_rounds_equal_harness():
    sess = []
    for s in range(6):
        turns = []
        for t in range(5):
            turns.append({"role": "user", "content": "question %d.%d about Paris" % (s, t)})
            if t != 3:
                turns.append({"role": "assistant", "content": "answer %d.%d" % (s, t)})
        sess += ["2023/05/%02d (Mon) 10:%02d" % (s + 1, s), turns]
    lit = repr(sess)
    df = pd.DataFrame({"context": [lit], "metadata": [{"source": "longmemeval_s*", "haystack_sessions": [[]],
                                                         "question_types": ["x"]}],
                       "questions": [["q?"]], "answers": [[["a"]]]})
    units, keys, par, order = harness_cut(df, 0)
    c = cut(lit, Plan(branch="rounds", atom=None, chunk=None, mscale=0, U=10, policy="flrs", render="flat", mode="exact"))
    assert list(c.units) == list(units) and list(c.keys) == list(keys) and list(c.parent) == list(par)


def test_native_slices_rebuild_the_atom():
    big = "東京 " * 3000
    c = cut(big, _plan("blank", 512, 0), exact=False)
    assert "".join(c.units) == big.strip("\n")
