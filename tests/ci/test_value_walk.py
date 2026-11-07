"""A read of a numbered log follows what its lines name. Native mode only; the exact path and a log naming no subject are unchanged."""
import numpy as np
import pytest

import muster
from muster.index import Index, ValueLinks, plain_words
from muster.search import H, SLOT_CE_CUT, Scorer, value_walk

FAKE = muster.Models.fake(16)

FILLER = ["%s was born in the city of %s." % (a, b) for a, b in (
    ("Mira Ona", "Lund"), ("Pavel Rusk", "Oslo"), ("Hana Sol", "Bern"), ("Ivo Tamm", "Riga"), ("Lena Voss", "Graz"),
    ("Omar Wynn", "Kiel"), ("Rita Yule", "Pisa"), ("Sven Zorn", "Turin"), ("Tove Arden", "Porto"), ("Uma Blake", "Cork"),
    ("Vera Crane", "Ghent"), ("Wim Doyle", "Malmo"), ("Xena Ellis", "Bari"), ("Yuri Fane", "Bonn"), ("Zoe Grant", "Basel"),
    ("Adam Hale", "Nice"), ("Bea Irwin", "Lyon"), ("Carl Judd", "Split"), ("Dina Kerr", "Tartu"), ("Egon Lowe", "Delft"),
    ("Fay Marsh", "Leeds"), ("Gus Nolan", "Perth"), ("Hugo Orr", "Tampa"), ("Inga Pryce", "Reno"), ("Jon Quill", "Nome"),
)]


def numbered(*lines):
    return "\n".join("%d. %s" % (i + 1, x) for i, x in enumerate(lines))


def plain(*chain, updates=()):
    """The chain, then filler, then four updates so the log passes the numbered checks."""
    extra = [FILLER[i].replace(" was born in the city of ", " was born in the city of New ") for i in range(4)]
    return numbered(*(list(updates) + list(chain) + FILLER + extra))


def read(text, question, U=10):
    mem = muster.build(text, models=FAKE)
    assert mem.plan.branch == "numbered", mem.plan.trace
    return mem, mem.context(question, U=U).text


CHAIN = ["Steve Sax is associated with the sport of association football.",
         "association football was created in the country of Italy.",
         "The capital of Italy is Rome."]


def test_a_lower_case_value_leads_on():
    _, text = read(plain(*CHAIN), "What is the capital of the country where Steve Sax's sport was created?")
    for line in CHAIN:
        assert line in text


def test_a_possessive_in_the_question_and_in_a_line_names_the_same_subject():
    chain = ["The author of Categories is Aristotle.", "Aristotle's child is Nia Vale."]
    _, text = read(plain(*chain), "Who is the child of the author of Categories?")
    for line in chain:
        assert line in text
    _, text = read(plain(*chain), "Who is Aristotle's child?")
    assert chain[1] in text


def test_a_value_many_lines_share_still_leads_to_the_lines_about_it():
    many = ["%s is a citizen of Ruritania." % f.split(" was ")[0] for f in FILLER]
    chain = ["Ellen Pompeo is a citizen of Ruritania.", "The capital of Ruritania is Harrisville."]
    mem, text = read(numbered(*(chain + many + FILLER + [f.replace(" of ", " of New ") for f in FILLER[:8]])),
                     "What is the capital of the country Ellen Pompeo is a citizen of?")
    assert "Ruritania" in mem.index.hubs
    for line in chain:
        assert line in text


def test_a_new_value_replaces_the_old_line_on_the_walk_too():
    chain = ["Steve Sax is associated with the sport of baseball.", "baseball was created in the country of Italy."]
    old = "Steve Sax is associated with the sport of cricket."
    other = "Mia Hamm is associated with the sport of tennis."  # a relation counts once two subjects share it
    _, text = read(plain(*(chain + [other]), updates=[old]), "Where was the sport of Steve Sax created?")
    assert chain[0] in text and chain[1] in text and old not in text


def links_of(*lines):
    return ValueLinks(list(lines), {i: i for i in range(len(lines))})


def test_words_of_no_subject_lead_nowhere():
    links = links_of("it is raining in Oslo.", "the team is based in Bern.", "they are resting in Lund.")
    assert ("it",) not in links.slots_of and ("they",) not in links.slots_of
    assert ("the", "team") in links.slots_of
    assert links.subjects_in(plain_words("Is it raining?")) == []


def test_a_subject_of_too_many_slots_leads_nowhere():
    lines = ["Alice is a citizen of Place%d." % i for i in range(40)] + ["Bob%d is a citizen of Alice." % i for i in range(40)]
    assert ("alice",) not in links_of(*lines).slots_of


def test_a_line_does_not_lead_to_itself():
    links = links_of("Alice is married to Bob.", "Bob is married to Alice.", "Carol is married to Dan.", "Dan is married to Erin.")
    assert links.named[0] == [("bob",)]
    assert links.named[1] == [("alice",)]


def walked(lines, query, U=10, score=None):
    """The walk over `lines`, numbered from 0; an older line of the first slot makes the index structural."""
    units = ["An older line."] + list(lines)
    idx = Index(units, units, ["s0"] + ["s%d" % i for i in range(len(lines))], list(range(len(units))), has_order=True)
    assert idx.structural

    class Reranker:
        def score(self, q, docs):
            return [score(d) if score else 0.0 for d in docs]
    return [i - 1 for i in value_walk(idx, query, Scorer(query, units, reranker=Reranker()), U)[0]]


def test_each_hop_gets_its_best_line_before_any_hop_gets_its_second():
    lines = ["Alice is a citizen of Oslo.", "Alice is a friend of Bern.", "Oslo is located in the continent of Asia.",
             "Bern is located in the continent of Zed."]
    score = {lines[0]: 2.0, lines[1]: 1.0, lines[2]: 1.0, lines[3]: 2.0}.get
    assert walked(lines, "Where does Alice live?", score=score) == [0, 3, 1, 2]


def test_the_walk_stops_at_its_depth():
    names = ["Name%s" % chr(65 + i) for i in range(H + 3)]
    lines = ["%s is a friend of %s." % (a, b) for a, b in zip(names, names[1:])]
    assert walked(lines, "Who is a friend of %s?" % names[0]) == list(range(H))


def test_the_walk_scores_a_bounded_number_of_lines():
    lines = ["Hub%d is a friend of Name%d." % (i, j) for i in range(40) for j in range(40)]
    assert len(walked(lines, " ".join("Hub%d" % i for i in range(40)), U=10 ** 6)) <= SLOT_CE_CUT


def test_a_log_naming_no_subject_walks_nowhere():
    lines = ["%d %d" % (i, i) for i in range(30)]
    assert walked(lines, "Where?") == []


def test_a_question_naming_no_subject_reads_the_ranking_as_before():
    mem, _ = read(plain(*CHAIN), "Which one?")
    assert mem.search("Which one?").heads["walk"] == []


def test_exact_mode_does_not_walk():
    from muster.search import select
    text, q = plain(*CHAIN), "Who is Steve Sax?"
    mem = muster.build(text, muster.derive(text), models=FAKE)
    qv = np.asarray(FAKE.encoder.encode([q], normalize_embeddings=True), dtype=np.float32)[0]
    native = select(mem.index, mem.vecs, qv, q, Scorer(q, list(mem.cut.units), reranker=FAKE.reranker), U=10, policy="attprio_sess", exact=False)
    exact = select(mem.index, mem.vecs, qv, q, Scorer(q, list(mem.cut.units), reranker=FAKE.reranker), U=10, policy="attprio_sess", exact=True)
    assert native[2]["walk"] and "walk" not in exact[2]


def test_records_are_walked_on_their_bodies_not_their_metadata():
    lines = list(plain(*CHAIN).split("\n"))
    recs = [{"text": x, "meta": {"user": "Steve Sax"}} for x in lines]
    mem = muster.build(recs, models=FAKE)
    assert mem.plan.branch == "numbered"
    text = mem.context("What is the capital of the country where Steve Sax's sport was created?").text
    for line in CHAIN:
        assert line in text


@pytest.mark.parametrize("line", [
    "word " * 200000,
    " ".join("Name%d" % i for i in range(50000)),
    "a" * 1000000,
    "the " * 100000 + "is here",
    ". " * 200000,
])
def test_a_long_line_stays_cheap(line):
    import time
    lines = [line] + ["Alice%d is a friend of word." % i for i in range(200)] + ["word is located in the continent of Asia."]
    t0 = time.time()
    links = links_of(*lines)
    links.subjects_in(plain_words(line))
    assert time.time() - t0 < 10


def test_a_function_word_is_no_subject_and_a_long_name_is_one():
    long_name = "The Church of Jesus Christ of Latter-day Saints was founded by Lyndon Baines."
    links = links_of("With or Without You was performed by U2.", "And I Love Her was performed by The Beatles.", long_name,
                     "Nicholas Five is affiliated with the religion of The Church of Jesus Christ of Latter-day Saints.")
    assert ("with",) not in links.slots_of and ("and",) not in links.slots_of
    assert tuple("the church of jesus christ of latter day saints".split()) in links.slots_of
    assert links.named[3] == [tuple("the church of jesus christ of latter day saints".split())]


def test_the_slot_graph_answers_as_the_full_table_did():
    import collections
    text = plain(*CHAIN)
    idx = muster.build(text, muster.derive(text), models=FAKE).index
    ent_slots = collections.defaultdict(set)
    for s2, i2 in idx.newest_row.items():
        for e2 in (idx.r_ents[i2] - idx.hubs):
            ent_slots[e2].add(s2)
    cut = max(3, int(0.05 * max(len(idx.newest_row), 1)))
    table = collections.defaultdict(set)
    for ss2 in (v for v in ent_slots.values() if len(v) <= cut):
        for s2 in ss2:
            table[s2] |= ss2 - {s2}
    assert table
    for s2 in idx.newest_row:
        assert set(idx.adj_slot.get(s2, ())) == table.get(s2, set())
    assert idx.adj_slot.get("no such slot", ()) == ()


def test_an_entity_of_many_slots_answers_without_a_table():
    from muster.index import _SlotGraph
    slots = set(range(20000))
    graph = _SlotGraph({"Zone": slots})
    assert len(graph.get(7)) == 19999 and 7 not in graph.get(7)
    assert graph.get(30000, ()) == ()


QUESTION = "What is the capital of the country where Steve Sax's sport was created?"


def test_a_read_shows_the_chain_in_the_order_it_runs():
    text = muster.build(plain(*reversed(CHAIN)), models=FAKE).context(QUESTION).text
    shown = text.split("\n")
    assert [x.split("] ", 1)[1] for x in shown[:3]] == CHAIN
    assert [int(x[1:x.index("]")]) for x in shown[:3]] == [3, 2, 1]  # the serials stay as written
    rest = [int(x[1:x.index("]")]) for x in shown[3:]]
    assert rest == sorted(rest)


def test_a_plain_list_of_lines_keeps_serial_order():
    mem = muster.build(plain(*reversed(CHAIN)), models=FAKE)
    sel = mem.search(QUESTION)
    assert mem.render(sel).text != mem.render(list(sel.ids)).text
    serials = [int(x[1:x.index("]")]) for x in mem.render(list(sel.ids)).text.split("\n")]
    assert serials == sorted(serials)
    assert sorted(mem.render(sel).text.split("\n")) == sorted(mem.render(list(sel.ids)).text.split("\n"))


def test_a_question_naming_no_subject_keeps_serial_order():
    text = muster.build(plain(*CHAIN), models=FAKE).context("What happened?").text
    serials = [int(x[1:x.index("]")]) for x in text.split("\n")]
    assert serials == sorted(serials)


def test_a_numbered_store_under_another_policy_reads_as_before():
    from muster.search import select
    text = plain(*CHAIN)
    mem = muster.build(text, muster.derive(text), models=FAKE)
    qv = np.asarray(FAKE.encoder.encode([QUESTION], normalize_embeddings=True), dtype=np.float32)[0]
    heads = select(mem.index, mem.vecs, qv, QUESTION, Scorer(QUESTION, list(mem.cut.units), reranker=FAKE.reranker), U=10, policy="flrs", exact=False)[2]
    assert "first" not in heads and "walk" not in heads


def test_exact_mode_keeps_serial_order():
    from muster.search import select
    text = plain(*reversed(CHAIN))
    mem = muster.build(text, muster.derive(text), models=FAKE)
    qv = np.asarray(FAKE.encoder.encode([QUESTION], normalize_embeddings=True), dtype=np.float32)[0]
    heads = select(mem.index, mem.vecs, qv, QUESTION, Scorer(QUESTION, list(mem.cut.units), reranker=FAKE.reranker), U=10, policy="attprio_sess", exact=True)[2]
    assert "first" not in heads


def test_ties_in_serial_keep_walk_order():
    from muster.render import numbered as show
    assert show([1, 0], ["a", "b"], [5, 5]).ids == (1, 0)
    assert show([2, 0, 1], ["a", "b", "c"], [5, 6, 7], None, (2,)).ids == (2, 0, 1)
