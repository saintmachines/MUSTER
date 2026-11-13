"""Source shapes choose recipes; explicit published plans retain their policy."""
import json
import math
import re

import numpy as np
import pytest

import muster
from muster.models import Models
from muster.serve import Service


@pytest.mark.parametrize("text,branch,policy,style", [
    ("first record\n\nsecond record", "record", "flrs", "flat"),
    ("A long uninterrupted narrative. " * 1800, "prose", "flr", "r4"),
], ids=["record", "prose"])
def test_nonempty_derived_plan_uses_its_source_recipe(text, branch, policy, style):
    plan = muster.derive(text)
    assert (plan.branch, plan.policy, plan.render) == (branch, policy, style)
    restored = muster.Plan.from_dict(json.loads(plan.to_json()))
    assert restored == plan and restored.plan_id == plan.plan_id


@pytest.mark.parametrize("mode", ["native", "exact"])
def test_collection_record_policy_and_historical_exact_statistics_path(mode):
    plan = muster.derive_collection(["one record\n\nanother record"] * 3, mode=mode)
    assert plan.branch == "record" and plan.policy == "flrs"
    assert plan.render == ("r4" if mode == "exact" else "flat")


def test_numbered_and_empty_branches_keep_their_existing_policies():
    numbered = muster.derive("1. Atlas is owned by Ann.\n2. Atlas is owned by Ben.")
    assert numbered.policy == "attprio_sess" and numbered.supersede == "numbered"
    assert muster.derive("").policy == "flr"


class _Encoder:
    name, revision = "fixed-policy-fixture", "1"

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            match = re.search(r"\bitem_(\d+)\b", text)
            if match is None:
                vectors.append([1.0, 0.0])
            else:
                i = int(match[1])
                score = -1.0 if i == 1 else 0.99 - 0.003 * i
                vectors.append([score, math.sqrt(max(0.0, 1.0 - score * score))])
        return np.asarray(vectors, dtype=np.float32)


class _Reranker:
    name, revision = "fixed-policy-fixture", "1"

    def score(self, query, texts):
        scores = []
        for text in texts:
            i = int(re.search(r"\bitem_(\d+)\b", text)[1])
            scores.append(200.0 if i == 1 else 100.0 if i == 0 else 50.0 - i / 100.0)
        return scores


def test_derived_search_can_retrieve_a_sibling_outside_the_dense_candidates():
    # One intact record per unit. Unit 1 is the best reranker match but outside
    # the top-200 dense candidates and the keyword floor. Only the sibling pool
    # exposes it to selection beside unit 0; both runs use identical scores.
    text = "\n\n".join("item_%d %s %s" %
                       (i, "harbor" if i != 1 and i <= 20 else "remote", "filler " * 300)
                       for i in range(220))
    models = Models(encoder=_Encoder(), reranker=_Reranker())
    derived = muster.build(text, models=models)
    legacy = muster.build(text, derived.plan.with_(policy="flr", source="custom", frozen=True), models=models)
    assert len(derived.units) == 220 and derived.units == legacy.units
    assert derived.parent[0] == derived.parent[1]
    current = derived.search("harbor")
    previous = legacy.search("harbor")
    assert 1 not in current.heads["ce_rank"] and 1 not in previous.ids
    assert 1 in current.ids and "item_1 " in derived.render(current).text
    assert "item_1 " not in legacy.render(previous).text
    assert current.ids != previous.ids and current.floors == previous.floors == (4, 3, 3)


def test_explicit_serialized_flr_stays_flr_after_store_reopen(tmp_path):
    plan = muster.derive("A saved record.").with_(policy="flr", source="custom", frozen=True)
    saved = plan.to_json()
    restored = muster.Plan.from_dict(json.loads(saved))
    assert restored.to_json() == saved and restored.policy == "flr"
    path = tmp_path / "explicit.muster"
    for add in (True, False):
        with muster.Store.open(path, plan=restored, models=Models.fake(16)) as store:
            if add:
                store.add({"text": "The spare key is in the blue drawer."})
            assert store.head().plan == restored
            assert store.search("spare key").policy == "flr"


def test_published_presets_keep_their_recorded_policies():
    for name in ("beam.u20", "beam.u40", "locomo.u10", "locomo.u20", "locomo.u5"):
        plan = muster.presets.get(name)
        assert plan.frozen and plan.mode == "exact" and plan.policy == "flr"
    assert muster.presets.get("mab.A").policy == "flrs"
    assert muster.presets.get("mab.C").policy == "attprio_sess"


def test_new_store_service_exposes_the_derived_policy(tmp_path):
    with muster.Store.open(tmp_path / "new.muster", models=Models.fake(16)) as store:
        service = Service(store)
        service.add([{"text": "The spare key is in the blue drawer."}])
        assert service.info()["policy"] == "flrs"
        assert store.search("spare key").policy == "flrs"
