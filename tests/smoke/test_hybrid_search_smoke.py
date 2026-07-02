"""Hybrid search smoke: real miniCOIL + dense models (local only, not the gate).

Verifies the two prefetch legs + RRF end to end: exact words hit via sparse,
synonyms hit via dense, junk queries stay empty, and the schema round-trips.
"""

import numpy as np
import pytest

from fleetmemory.memory.labels import LabelEmbedder
from fleetmemory.memory.store import Store, new_id

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = Store(tmp_path_factory.mktemp("hybrid") / "d", dim=32, label_embedder=LabelEmbedder())
    rng = np.random.default_rng(0)

    def vec():
        v = rng.normal(size=32).astype(np.float32)
        return v / np.linalg.norm(v)

    for label in ["coffee mug", "water bottle", "conference badge", "laptop charger"]:
        s.upsert_object(new_id(), label, [vec()], [{"view_id": "v", "human": True}])
    yield s
    s.close()


def labels_of(results):
    return [c.label for c, _similar in results]


def test_exact_words_hit_via_sparse(store):
    results, ms = store.search_text("mug")
    assert labels_of(results)[0] == "coffee mug"
    assert ms > 0


def test_synonyms_hit_via_dense_leg(store):
    results, _ = store.search_text("cup")  # no shared token with "coffee mug"
    assert "coffee mug" in labels_of(results)


def test_semantic_neighbors_rank_sensibly(store):
    results, _ = store.search_text("drink container")
    top2 = labels_of(results)[:2]
    assert "water bottle" in top2 or "coffee mug" in top2


def test_unrelated_query_returns_nothing(store):
    results, _ = store.search_text("helicopter")
    assert not [c.label for c, similar in results if not similar]


def test_partial_word_lands_via_substring(store):
    results, _ = store.search_text("charg")
    assert "laptop charger" in labels_of(results)
