"""Gate: phase 6 — BM25 label search, map projection, scale-stunt attach."""

import uuid

import numpy as np
from qdrant_edge import EdgeShard, Point, UpdateOperation

from fleetmemory.memory.core import MapRequest, ScaleStunt, SearchRequest, Teach
from fleetmemory.memory.store import shard_config


def teach(h, tid, concept, label):
    h.ingest(tid, h.geo.view(concept))
    h.send(Teach(tid=tid, epoch=1, label=label))


def test_search_finds_by_label_words(h):
    teach(h, 1, "mug", "red coffee mug")
    teach(h, 2, "pen", "blue pen")
    h.send(SearchRequest(text="coffee"))
    ev = h.last("search_results")
    assert [x["label"] for x in ev["hits"]] == ["red coffee mug"]
    h.send(SearchRequest(text=""))
    assert h.last("search_results")["hits"] == []


def test_map_projects_all_real_objects(h):
    for i, (concept, label) in enumerate([("mug", "mug"), ("pen", "pen"), ("cap", "cap")]):
        teach(h, i + 1, concept, label)
    h.send(MapRequest())
    ev = h.last("map")
    assert len(ev["points"]) == 3
    for p in ev["points"]:
        assert 0.0 <= p["x"] <= 1.0 and 0.0 <= p["y"] <= 1.0
        assert p["local"] is True


def test_scale_stunt_attach_detach(h, tmp_path):
    # a tiny synthetic "stunt" shard, same schema
    stunt = tmp_path / "scale"
    stunt.mkdir()
    shard = EdgeShard.create(str(stunt), shard_config(64))
    rng = np.random.default_rng(1)
    vs = rng.normal(size=(50, 3, 64)).astype(np.float32)
    vs /= np.linalg.norm(vs, axis=2, keepdims=True)
    shard.update(
        UpdateOperation.upsert_points(
            [
                Point(
                    id=str(uuid.uuid4()),
                    vector={"exemplars": vs[i].tolist()},
                    payload={"kind": "object", "label": f"obj-{i}", "synthetic": True},
                )
                for i in range(50)
            ]
        )
    )
    shard.close()

    teach(h, 1, "mug", "real mug")
    base = h.store.count()
    h.send(ScaleStunt(on=True, path=str(stunt)))
    assert h.last("scale")["on"] is True
    assert h.store.count() == base + 50  # HUD count includes the stunt shard

    # recognition still finds the real object, now over the bigger fanout
    h.ingest(9, h.geo.view("mug", 0.9))
    assert h.last("track_update")["label"] == "real mug"

    # inventory and map must NOT show synthetics
    from fleetmemory.memory.core import InventoryRequest, MapRequest

    h.send(InventoryRequest())
    assert len(h.last("inventory")["items"]) == 1
    h.send(MapRequest())
    assert len(h.last("map")["points"]) == 1

    h.send(ScaleStunt(on=False))
    assert h.store.count() == base


def test_scale_stunt_missing_shard_is_graceful(h):
    h.send(ScaleStunt(on=True, path="does/not/exist"))
    assert "demo-scale" in h.last("error")["message"]
