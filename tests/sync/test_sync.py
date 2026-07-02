"""§6 sync tests against a real Qdrant server: schema round-trip, curated push,
partial-snapshot pull, the id-present dedup rule, and the two-device scenario.
The §9.3 spike verified the docs' blanket pattern; the curated variant is OUR
design delta and gets asserted here explicitly."""

import time


def test_pull_delivers_another_devices_teachings(fleet):
    a, b = fleet("unit-a"), fleet("unit-b")
    a.sync.client.ensure_collection()
    # A teaches + pushes
    oid = a.teach_direct("stapler", "stapler")
    a.sync.push([oid])
    a.wait_event("push_done")
    # B pulls -> recognizes what A taught (MAX_SIM intact through the snapshot)
    b.sync.pull_once()
    b.wait_event("pull_applied")
    res = b.recognize("stapler")
    assert res.candidates and res.candidates[0].label == "stapler"
    assert not res.candidates[0].from_mutable  # served from the fleet mirror


def test_dedup_pushed_leaves_unpushed_survives(fleet):
    a = fleet("unit-a")
    a.sync.client.ensure_collection()
    pushed = a.teach_direct("badge", "badge")
    local = a.teach_direct("secret prototype", "proto")
    a.sync.push([pushed])
    a.wait_event("push_done")

    a.sync.pull_once()
    a.wait_event("pull_applied")
    # pushed object left mutable (id present in the fresh mirror)...
    assert a._core_call(lambda: a.store.get_object(pushed)) is None
    # ...but is still recognized via the mirror
    res = a.recognize("badge")
    assert res.candidates and res.candidates[0].label == "badge"
    # unpushed local object SURVIVED the pull (the §3.3 rule, not bare timestamps)
    assert a._core_call(lambda: a.store.get_object(local)) is not None
    res = a.recognize("proto")
    assert res.candidates[0].label == "secret prototype"
    assert res.candidates[0].from_mutable


def test_two_devices_same_item_folds_one_fleet_point(fleet):
    """Instance model: the SAME physical mug seen by two devices folds into one
    fleet point (same name + it looks like it)."""
    a, b = fleet("unit-a"), fleet("unit-b")
    a.sync.client.ensure_collection()
    a.sync.push([a.teach_direct("coffee mug", "mug", n_views=3)])
    a.wait_event("push_done")

    b.sync.pull_once()
    b.wait_event("pull_applied")
    b_oid = b.teach_direct("coffee mug", "mug", n_views=3)  # same mug, B's angles
    b.sync.push([b_oid])
    b.wait_event("push_done")

    recs, _ = b.client.client.scroll(b.client.collection, limit=100, with_payload=True)
    mugs = [r for r in recs if (r.payload or {}).get("label") == "coffee mug"]
    assert len(mugs) == 1, "the same item from two devices must fold into ONE fleet point"
    assert len(mugs[0].payload["views"]) >= 4  # carries views of both devices

    # B's local copy was rewritten under the fleet id; next pull dedups it away
    b.sync.pull_once()
    time.sleep(0.3)
    assert b._core_call(lambda: b.store.get_object(b_oid)) is None
    fleet_id = str(mugs[0].id)
    local_copy = b._core_call(lambda: b.store.get_object(fleet_id))
    assert local_copy is None  # deduped: mirror now carries it
    res = b.recognize("mug")
    assert res.candidates and res.candidates[0].label == "coffee mug"


def test_different_items_same_name_stay_separate_fleet_points(fleet):
    """Two DIFFERENT mugs both called "coffee mug" are two fleet points —
    a shared display name must not melt distinct items together."""
    a, b = fleet("unit-a"), fleet("unit-b")
    a.sync.client.ensure_collection()
    a.sync.push([a.teach_direct("coffee mug", "mug-red", n_views=3)])
    a.wait_event("push_done")
    b.sync.push([b.teach_direct("coffee mug", "mug-blue", n_views=3)])
    b.wait_event("push_done")
    recs, _ = b.client.client.scroll(b.client.collection, limit=100, with_payload=True)
    mugs = [r for r in recs if (r.payload or {}).get("label") == "coffee mug"]
    assert len(mugs) == 2


def test_blocklist_never_pushed(fleet):
    a = fleet("unit-a")
    a.sync.client.ensure_collection()
    from fleetmemory.memory.core import IgnoreTrack, Ingest

    a.core.submit(
        Ingest(tid=1, epoch=1, vec=a.geo.view("wall-art"), thumb_jpeg=None, quality=1.0, t=1.0)
    )
    a.core.submit(IgnoreTrack(tid=1, epoch=1))
    ignored = a._core_call(
        lambda: [pid for pid, _, _ in a.store.scroll_objects(kind="ignored", mutable_only=True)]
    )
    assert ignored
    a.sync.push(ignored)  # curation can never leak the blocklist
    a.wait_event("push_done")
    recs, _ = a.client.client.scroll(a.client.collection, limit=10)
    assert recs == []


def test_broken_snapshot_rebuilds_mirror_and_recovers(fleet, tmp_path):
    """A damaged pull must not wedge the device: the mirror is disposable —
    rebuild it empty, and the next pull re-seeds it from the fleet."""
    from fleetmemory.memory.core import ApplyPartialSnapshot

    a = fleet("unit-a")
    a.sync.client.ensure_collection()
    a.sync.push([a.teach_direct("pen", "pen")])
    a.wait_event("push_done")

    garbage = tmp_path / "garbage.snapshot"
    garbage.write_bytes(b"this is not a snapshot tar")
    a.core.submit(ApplyPartialSnapshot(path=str(garbage)))
    a.wait_event("fleet_error")

    a.sync.pull_once()  # fresh (rebuilt) mirror re-seeds from the fleet
    a.wait_event("pull_applied")
    res = a.recognize("pen")
    assert res.candidates and res.candidates[0].label == "pen"


def test_pull_now_is_idempotent(fleet):
    a = fleet("unit-a")
    a.sync.client.ensure_collection()
    a.sync.push([a.teach_direct("pen", "pen")])
    a.wait_event("push_done")
    a.sync.pull_once()
    a.wait_event("pull_applied")
    n1 = a._core_call(a.store.count)
    a.sync.pull_once()
    time.sleep(0.2)
    n2 = a._core_call(a.store.count)
    assert n1 == n2  # repeated pulls don't duplicate or destroy anything


def test_hydrated_confirm_and_rename_fold_by_id(fleet):
    """Same-id ALWAYS folds at push time, even after a local rename (the
    label lookup would miss it and a plain upsert would clobber views other
    units folded in meanwhile). Also covers copy-on-write hydration end to
    end: pull -> confirm adds a view locally -> push folds it back."""
    from fleetmemory.memory.core import Confirm, Ingest, Rename

    a = fleet("unit-a")
    b = fleet("unit-b")
    a.sync.client.ensure_collection()
    oid = a.teach_direct("badge", "badge", n_views=2)
    a.sync.push([oid])
    a.wait_event("push_done")

    # unit B pulls, confirms the fleet object live -> hydrates + adds a view
    b.sync.pull_once()
    b.wait_event("pull_applied")
    b.core.submit(
        Ingest(tid=1, epoch=1, vec=b.geo.view("badge", 0.9), thumb_jpeg=None, quality=1.0, t=1.0)
    )
    b.core.submit(Confirm(tid=1, epoch=1, object_id=oid))
    b.wait_event("object_updated")
    payload, rows = b._core_call(lambda: b.store.get_object(oid))
    assert len(rows) == 3 and "t_sync" not in payload  # hydrated, dirty
    b.sync.push([oid])
    b.wait_event("push_done")

    # unit A renames its copy (dirty again) and pushes: must fold by id
    a.core.submit(Rename(object_id=oid, label="my badge"))
    a.sync.push([oid])
    t0 = time.time()  # wait_event returns the FIRST match; wait for push #2
    while sum(e["type"] == "push_done" for e in a.events) < 2:
        assert time.time() - t0 < 15, "second push_done never arrived"
        time.sleep(0.05)

    recs = a.client.client.retrieve(a.client.collection, ids=[oid], with_payload=True)
    assert len(recs) == 1
    pl = recs[0].payload or {}
    assert pl["label"] == "my badge"  # rename carried into the fold
    assert len(pl.get("views") or []) == 3  # B's confirmed view NOT clobbered
