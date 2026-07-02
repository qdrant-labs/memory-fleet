"""§6 gate: two-shard merge dedup, fleet-object semantics, dedup-after-pull."""

from fleetmemory.memory.core import Reject, Teach
from fleetmemory.memory.store import new_id


def test_fleet_object_recognized_from_immutable(hf):
    fid = new_id()
    hf.seed_immutable(fid, "stapler", [hf.geo.view("stapler", 0.98)])
    hf.ingest(1, hf.geo.view("stapler", 0.9))
    ev = hf.last("track_update")
    assert (ev["state"], ev["label"]) == ("recognized", "stapler")


def test_two_shard_dedup_mutable_wins(hf):
    """Same point id in both shards (pushed, then locally re-taught before pull dedup):
    recognition must return ONE candidate, the mutable one."""
    pid = new_id()
    rows = [hf.geo.view("badge", 0.98)]
    hf.seed_immutable(pid, "badge (fleet)", rows)
    hf.store.upsert_object(pid, "badge", rows, [{"view_id": "v0", "human": True}])
    res = hf.store.recognize(hf.geo.view("badge", 0.9))
    hits = [c for c in res.candidates if c.id == pid]
    assert len(hits) == 1
    assert hits[0].from_mutable and hits[0].label == "badge"


def test_reject_on_fleet_object_is_session_scoped(hf):
    fid = new_id()
    hf.seed_immutable(fid, "stapler", [hf.geo.view("stapler", 0.98)])
    look = hf.geo.view("stapler", 0.9)
    hf.ingest(1, look)
    assert hf.track_state(1).state == "recognized"
    hf.send(Reject(tid=1, epoch=1, object_id=fid))
    assert hf.track_state(1).state == "unknown"
    # nothing persisted: the immutable mirror is read-only, the negative lives in core memory
    assert hf.core.session_negs[fid]
    hf.ingest(2, look)  # other tracks are protected too, via the session negative
    assert hf.track_state(2).state == "unknown"


def test_dedup_after_pull_pushed_leaves_unpushed_survives(hf):
    """THE §3.3 rule: delete mutable only if t_sync stamped AND id present in the
    fresh mirror. Never by bare timestamp."""
    pushed_id, local_id = new_id(), new_id()
    rows_p = [hf.geo.view("pushed-item", 0.98)]
    rows_l = [hf.geo.view("local-item", 0.98)]
    # pushed yesterday: t_sync stamped, and the pull just delivered it to the mirror
    hf.store.upsert_object(
        pushed_id, "pushed item", rows_p, [{"view_id": "p0", "human": True}], t_sync=1000.0
    )
    hf.seed_immutable(pushed_id, "pushed item", rows_p)
    # taught locally, never pushed — must survive any pull
    hf.store.upsert_object(local_id, "local item", rows_l, [{"view_id": "l0", "human": True}])

    removed = hf.store.dedup_after_pull()
    assert removed == [pushed_id]
    assert hf.store.get_object(pushed_id) is None  # gone from mutable...
    r = hf.store.recognize(hf.geo.view("pushed-item", 0.9))
    assert r.candidates and r.candidates[0].id == pushed_id  # ...but still recognized via mirror
    assert hf.store.get_object(local_id) is not None


def test_local_edit_after_push_survives_the_next_pull(hf):
    """Codex finding: a pushed object edited locally must NOT be deleted by
    dedup — the edit clears t_sync (content no longer matches the fleet)."""
    from fleetmemory.memory.core import Rename

    pid = new_id()
    rows = [hf.geo.view("badge", 0.98)]
    hf.store.upsert_object(pid, "badge", rows, [{"view_id": "b0", "human": True}], t_sync=1000.0)
    hf.seed_immutable(pid, "badge", rows)  # the pull delivered the pushed copy
    hf.send(Rename(object_id=pid, label="my badge"))  # local edit AFTER the push
    assert hf.store.dedup_after_pull() == []  # edit made it dirty: survives
    payload, _ = hf.store.get_object(pid)
    assert payload["label"] == "my badge" and "t_sync" not in payload


def test_same_id_mutable_beats_higher_scoring_mirror_copy(hf):
    """Mutable copy is the local truth for a shared id, even when the older
    mirror copy scores higher on this particular query."""
    pid = new_id()
    close = hf.geo.view("badge", 0.99)
    far = hf.geo.view("badge", 0.7)
    hf.seed_immutable(pid, "badge (stale fleet)", [close])  # scores higher...
    hf.store.upsert_object(pid, "badge", [far], [{"view_id": "v", "human": True}])
    res = hf.store.recognize(hf.geo.view("badge", 0.98))
    hit = next(c for c in res.candidates if c.id == pid)
    assert hit.from_mutable and hit.label == "badge"


def test_markpushed_skips_points_edited_mid_push(hf):
    """Codex finding: the push handshake snapshots content; if the user edits
    the object while the network push is in flight, MarkPushed must NOT stamp
    t_sync on the newer content (it would be deduped away on the next pull)."""
    import queue

    from fleetmemory.memory.core import MarkPushed, PreparePush, Rename

    hf.ingest(1, hf.geo.view("badge"))
    hf.send(Teach(tid=1, epoch=1, label="badge"))
    oid = hf.track_state(1).object_id

    reply = queue.Queue()
    hf.send(PreparePush(object_ids=[oid], reply=reply))
    prepared = reply.get_nowait()  # what the sync worker would upload
    hf.send(Rename(object_id=oid, label="my badge"))  # edit lands mid-flight
    hf.send(
        MarkPushed(
            items=[{"old_id": oid, "t_sync": 111.0, "fingerprint": prepared[0]["fingerprint"]}]
        )
    )
    payload, _ = hf.store.get_object(oid)
    assert "t_sync" not in payload  # still dirty: the edit will be re-pushed
    assert payload["label"] == "my badge"


def test_stamped_but_not_yet_mirrored_survives(hf):
    """t_sync alone must not kill a point: if the push landed but the pull hasn't
    delivered it yet, deleting would lose the object entirely."""
    pid = new_id()
    hf.store.upsert_object(
        pid, "in flight", [hf.geo.view("x", 0.98)], [{"view_id": "v", "human": True}], t_sync=1000.0
    )
    assert hf.store.dedup_after_pull() == []
    assert hf.store.get_object(pid) is not None


def test_teach_while_fleet_knows_similar_creates_local_object(hf):
    """A fleet object at suggest-tier similarity doesn't block teaching a new
    local object; the new object is mutable and pushable."""
    fid = new_id()
    hf.seed_immutable(fid, "team mug", [hf.geo.view("mug", 0.98)])
    hf.ingest(1, hf.geo.view("mug", 0.65))
    assert hf.track_state(1).state == "suggest"
    hf.send(Teach(tid=1, epoch=1, label="my mug"))
    oid = hf.track_state(1).object_id
    assert oid != fid
    payload, _ = hf.store.get_object(oid)
    assert payload["label"] == "my mug"


def test_unknown_still_surfaces_with_fleet_mirror_present(hf):
    hf.seed_immutable(new_id(), "stapler", [hf.geo.view("stapler", 0.98)])
    hf.ingest(1, hf.geo.view("novel-thing"))
    assert hf.last("track_update")["state"] == "unknown"
