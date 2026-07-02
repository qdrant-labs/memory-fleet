"""Gate: 2026-07-01 feedback rounds — ignore folding, vector-count memories,
soft suppression, memory guesses, cap-full human replacement."""

from fleetmemory.memory.core import (
    VIEW_CAP,
    Confirm,
    IgnoreObject,
    IgnoreTrack,
    InventoryRequest,
    Teach,
)
from fleetmemory.memory.store import new_id


def test_taught_object_outranks_soft_suppression(h):
    h.ingest(1, h.geo.view("door", 0.7))
    h.send(IgnoreTrack(tid=1, epoch=1))  # blocklist a door-ish look
    h.ingest(2, h.geo.view("door", 0.98))
    h.send(Teach(tid=2, epoch=1, label="door sign"))
    for _ in range(3):
        h.ingest(2, h.geo.view("door", 0.95))
    h.ingest(3, h.geo.view("door", 0.97))  # matches the TAUGHT object strongly
    ev = h.last("track_update")
    assert (ev["state"], ev["label"]) == ("recognized", "door sign")


def test_ignored_object_keeps_its_name(h):
    h.ingest(1, h.geo.view("door"))
    h.send(Teach(tid=1, epoch=1, label="door"))
    h.send(IgnoreObject(object_id=h.track_state(1).object_id))
    h.send(InventoryRequest())
    ignored = [i for i in h.last("inventory")["items"] if i["ignored"]]
    assert ignored and ignored[0]["label"] == "door"  # not "(unnamed)"


def test_unknown_carries_memory_guesses(h):
    h.ingest(1, h.geo.view("watch"))
    h.send(Teach(tid=1, epoch=1, label="watch"))
    h.ingest(2, h.geo.view("watch", 0.45))  # below suggest: unknown, but close-ish
    ev = h.last("track_update")
    assert ev["state"] == "unknown"
    assert "watch" in [g["label"] for g in ev["guesses"]]


def test_full_object_still_learns_from_confirms(h):
    """A watch at VIEW_CAP views must keep improving: a confirmed human view
    replaces the most redundant auto view instead of being dropped."""
    rows = [h.geo.view("watch", 0.9) for _ in range(VIEW_CAP)]
    views = [{"view_id": f"v{i}", "human": i < 2} for i in range(VIEW_CAP)]
    oid = new_id()
    h.store.upsert_object(oid, "watch", rows, views)
    h.ingest(1, h.geo.view("watch", 0.7))
    assert h.track_state(1).state == "suggest"
    h.send(Confirm(tid=1, epoch=1, object_id=oid))
    payload, rows2 = h.store.get_object(oid)
    assert len(rows2) == VIEW_CAP  # still capped
    assert sum(1 for v in payload["views"] if v["human"]) == 3  # took an auto slot


def test_repeated_ignores_fold_into_one_blocklist_entry(h):
    """Hair keeps coming back: each ignore of a similar look must strengthen
    ONE blocklist entry (more views -> better suppression), not litter dozens."""
    for tid, s in [(1, 0.98), (2, 0.75), (3, 0.7)]:
        h.ingest(tid, h.geo.view("hair", s))
        h.send(IgnoreTrack(tid=tid, epoch=1))
    entries = h.store.scroll_objects(kind="ignored", mutable_only=True)
    assert len(entries) == 1
    assert len(entries[0][1]["views"]) == 3  # all three looks live on one entry


def test_unrelated_ignore_creates_its_own_entry(h):
    h.ingest(1, h.geo.view("hair", 0.98))
    h.send(IgnoreTrack(tid=1, epoch=1))
    h.ingest(2, h.geo.view("smoke-detector", 0.98))  # nothing like hair
    h.send(IgnoreTrack(tid=2, epoch=1))
    assert len(h.store.scroll_objects(kind="ignored", mutable_only=True)) == 2


def test_memories_count_vectors_not_points(h):
    assert h.store.vector_count() == 0
    h.ingest(1, h.geo.view("mug"))
    h.send(Teach(tid=1, epoch=1, label="mug"))
    for _ in range(3):  # burst accretes diverse views
        h.ingest(1, h.geo.view("mug", 0.9))
    payload, rows = h.store.get_object(h.track_state(1).object_id)
    assert h.store.vector_count() == len(rows) >= 4  # one object, MANY memories
    h.ingest(2, h.geo.view("something-else"))  # unknown: queries but accretes nothing
    assert h.last("query")["searched"] == h.store.vector_count()
