"""Gate: 2026-07-01 feedback round 3 — ignore folding, vector-count memories."""

from fleetmemory.memory.core import IgnoreTrack, Teach


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
