"""Gate: 2026-07-01 feedback round — person filter, ignored inventory,
archived (departed) unknowns."""

from fleetmemory.memory.core import (
    DismissUnknown,
    Forget,
    IgnoreTrack,
    InventoryRequest,
    Teach,
    TrackDied,
)
from fleetmemory.perception.detector import is_person_like


def test_person_and_bodypart_classes_filtered():
    for name in ["person", "human face", "hand", "left arm", "woman", "hair"]:
        assert is_person_like(name), name
    for name in ["coffee mug", "watch", "handbag-strap", "person-shaped lamp"]:
        # handbag must NOT match "hand"; word-level matching only
        if name in ("coffee mug", "watch", "handbag-strap"):
            assert not is_person_like(name), name
    assert is_person_like("person-shaped lamp")  # contains the word "person"


def test_ignored_items_appear_in_inventory_and_unignore_works(h):
    h.ingest(1, h.geo.view("wall-art", 0.98))
    h.send(IgnoreTrack(tid=1, epoch=1))
    h.send(InventoryRequest())
    items = h.last("inventory")["items"]
    ignored = [i for i in items if i["ignored"]]
    assert len(ignored) == 1
    assert ignored[0]["views"], "blocklist entry must expose its vectors for review"
    assert ignored[0]["thumb"], "blocklist entry needs a thumbnail to be reviewable"

    h.send(Forget(object_id=ignored[0]["object_id"]))  # unignore
    h.ingest(2, h.geo.view("wall-art", 0.98))
    assert h.track_state(2).state == "unknown"  # tracked again, not suppressed
    h.send(InventoryRequest())
    assert not [i for i in h.last("inventory")["items"] if i["ignored"]]


def test_departed_unknown_stays_teachable(h):
    h.ingest(1, h.geo.view("watch"))
    assert h.track_state(1).state == "unknown"
    h.send(TrackDied(tid=1, epoch=1))  # wrist went down to type
    ev = h.last("unknown_archived")
    assert (ev["tid"], ev["epoch"]) == (1, 1) and ev["thumb"]

    h.send(Teach(tid=1, epoch=1, label="my watch"))  # taught from the drawer
    assert h.last("object_created")["label"] == "my watch"
    assert h.last("unknown_removed")["tid"] == 1

    h.ingest(2, h.geo.view("watch", 0.9))  # wrist comes back up
    ev = h.last("track_update")
    assert (ev["state"], ev["label"]) == ("recognized", "my watch")


def test_departed_unknown_can_fold_into_existing_label(h):
    h.ingest(1, h.geo.view("mug"))
    h.send(Teach(tid=1, epoch=1, label="mug"))
    h.ingest(2, h.geo.view("mug", 0.8))  # the same mug, briefly out of frame
    h.send(TrackDied(tid=2, epoch=2))
    h.send(Teach(tid=2, epoch=2, label="mug"))  # same name + same look: folds
    assert len(h.store.scroll_objects(mutable_only=True)) == 1


def test_dismiss_archived_unknown(h):
    h.ingest(1, h.geo.view("junk"))
    h.send(TrackDied(tid=1, epoch=1))
    h.send(DismissUnknown(tid=1, epoch=1))
    assert h.last("unknown_removed")["tid"] == 1
    h.send(Teach(tid=1, epoch=1, label="too late"))  # dismissed: teach is a no-op
    assert not h.store.find_label("too late")


def test_recognized_track_death_is_not_archived(h):
    h.ingest(1, h.geo.view("mug"))
    h.send(Teach(tid=1, epoch=1, label="mug"))
    for _ in range(6):
        h.ingest(1, h.geo.view("mug", 0.9))
    h.send(TrackDied(tid=1, epoch=1))
    assert h.last("unknown_archived") is None  # only unnamed tracks are worth keeping
