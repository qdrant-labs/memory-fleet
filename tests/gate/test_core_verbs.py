"""§6 gate scenarios: teach/recognize/suggest/reject/ignore/merge/prune/forget,
unknown surfacing, burst, incarnation guard — real core, real Edge shards,
mock geometry."""

from fleetmemory.memory.core import (
    BURST_VIEWS,
    Confirm,
    Forget,
    IgnoreTrack,
    Merge,
    Prune,
    Reject,
    Rename,
    Teach,
    TrackDied,
)


def teach_and_burst(h, tid, concept, label, views=BURST_VIEWS, epoch=1):
    """Teach a track, then feed the burst diverse views. Returns object_id."""
    h.ingest(tid, h.geo.view(concept), epoch=epoch)
    h.send(Teach(tid=tid, epoch=epoch, label=label))
    for _ in range(views):
        h.ingest(tid, h.geo.view(concept, 0.9), epoch=epoch)
    return h.track_state(tid).object_id


# ---------- unknown surfacing + teach + recognize ----------


def test_novel_track_surfaces_as_unknown(h):
    h.ingest(1, h.geo.view("mug"))
    ev = h.last("track_update")
    assert ev["state"] == "unknown" and ev["tid"] == 1
    assert h.last("query")["ms"] >= 0  # HUD latency emitted on every recognition


def test_teach_creates_object_and_burst_accretes_views(h):
    oid = teach_and_burst(h, 1, "mug", "red mug")
    assert h.last("object_created")["label"] == "red mug"
    payload, rows = h.store.get_object(oid)
    assert len(rows) == 1 + BURST_VIEWS  # first view + burst
    assert all(v["human"] for v in payload["views"])
    assert h.all("burst_progress")
    assert h.track_state(1).state == "recognized"


def test_taught_object_recognized_on_new_track(h):
    teach_and_burst(h, 1, "mug", "red mug")
    h.ingest(2, h.geo.view("mug", 0.9))
    ev = h.last("track_update")
    assert (ev["tid"], ev["state"], ev["label"]) == (2, "recognized", "red mug")


def test_reteaching_the_same_item_folds(h):
    """Instance model: same name AND it looks like that object -> fold."""
    oid = teach_and_burst(h, 1, "mug", "red mug")
    h.ingest(2, h.geo.view("mug", 0.85))  # the same mug, seen again
    h.send(Teach(tid=2, epoch=1, label="red mug"))
    assert h.track_state(2).object_id == oid
    assert len(h.all("object_created")) == 1
    assert len(h.store.scroll_objects(mutable_only=True)) == 1


def test_different_item_with_same_name_is_its_own_instance(h):
    """Instance model (Dylan, 2026-07-01): five different watches are five
    clean objects, all displayed as "watch" — never a melting-pot point."""
    a = teach_and_burst(h, 1, "watch-a", "watch")
    h.ingest(2, h.geo.view("watch-b", 0.9))  # a DIFFERENT-looking watch
    h.send(Teach(tid=2, epoch=1, label="watch"))
    b = h.track_state(2).object_id
    assert b != a
    objs = h.store.scroll_objects(mutable_only=True)
    assert len(objs) == 2 and all(pl["label"] == "watch" for _, pl, _ in objs)
    # each instance recognizes its own looks, both answer to "watch"
    h.ingest(3, h.geo.view("watch-b", 0.95))
    ev = h.last("track_update")
    assert (ev["state"], ev["label"], ev["object_id"]) == ("recognized", "watch", b)


def test_burst_ends_by_timeout_with_undiverse_views(h):
    h.ingest(1, h.geo.view("mug"))
    h.send(Teach(tid=1, epoch=1, label="mug"))
    same = h.geo.view("mug", 0.9)
    for _ in range(10):  # identical view: diversity gate blocks accretion
        h.ingest(1, same, dt=0.5)
    assert h.track_state(1).state == "recognized"  # timeout (3 s) ended the burst
    _, rows = h.store.get_object(h.track_state(1).object_id)
    assert len(rows) == 2  # original + one copy of the repeated view


# ---------- suggest tier ----------


def test_borderline_match_suggests_not_binds(h):
    teach_and_burst(h, 1, "mug", "red mug")
    h.ingest(2, h.geo.view("mug", 0.65))  # between S_suggest and S_same
    ev = h.last("track_update")
    assert ev["state"] == "suggest" and ev["label"] == "red mug"


def test_confirm_accretes_human_view_and_binds(h):
    oid = teach_and_burst(h, 1, "mug", "red mug", views=2)
    h.ingest(2, h.geo.view("mug", 0.65))
    h.send(Confirm(tid=2, epoch=1, object_id=oid))
    assert h.track_state(2).state == "recognized"
    payload, rows = h.store.get_object(oid)
    assert len(rows) == 4  # 1 + 2 burst + confirmed view


def test_reject_persists_negative_and_stops_resuggesting(h):
    oid = teach_and_burst(h, 1, "mug", "red mug")
    look = h.geo.view("mug", 0.65)
    h.ingest(2, look)
    assert h.last("track_update")["state"] == "suggest"
    h.send(Reject(tid=2, epoch=1, object_id=oid))
    assert h.track_state(2).state == "unknown"
    assert h.store.get_object(oid)[0]["neg"]  # negative persisted on the object
    h.ingest(3, look)  # fresh track, same look: the negative now vetoes it
    ev = h.last("track_update")
    assert ev["tid"] == 3 and ev["state"] == "unknown"


def test_reject_recovers_a_false_recognition_in_one_tap(h):
    oid = teach_and_burst(h, 1, "mug", "red mug")
    look = h.geo.view("mug", 0.9)
    h.ingest(2, look)
    assert h.last("track_update")["state"] == "recognized"
    h.send(Reject(tid=2, epoch=1, object_id=oid))  # reject works on recognized boxes too
    assert h.track_state(2).state == "unknown"
    h.ingest(2, look)
    assert h.track_state(2).state == "unknown"  # vetoed for this track


# ---------- ignore ----------


def test_ignored_track_becomes_blocklist_entry(h):
    look = h.geo.view("wall-art", 0.98)
    h.ingest(1, look)
    h.send(IgnoreTrack(tid=1, epoch=1))
    assert h.track_state(1).state == "ignored"
    h.ingest(2, h.geo.view("wall-art", 0.98))  # near-identical view on a new track
    assert h.track_state(2).state == "ignored"
    assert h.last("track_update")["state"] == "ignored"
    # blocklist entries are local-only points of kind=ignored
    assert not h.store.scroll_objects(mutable_only=True)  # no kind=object created
    assert len(h.store.scroll_objects(kind="ignored", mutable_only=True)) == 1


def test_ignored_lookalikes_softly_suppressed_but_distant_items_are_not(h):
    """Semantics changed 2026-07-01 (Dylan: ignored doors kept re-flooding the
    unknowns queue): a mid-similarity match to a blocklist entry is suppressed
    too — but stays VISIBLE as a faint box, one click from rescue. Genuinely
    different items still surface as unknown."""
    h.ingest(1, h.geo.view("plant"))
    h.send(IgnoreTrack(tid=1, epoch=1))
    h.ingest(2, h.geo.view("plant", 0.85))  # look-alike: soft-suppressed
    assert h.track_state(2).state == "ignored"
    h.ingest(3, h.geo.view("plant", 0.4))  # too different: must NOT suppress
    assert h.track_state(3).state == "unknown"


# ---------- forget / merge / rename / prune ----------


def test_forget_unbinds_and_deletes(h):
    oid = teach_and_burst(h, 1, "mug", "red mug")
    h.send(Forget(object_id=oid))
    assert h.store.get_object(oid) is None
    assert h.track_state(1).state == "unknown"
    h.ingest(2, h.geo.view("mug", 0.9))
    assert h.track_state(2).state == "unknown"  # memory really gone


def test_merge_folds_views_and_rebinds_tracks(h):
    a = teach_and_burst(h, 1, "mug", "mug A", views=2)
    b = teach_and_burst(h, 2, "mugB", "mug B", views=2)
    h.send(Merge(keep_id=a, fold_id=b))
    assert h.store.get_object(b) is None
    payload, rows = h.store.get_object(a)
    assert len(rows) > 3  # b's diverse views folded in
    assert h.track_state(2).object_id == a
    h.ingest(3, h.geo.view("mugB", 0.9))  # b's look now recognized as a
    ev = h.last("track_update")
    assert ev["state"] == "recognized" and ev["label"] == "mug A"


def test_rename_updates_label_and_repeats_are_legal(h):
    a = teach_and_burst(h, 1, "mug", "mug A", views=1)
    b = teach_and_burst(h, 2, "pen", "pen", views=1)
    h.send(Rename(object_id=a, label="my mug"))
    assert h.store.get_object(a)[0]["label"] == "my mug"
    assert h.track_state(1).label == "my mug"
    h.send(Rename(object_id=b, label="my mug"))  # display names may repeat
    assert h.store.get_object(b)[0]["label"] == "my mug"


def test_prune_removes_exactly_the_named_view(h):
    oid = teach_and_burst(h, 1, "mug", "red mug", views=3)
    payload, rows = h.store.get_object(oid)
    victim = payload["views"][2]["view_id"]
    n = len(rows)
    h.send(Prune(object_id=oid, view_id=victim))
    payload2, rows2 = h.store.get_object(oid)
    assert len(rows2) == n - 1
    assert victim not in [v["view_id"] for v in payload2["views"]]
    assert len(rows2) == len(payload2["views"])  # row-aligned metadata survived


def test_prune_last_view_forgets_object(h):
    h.ingest(1, h.geo.view("pen"))
    h.send(Teach(tid=1, epoch=1, label="pen"))
    oid = h.track_state(1).object_id
    vid = h.store.get_object(oid)[0]["views"][0]["view_id"]
    h.send(Prune(object_id=oid, view_id=vid))
    assert h.store.get_object(oid) is None
    assert h.last("object_deleted")["object_id"] == oid


# ---------- incarnation guard (§3.2) ----------


def test_stale_ingest_from_dead_incarnation_dropped(h):
    h.ingest(1, h.geo.view("mug"), epoch=1)
    h.ingest(1, h.geo.view("pen"), epoch=2)  # track re-bound: new incarnation
    n_events = len(h.events)
    h.ingest(1, h.geo.view("mug"), epoch=1)  # late worker result from the old life
    assert len(h.events) == n_events  # silently dropped
    assert h.track_state(1).epoch == 2


def test_stale_verb_dropped(h):
    teach_and_burst(h, 1, "mug", "red mug")
    h.ingest(1, h.geo.view("pen"), epoch=2)  # re-bound before the user's click landed
    h.send(Teach(tid=1, epoch=1, label="stale name"))  # aimed at the dead incarnation
    assert h.track_state(1).label != "stale name"
    assert len(h.store.scroll_objects(mutable_only=True)) == 1


def test_track_death_cleans_state(h):
    h.ingest(1, h.geo.view("mug"))
    h.send(TrackDied(tid=1, epoch=1))
    assert h.track_state(1) is None
