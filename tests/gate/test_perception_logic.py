"""Phase 2 gate: model-free perception logic — area band, stability, cadence, epochs."""

from fleetmemory.perception.cadence import EmbedScheduler
from fleetmemory.perception.detector import area_band_ok


def test_area_band_rejects_specks_and_room_blobs():
    assert not area_band_ok((0.5, 0.5, 0.51, 0.51))  # speck
    assert not area_band_ok((0.0, 0.0, 1.0, 0.9))  # room-spanning blob
    assert area_band_ok((0.4, 0.4, 0.6, 0.6))  # hand-held object
    # the cap is live-tunable: a half-frame box passes only if the user raises it
    assert not area_band_ok((0.2, 0.2, 0.8, 0.8))  # 36% of frame, default cap 20%
    assert area_band_ok((0.2, 0.2, 0.8, 0.8), max_area=0.5)


def test_track_embeds_only_after_stability():
    s = EmbedScheduler(stable_frames=3, requery_interval=2.0)
    assert s.tick([7], now=0.0) == ([], [])
    assert s.tick([7], now=0.2) == ([], [])
    embed, _ = s.tick([7], now=0.4)  # 3rd consecutive tick
    assert [(i.tid, i.epoch) for i in embed] == [(7, 1)]


def test_requery_cadence_not_every_frame():
    s = EmbedScheduler(stable_frames=1, requery_interval=2.0)
    assert len(s.tick([1], now=0.0)[0]) == 1
    assert s.tick([1], now=0.5)[0] == []  # too soon
    assert s.tick([1], now=1.9)[0] == []
    assert len(s.tick([1], now=2.1)[0]) == 1  # stale -> re-embed


def test_flicker_resets_stability_but_keeps_epoch():
    s = EmbedScheduler(stable_frames=3, requery_interval=2.0, dead_after=1.5)
    s.tick([1], now=0.0)
    s.tick([1], now=0.2)
    s.tick([], now=0.4)  # flicker out one tick
    assert s.tick([1], now=0.6)[0] == []  # streak restarted
    assert s.epoch(1) == 1  # same incarnation: it never died
    s.tick([1], now=0.8)
    embed, _ = s.tick([1], now=1.0)
    assert [(i.tid, i.epoch) for i in embed] == [(1, 1)]


def test_same_tid_after_tick_pause_is_new_incarnation():
    """Ticks pause (camera off), the tracker re-emits the same tid on resume:
    that must be a new incarnation — the old binding must not leak."""
    s = EmbedScheduler(stable_frames=1, requery_interval=2.0, dead_after=1.5)
    s.tick([5], now=0.0)
    assert s.epoch(5) == 1
    to_embed, died = s.tick([5], now=60.0)  # visible again after a long pause
    assert [(i.tid, i.epoch) for i in died] == [(5, 1)]
    assert s.epoch(5) == 2
    assert [(i.tid, i.epoch) for i in to_embed] == [(5, 2)]


def test_dead_track_reappearing_is_new_incarnation():
    s = EmbedScheduler(stable_frames=1, requery_interval=2.0, dead_after=1.5)
    s.tick([5], now=0.0)
    _, died = s.tick([], now=2.0)  # unseen past dead_after
    assert [(i.tid, i.epoch) for i in died] == [(5, 1)]
    embed, _ = s.tick([5], now=3.0)  # same tracker id, new life
    assert [(i.tid, i.epoch) for i in embed] == [(5, 2)]


def test_only_new_or_stale_tracks_embed_never_all_proposals():
    s = EmbedScheduler(stable_frames=1, requery_interval=2.0)
    s.tick([1, 2, 3], now=0.0)
    embed, _ = s.tick([1, 2, 3, 4], now=0.5)  # only the newcomer embeds
    assert [i.tid for i in embed] == [4]
