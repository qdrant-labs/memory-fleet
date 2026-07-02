"""Embed scheduling: which tracks get embedded this tick.

Pure logic, no models — the budget closes because we embed new/stale stable
tracks only, never all proposals every frame. Also the home of incarnation
epochs: a tracker id that dies and later reappears is a NEW incarnation; the
core drops stale messages by (tid, epoch) mismatch.
"""

from dataclasses import dataclass

STABLE_FRAMES = 3  # K consecutive ticks before a track is worth embedding
REQUERY_INTERVAL = 2.0  # s between re-embeds of a bound track (self-correction)
DEAD_AFTER = 1.5  # s unseen before a track is declared dead


@dataclass(slots=True)
class _TrackState:
    epoch: int
    streak: int = 1  # consecutive ticks seen
    last_seen: float = 0.0
    last_embed: float = -1.0  # -1: never
    seen_prev_tick: bool = True


@dataclass(slots=True)
class Incarnation:
    tid: int
    epoch: int


class EmbedScheduler:
    """Feed it every tick's visible tids; it returns which incarnations to embed."""

    def __init__(
        self,
        stable_frames: int = STABLE_FRAMES,
        requery_interval: float = REQUERY_INTERVAL,
        dead_after: float = DEAD_AFTER,
    ):
        self.stable_frames = stable_frames
        self.requery_interval = requery_interval
        self.dead_after = dead_after
        self._tracks: dict[int, _TrackState] = {}
        self._next_epoch: int = 0

    def tick(
        self, visible_tids: list[int], now: float, burst_tids: set[int] | None = None
    ) -> tuple[list[Incarnation], list[Incarnation]]:
        """Returns (to_embed, died) for this tick. Tracks in burst_tids (a teach
        capture burst is running) embed EVERY tick, bypassing cadence."""
        visible = set(visible_tids)
        burst_tids = burst_tids or set()

        died: list[Incarnation] = []
        for tid, st in list(self._tracks.items()):
            if tid not in visible:
                st.seen_prev_tick = False
                st.streak = 0
                if now - st.last_seen > self.dead_after:
                    died.append(Incarnation(tid, st.epoch))
                    del self._tracks[tid]

        to_embed: list[Incarnation] = []
        for tid in visible_tids:
            st = self._tracks.get(tid)
            if st is not None and now - st.last_seen > self.dead_after:
                # ticks paused (camera off) and the tracker re-emitted the same
                # tid: that's a NEW incarnation, not a continuation — the old
                # binding must not leak onto whatever object holds the tid now
                died.append(Incarnation(tid, st.epoch))
                del self._tracks[tid]
                st = None
            if st is None:
                self._next_epoch += 1
                st = _TrackState(epoch=self._next_epoch)
                self._tracks[tid] = st
            else:
                st.streak = st.streak + 1 if st.seen_prev_tick else 1
                st.seen_prev_tick = True
            st.last_seen = now

            due = st.streak >= self.stable_frames and (
                st.last_embed < 0 or now - st.last_embed >= self.requery_interval
            )
            if due or tid in burst_tids:
                st.last_embed = now
                to_embed.append(Incarnation(tid, st.epoch))
        return to_embed, died

    def stability(self, tid: int) -> int:
        st = self._tracks.get(tid)
        return st.streak if st else 0

    def epoch(self, tid: int) -> int | None:
        st = self._tracks.get(tid)
        return st.epoch if st else None
