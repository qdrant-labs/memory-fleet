"""Deterministic gate harness: mock embedding geometry + a synchronous core.

The geometry gives exact control over cosine similarity: view(concept, s)
returns a unit vector whose cosine to the concept's base vector is exactly s.
Cross-concept similarities are small (dim 64, fixed seed). The core runs with
drain() — no threads, no sleeps, no wall clock.
"""

import numpy as np
import pytest

from fleetmemory.memory.core import Core, Ingest
from fleetmemory.memory.store import Store

DIM = 64


class Geometry:
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self._bases: dict[str, np.ndarray] = {}

    def _unit(self) -> np.ndarray:
        v = self.rng.normal(size=DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    def concept(self, name: str) -> np.ndarray:
        if name not in self._bases:
            self._bases[name] = self._unit()
        return self._bases[name]

    def view(self, name: str, s: float = 0.9) -> np.ndarray:
        """Unit vector with cosine EXACTLY s to concept(name)."""
        base = self.concept(name)
        r = self._unit()
        r -= (r @ base) * base
        r /= np.linalg.norm(r)
        v = s * base + np.sqrt(1 - s * s) * r
        return (v / np.linalg.norm(v)).astype(np.float32)


class Harness:
    """Drives the real core synchronously and collects its events."""

    def __init__(self, tmp_path, with_immutable: bool = False):
        self.geo = Geometry()
        self.events: list[dict] = []
        self.store = Store(tmp_path / "data", dim=DIM, with_immutable=with_immutable)
        self.core = Core(
            self.store, device_name="unit-test", event_tag="gate", on_event=self.events.append
        )
        self.t = 0.0

    def close(self):
        self.store.close()

    def send(self, msg):
        self.core.submit(msg)
        self.core.drain()

    def ingest(self, tid: int, vec: np.ndarray, epoch: int = 1, dt: float = 0.5, quality=1.0):
        self.t += dt
        self.send(
            Ingest(tid=tid, epoch=epoch, vec=vec, thumb_jpeg=b"jpg", quality=quality, t=self.t)
        )

    def track_state(self, tid: int):
        return self.core.tracks.get(tid)

    def last(self, ev_type: str) -> dict | None:
        return next((e for e in reversed(self.events) if e["type"] == ev_type), None)

    def all(self, ev_type: str) -> list[dict]:
        return [e for e in self.events if e["type"] == ev_type]

    # -- direct immutable-shard seeding (stands in for a fleet pull) --

    def seed_immutable(self, object_id: str, label: str, rows, human=True, t_sync=1.0):
        from qdrant_edge import Point, UpdateOperation

        views = [{"view_id": f"fleet-{i}", "human": human} for i in range(len(rows))]
        self.store.immutable.update(
            UpdateOperation.upsert_points(
                [
                    Point(
                        id=object_id,
                        vector={
                            "exemplars": [r.tolist() for r in rows],
                            "label": self.store._bm25.embed_document(label),
                        },
                        payload={
                            "kind": "object",
                            "label": label,
                            "views": views,
                            "neg": [],
                            "device": "other-device",
                            "t_sync": t_sync,
                        },
                    )
                ]
            )
        )


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.close()


@pytest.fixture
def hf(tmp_path):
    """Harness with a fleet (immutable) mirror."""
    harness = Harness(tmp_path, with_immutable=True)
    yield harness
    harness.close()
