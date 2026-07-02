"""Sync test harness: real Docker/CI Qdrant server, random controlled vectors,
no models (PLAN.md §7 — CI-safe). Each test gets its own fleet collection."""

import os
import queue
import time
import uuid

import numpy as np
import pytest
import requests

from fleetmemory.memory.core import Call, Core
from fleetmemory.memory.store import Store
from fleetmemory.sync.client import FleetClient
from fleetmemory.sync.manager import SyncManager

QDRANT_URL = os.environ.get("QDRANT_TEST_URL", "http://localhost:6333")
DIM = 64


def _server_up() -> bool:
    try:
        return requests.get(QDRANT_URL, timeout=2).ok
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(not _server_up(), reason=f"no qdrant at {QDRANT_URL}")


class Geometry:
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self._bases: dict[str, np.ndarray] = {}

    def _unit(self):
        v = self.rng.normal(size=DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    def concept(self, name):
        if name not in self._bases:
            self._bases[name] = self._unit()
        return self._bases[name]

    def view(self, name, s: float = 0.9):
        base = self.concept(name)
        r = self._unit()
        r -= (r @ base) * base
        r /= np.linalg.norm(r)
        return (s * base + np.sqrt(1 - s * s) * r).astype(np.float32)


class Device:
    """One simulated unit: store + threaded core + sync manager, shared geometry."""

    def __init__(self, tmp_path, name: str, collection: str, geo: Geometry):
        self.geo = geo
        self.events: list[dict] = []
        self.store = Store(tmp_path / name, dim=DIM, with_immutable=True)
        self.core = Core(self.store, device_name=name, on_event=self.events.append)
        self.core.start()
        self.client = FleetClient(QDRANT_URL, None, collection=collection, dim=DIM)
        self.sync = SyncManager(self.core, self.client, on_event=self.events.append, interval=3600)

    def teach_direct(self, label: str, concept: str, n_views: int = 3, oid: str | None = None):
        """Plant a taught object straight into the mutable shard (models not needed)."""
        oid = oid or str(uuid.uuid4())
        rows = [self.geo.view(concept, 0.9) for _ in range(n_views)]
        views = [{"view_id": uuid.uuid4().hex[:12], "human": True} for _ in rows]
        self._core_call(
            lambda: self.store.upsert_object(oid, label, rows, views, device=self.core.device_name)
        )
        return oid

    def _core_call(self, fn):
        """Run fn on the core thread (the shard's only legal thread) and wait."""
        done: queue.Queue = queue.Queue()
        self.core.submit(Call(fn=fn, reply=done))
        return done.get(timeout=10)

    def wait_event(self, ev_type: str, timeout: float = 15.0) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            ev = next((e for e in self.events if e["type"] == ev_type), None)
            if ev:
                return ev
            time.sleep(0.05)
        raise AssertionError(
            f"no {ev_type} event within {timeout}s: {[e['type'] for e in self.events]}"
        )

    def recognize(self, concept: str, s: float = 0.9):
        return self._core_call(lambda: self.store.recognize(self.geo.view(concept, s)))

    def close(self):
        self.sync.stop()
        self.core.stop()
        self.store.close()


@pytest.fixture
def fleet(tmp_path):
    """(make_device, client) bound to a throwaway collection; cleaned up after."""
    collection = f"fm-test-{uuid.uuid4().hex[:8]}"
    geo = Geometry()
    devices: list[Device] = []

    def make(name: str) -> Device:
        d = Device(tmp_path, name, collection, geo)
        devices.append(d)
        return d

    yield make
    for d in devices:
        d.close()
    try:
        devices[0].client.client.delete_collection(collection)
    except Exception:
        pass
