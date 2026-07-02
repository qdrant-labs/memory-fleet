"""Sync test harness: runs against the real Qdrant Cloud cluster from .env —
the fleet is Cloud, full stop (no Docker anywhere; Dylan, 2026-07-01). Each
test gets a throwaway collection (fm-test-*), deleted in teardown. Random
controlled vectors, no models."""

import queue
import time
import uuid

import numpy as np
import pytest
import requests

from fleetmemory.config import load_settings
from fleetmemory.memory.core import Call, Core
from fleetmemory.memory.store import Store
from fleetmemory.sync.client import FleetClient
from fleetmemory.sync.manager import SyncManager

_settings = load_settings()
QDRANT_URL = _settings.qdrant_url or ""
QDRANT_KEY = _settings.qdrant_api_key
DIM = 64


def _fleet_reachable() -> bool:
    if not QDRANT_URL:
        return False
    try:
        headers = {"api-key": QDRANT_KEY} if QDRANT_KEY else {}
        return requests.get(QDRANT_URL, headers=headers, timeout=5).ok
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(
    not _fleet_reachable(), reason="no reachable fleet (QDRANT_URL) in .env"
)


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
        self.client = FleetClient(QDRANT_URL, QDRANT_KEY, collection=collection, dim=DIM)
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
