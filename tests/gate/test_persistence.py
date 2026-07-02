"""Gate: memory survives a restart — shard reload path, mock vectors."""

from fleetmemory.memory.core import Core
from fleetmemory.memory.store import Store
from tests.gate.conftest import Harness


def test_taught_object_survives_restart(tmp_path):
    h = Harness(tmp_path)
    from fleetmemory.memory.core import Teach

    h.ingest(1, h.geo.view("mug"))
    h.send(Teach(tid=1, epoch=1, label="red mug"))
    oid = h.track_state(1).object_id
    h.close()

    store = Store(tmp_path / "data", dim=64)  # loads, not creates
    events = []
    core = Core(store, on_event=events.append)
    payload, rows = store.get_object(oid)
    assert payload["label"] == "red mug" and rows

    from fleetmemory.memory.core import Ingest

    core.submit(
        Ingest(tid=9, epoch=1, vec=h.geo.view("mug", 0.9), thumb_jpeg=None, quality=1.0, t=100.0)
    )
    core.drain()
    ev = next(e for e in reversed(events) if e["type"] == "track_update")
    assert ev["state"] == "recognized" and ev["label"] == "red mug"
    store.close()
