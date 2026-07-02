"""Pre-stage ritual (PLAN.md §4): verify every model loads from LOCAL CACHE with
networking disabled — a first-run download at a venue is a classic demo killer.
Run via `make demo-check` (which also runs smoke + drive + sync tests).
"""

import os
import sys
from pathlib import Path

# forbid every download path BEFORE the ML imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["YOLO_AUTOINSTALL"] = "false"

import numpy as np  # noqa: E402


def check(name, fn):
    try:
        fn()
        print(f"  ✓ {name}")
        return True
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return False


def detector():
    from fleetmemory.perception.detector import WEIGHTS, Detector

    weights = Path(WEIGHTS)
    assert weights.exists(), f"{WEIGHTS} missing from repo root (would download at the venue)"
    d = Detector()
    d.warm()


def embedder():
    from fleetmemory.perception.crops import padded_crop
    from fleetmemory.perception.embedder import DIM, Embedder

    frame = np.full((240, 320, 3), 90, dtype=np.uint8)
    crop = padded_crop(frame, (0.2, 0.2, 0.8, 0.8))
    vecs = Embedder().embed([crop])
    assert vecs[0].shape == (DIM,)


def edge():
    import tempfile

    from fleetmemory.memory.store import Store

    with tempfile.TemporaryDirectory() as td:
        s = Store(Path(td) / "d", dim=32)
        s.close()


def main():
    print("demo-check: offline model loads (network downloads disabled)")
    ok = all(
        [
            check("YOLOE detector (local weights)", detector),
            check("Unicom embedder (fastembed cache)", embedder),
            check("Qdrant Edge shard create/load", edge),
        ]
    )
    if not ok:
        sys.exit(1)
    print("all models load offline — stage-ready")


if __name__ == "__main__":
    main()
