"""Build the prebuilt 100k-memory stunt shard (PLAN.md §4 step 4, §9.4).

Synthetic points carry synthetic:true (inventory/map exclude them), ~3 exemplar
rows each (§9.4's honest ratio), no thumbnails. Building takes ~1 min — that is
exactly why it is prebuilt and swapped in on a keypress, never built on stage.

  uv run python scripts/preload_scale.py [--points 100000] [--dest edge-data-scale]
"""

import argparse
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
from qdrant_edge import EdgeShard, Point, UpdateOperation

from fleetmemory.memory.store import DIM, shard_config

ROWS = 3
BATCH = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=100_000)
    ap.add_argument("--dest", default="edge-data-scale")
    args = ap.parse_args()

    dest = Path(args.dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    rng = np.random.default_rng(42)
    shard = EdgeShard.create(str(dest), shard_config(DIM))
    t0 = time.time()
    for start in range(0, args.points, BATCH):
        k = min(BATCH, args.points - start)
        vs = rng.normal(size=(k, ROWS, DIM)).astype(np.float32)
        vs /= np.linalg.norm(vs, axis=2, keepdims=True)
        shard.update(
            UpdateOperation.upsert_points(
                [
                    Point(
                        id=str(uuid.uuid4()),
                        vector={"exemplars": vs[i].tolist()},
                        payload={
                            "kind": "object",
                            "label": f"obj-{start + i:06d}",
                            "device": f"unit-{(start + i) % 40:02d}",
                            "synthetic": True,
                        },
                    )
                    for i in range(k)
                ]
            )
        )
        if (start // BATCH) % 20 == 0:
            print(f"  {start + k:>7}/{args.points} ({time.time() - t0:.0f}s)", flush=True)
    print(f"inserted {args.points} in {time.time() - t0:.0f}s; optimizing…", flush=True)
    t1 = time.time()
    shard.optimize()
    print(f"optimize() took {time.time() - t1:.0f}s")
    shard.close()
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e9
    print(f"stunt shard ready at {dest} ({size:.1f} GB). Press S in the UI to swap it in.")


if __name__ == "__main__":
    main()
