"""Phase 2 exit criterion (PLAN.md §8.2): integrated perception soak.

detect + track + embed-on-cadence + MAX_SIM query, ≥2 fps ingest with stable
memory over the run. Conservative by design: embeds run inline (the app moves
them to a worker), and the query hits a seeded Edge shard.

  uv run python scripts/soak.py --source 0 --minutes 10          # live webcam
  uv run python scripts/soak.py --source tests/fixtures/clip.mp4 # looped file
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from fleetmemory.perception.cadence import EmbedScheduler
from fleetmemory.perception.crops import padded_crop
from fleetmemory.perception.detector import Detector
from fleetmemory.perception.embedder import DIM, Embedder


def seeded_shard(path: Path, n: int):
    from qdrant_edge import (
        Distance,
        EdgeConfig,
        EdgeShard,
        EdgeVectorParams,
        MultiVectorComparator,
        MultiVectorConfig,
        Point,
        UpdateOperation,
    )

    rng = np.random.default_rng(1)
    cfg = EdgeConfig(
        vectors={
            "exemplars": EdgeVectorParams(
                size=DIM,
                distance=Distance.Cosine,
                multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MaxSim),
            )
        }
    )
    path.mkdir(parents=True, exist_ok=True)
    shard = EdgeShard.create(str(path), cfg)
    for start in range(0, n, 500):
        k = min(500, n - start)
        vs = rng.normal(size=(k, 3, DIM)).astype(np.float32)
        vs /= np.linalg.norm(vs, axis=2, keepdims=True)
        shard.update(
            UpdateOperation.upsert_points(
                [
                    Point(
                        id=str(uuid.uuid4()),
                        vector={"exemplars": vs[i].tolist()},
                        payload={"label": f"seed-{start + i}"},
                    )
                    for i in range(k)
                ]
            )
        )
    shard.optimize()
    return shard


def rss_mb() -> float:
    """CURRENT rss via ps — ru_maxrss is a lifetime high-water mark, which lets
    an early allocation spike mask steady growth underneath it."""
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index or video path (looped)")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--seed-points", type=int, default=500)
    ap.add_argument("--frame-step", type=int, default=6, help="file mode: sample every Nth frame")
    args = ap.parse_args()

    from qdrant_edge import Query, QueryRequest

    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {args.source!r}")

    work = Path(tempfile.mkdtemp(prefix="fm-soak-"))
    shard = seeded_shard(work / "shard", args.seed_points)
    det = Detector()
    det.warm()
    emb = Embedder()
    emb.load()
    sched = EmbedScheduler()

    detect_ms, embed_ms, query_ms = [], [], []
    ticks = embeds = 0
    rss_track = []  # (elapsed_s, rss_mb)
    t_start = time.time()
    t_report = t_start
    fidx = 0

    print(f"soak: source={args.source} minutes={args.minutes} seed={args.seed_points}")
    while time.time() - t_start < args.minutes * 60:
        ok, frame = cap.read()
        if not ok:
            if is_cam:
                raise SystemExit("webcam read failed")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the file
            continue
        fidx += 1
        if not is_cam and args.frame_step > 1 and fidx % args.frame_step:
            continue

        now = time.time()
        props, dms = det.track(frame)
        detect_ms.append(dms)
        ticks += 1

        to_embed, _died = sched.tick([p.tid for p in props], now)
        due = {i.tid for i in to_embed}
        crops = [padded_crop(frame, p.box, p.mask) for p in props if p.tid in due]
        if crops:
            t0 = time.perf_counter_ns()
            vecs = emb.embed(crops)
            embed_ms.append((time.perf_counter_ns() - t0) / 1e6 / len(crops))
            embeds += len(crops)
            for v in vecs:
                t0 = time.perf_counter_ns()
                shard.query(
                    QueryRequest(
                        query=Query.Nearest([v.tolist()], using="exemplars"),
                        limit=5,
                        with_payload=True,
                        with_vector=False,
                    )
                )
                query_ms.append((time.perf_counter_ns() - t0) / 1e6)

        if now - t_report >= 15:
            t_report = now
            el = now - t_start
            fps = ticks / el
            rss_track.append((el, rss_mb()))
            print(
                f"[{el:5.0f}s] fps {fps:4.2f} | detect med {np.median(detect_ms):4.0f} ms"
                f" | embed/crop {np.median(embed_ms) if embed_ms else 0:5.1f} ms"
                f" | query {np.median(query_ms) if query_ms else 0:5.2f} ms"
                f" | embeds {embeds} | rss {rss_track[-1][1]:5.0f} MB",
                flush=True,
            )

    cap.release()
    el = time.time() - t_start
    fps = ticks / el
    # ru_maxrss is a high-water mark: warm-up climbs, stability = the curve flattens.
    # Judge the final quarter — a leak keeps climbing there, a plateau doesn't.
    tail = [r for t, r in rss_track if t > el * 0.75] or [rss_mb()]
    growth = (tail[-1] - tail[0]) if len(tail) > 1 else 0.0
    print("\n--- soak summary ---")
    print(f"duration {el / 60:.1f} min | ticks {ticks} | ingest {fps:.2f} fps")
    print(
        f"detect med {np.median(detect_ms):.0f} ms p90 {np.percentile(detect_ms, 90):.0f} ms"
        f" | embed/crop med {np.median(embed_ms) if embed_ms else 0:.1f} ms"
        f" | query med {np.median(query_ms) if query_ms else 0:.2f} ms | embeds {embeds}"
    )
    peak = max([r for _, r in rss_track] + [rss_mb()])
    print(f"rss peak {peak:.0f} MB, final-quarter growth {growth:+.0f} MB")
    ok_fps = fps >= 2.0
    ok_mem = growth < 100
    print(f"VERDICT: fps {'PASS' if ok_fps else 'FAIL'} | memory {'PASS' if ok_mem else 'FAIL'}")
    shard.close()
    shutil.rmtree(work)
    raise SystemExit(0 if ok_fps and ok_mem else 1)


if __name__ == "__main__":
    main()
