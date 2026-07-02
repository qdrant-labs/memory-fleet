"""Perception smoke: real models over the bundled clip (local only — not CI).

Sanity, not benchmarks: proposals come with masks, masked crops actually
flatten the background, embeddings are unit 512-d, and same-track crops sit
closer than different-track crops (the §9.2 separation, coarsely).
"""

import cv2
import numpy as np
import pytest

from fleetmemory.perception.crops import FILL, padded_crop
from fleetmemory.perception.detector import Detector
from fleetmemory.perception.embedder import DIM, Embedder

CLIP = "tests/fixtures/clip.mp4"

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def tracked_frames():
    det = Detector()
    det.warm()
    cap = cv2.VideoCapture(CLIP)
    out = []  # (frame, proposals)
    fidx = 0
    while len(out) < 12:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % 6 == 0:  # ~5 fps ingest, like live
            props, _ = det.track(frame)
            out.append((frame, props))
        fidx += 1
    cap.release()
    assert out, "clip unreadable"
    return out


def test_detector_proposes_tracked_masked_regions(tracked_frames):
    all_props = [p for _, props in tracked_frames for p in props]
    assert len(all_props) >= 10, "expected a lively desk scene"
    assert any(p.mask is not None for p in all_props)
    tids = {p.tid for p in all_props}
    assert len(tids) >= 2
    # continuity: at least one track spans several sampled frames
    from collections import Counter

    per_tid = Counter(p.tid for _, props in tracked_frames for p in props)
    assert max(per_tid.values()) >= 4


def test_masked_crop_flattens_background(tracked_frames):
    for frame, props in tracked_frames:
        for p in props:
            if p.mask is not None and len(p.mask) >= 3:
                crop = np.asarray(padded_crop(frame, p.box, p.mask))
                gray = (crop == FILL[0]).all(axis=2).mean()
                if gray > 0.02:  # some crops are mask-filling the pad margin
                    return
    pytest.fail("no crop showed the neutral-gray mask fill")


def test_embedder_unit_512(tracked_frames):
    frame, props = next((f, p) for f, p in tracked_frames if p)
    crops = [padded_crop(frame, p.box, p.mask) for p in props[:4]]
    vecs = Embedder().embed(crops)
    assert all(v.shape == (DIM,) for v in vecs)
    assert all(abs(float(np.linalg.norm(v)) - 1.0) < 1e-3 for v in vecs)


def test_same_track_closer_than_cross_track(tracked_frames):
    from collections import defaultdict

    emb = Embedder()
    by_tid = defaultdict(list)
    for frame, props in tracked_frames:
        for p in props:
            if p.mask is not None:
                by_tid[p.tid].append(padded_crop(frame, p.box, p.mask))
    multi = {t: c[:3] for t, c in by_tid.items() if len(c) >= 2}
    if len(multi) < 2:
        pytest.skip("clip gave too few multi-view tracks")
    vecs = {t: emb.embed(c) for t, c in multi.items()}
    same = [float(a @ b) for vs in vecs.values() for i, a in enumerate(vs) for b in vs[i + 1 :]]
    tids = list(vecs)
    diff = [
        float(a @ b)
        for i, t in enumerate(tids)
        for u in tids[i + 1 :]
        for a in vecs[t]
        for b in vecs[u]
    ]
    assert np.mean(same) > np.mean(diff)
