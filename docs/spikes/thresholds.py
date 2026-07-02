"""Operating-point curves for Unicom-ViT-B-32: FMR/TPR at candidate thresholds.

Same pair protocol as bench_embeddings.py (within-track same pairs,
coexisting-track diff pairs, same-class hard subset).
"""

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

CROPS = Path(sys.argv[1])
manifest = json.loads((CROPS / "manifest.json").read_text())

tracks = defaultdict(list)
for m in manifest:
    tracks[(m["clip"], m["tid"])].append(m)
tracks = {k: sorted(v, key=lambda m: m["frame"]) for k, v in tracks.items() if len(v) >= 4}
frames_of = {k: {m["frame"] for m in v} for k, v in tracks.items()}
cls_of = {k: v[0]["cls"] for k, v in tracks.items()}

paths, owner = [], []
for k, rows in tracks.items():
    for m in rows:
        paths.append(CROPS / m["file"])
        owner.append(k)
images = [Image.open(p).convert("RGB") for p in paths]
idx_of = defaultdict(list)
for i, k in enumerate(owner):
    idx_of[k].append(i)

same_pairs, diff_pairs, hard_mask = [], [], []
keys = sorted(tracks)
for k in keys:
    same_pairs += list(combinations(idx_of[k], 2))
rng = np.random.default_rng(11)
for a, b in combinations(keys, 2):
    if a[0] != b[0] or not (frames_of[a] & frames_of[b]):
        continue
    ia = rng.choice(idx_of[a], size=min(3, len(idx_of[a])), replace=False)
    ib = rng.choice(idx_of[b], size=min(3, len(idx_of[b])), replace=False)
    hard = cls_of[a] == cls_of[b]
    for i in ia:
        for j in ib:
            diff_pairs.append((int(i), int(j)))
            hard_mask.append(hard)
hard_mask = np.array(hard_mask)

from fastembed import ImageEmbedding

fe = ImageEmbedding("Qdrant/Unicom-ViT-B-32")
vecs = np.stack(list(fe.embed(images, batch_size=16))).astype(np.float32)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9

same = np.array([float(vecs[i] @ vecs[j]) for i, j in same_pairs])
diff = np.array([float(vecs[i] @ vecs[j]) for i, j in diff_pairs])
hard = diff[hard_mask]

print(f"pairs: same={len(same)} diff={len(diff)} hard={len(hard)}")
print(f"{'thr':>5} | {'TPR(same>=t)':>12} | {'FMR(diff>=t)':>12} | {'hardFMR':>8}")
rows = []
for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    tpr = float((same >= t).mean())
    fmr = float((diff >= t).mean())
    hfmr = float((hard >= t).mean())
    rows.append(dict(thr=t, tpr=round(tpr, 3), fmr=round(fmr, 4), hard_fmr=round(hfmr, 4)))
    print(f"{t:5.2f} | {tpr:12.3f} | {fmr:12.4f} | {hfmr:8.4f}")

(CROPS / "unicom_thresholds.json").write_text(json.dumps(rows, indent=1))
print("saved ->", CROPS / "unicom_thresholds.json")
