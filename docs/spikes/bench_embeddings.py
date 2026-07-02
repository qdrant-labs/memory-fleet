"""Instance-separation benchmark across embedding models.

Self-verifying ground truth from tracked crops (extract_crops.py):
  SAME-item pairs = crop pairs within one track id (near-pure by construction).
  DIFF-item pairs = crop pairs across two tracks that COEXIST in >=1 sampled
                    frame of the same clip -> provably distinct physical items.
  HARD-diff pairs = the subset of DIFF pairs where YOLOE gave both tracks the
                    same class (e.g. two different chairs in frame together) —
                    the regime that killed v1 re-id.

Metrics per model: AUC (same vs diff), median/p10 same-sim, p90/p99 diff-sim,
recall of same-item pairs at 1% false-match rate, same for the hard subset,
and per-crop embed latency.
"""

import json
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

CROPS = Path(sys.argv[1])
manifest = json.loads((CROPS / "manifest.json").read_text())

tracks = defaultdict(list)   # (clip,tid) -> [manifest rows]
for m in manifest:
    tracks[(m["clip"], m["tid"])].append(m)
tracks = {k: sorted(v, key=lambda m: m["frame"]) for k, v in tracks.items()
          if len(v) >= 4}

frames_of = {k: {m["frame"] for m in v} for k, v in tracks.items()}
cls_of = {k: v[0]["cls"] for k, v in tracks.items()}

# Load all crops once, remember index ranges per track
paths, owner = [], []
for k, rows in tracks.items():
    for m in rows:
        paths.append(CROPS / m["file"])
        owner.append(k)
images = [Image.open(p).convert("RGB") for p in paths]
idx_of = defaultdict(list)
for i, k in enumerate(owner):
    idx_of[k].append(i)

# Pair construction (index pairs, fixed across models)
same_pairs, diff_pairs, hard_mask = [], [], []
keys = sorted(tracks)
for k in keys:
    same_pairs += list(combinations(idx_of[k], 2))

rng = np.random.default_rng(11)
for a, b in combinations(keys, 2):
    if a[0] != b[0] or not (frames_of[a] & frames_of[b]):
        continue  # different clip, or never coexist -> can't certify distinct
    ia = rng.choice(idx_of[a], size=min(3, len(idx_of[a])), replace=False)
    ib = rng.choice(idx_of[b], size=min(3, len(idx_of[b])), replace=False)
    hard = cls_of[a] == cls_of[b]
    for i in ia:
        for j in ib:
            diff_pairs.append((int(i), int(j)))
            hard_mask.append(hard)
hard_mask = np.array(hard_mask)
print(f"{len(images)} crops | {len(tracks)} tracks | "
      f"{len(same_pairs)} same-pairs, {len(diff_pairs)} diff-pairs "
      f"({int(hard_mask.sum())} hard same-class)")


def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    order = allv.argsort().argsort()
    return (order[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))


def eval_model(name, embed_fn):
    t0 = time.perf_counter()
    vecs = np.asarray(embed_fn(images), dtype=np.float32)
    ms = (time.perf_counter() - t0) * 1000 / len(images)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9

    same = np.array([float(vecs[i] @ vecs[j]) for i, j in same_pairs])
    diff = np.array([float(vecs[i] @ vecs[j]) for i, j in diff_pairs])
    hard = diff[hard_mask]

    thr = np.quantile(diff, 0.99)
    thr_h = np.quantile(hard, 0.99)
    out = dict(
        model=name, dim=int(vecs.shape[1]), ms=round(ms, 1),
        auc=round(float(auc(same, diff)), 3),
        auc_hard=round(float(auc(same, hard)), 3),
        same_med=round(float(np.median(same)), 3),
        same_p10=round(float(np.quantile(same, .1)), 3),
        diff_p90=round(float(np.quantile(diff, .9)), 3),
        diff_p99=round(float(thr), 3),
        hard_p99=round(float(thr_h), 3),
        recall_1fmr=round(float((same >= thr).mean()), 3),
        recall_1fmr_hard=round(float((same >= thr_h).mean()), 3),
    )
    print(f"{name:24s} dim={out['dim']:4d} {out['ms']:7.1f} ms | "
          f"AUC {out['auc']:.3f} (hard {out['auc_hard']:.3f}) | "
          f"same med {out['same_med']:.3f} p10 {out['same_p10']:.3f} | "
          f"diff p99 {out['diff_p99']:.3f} hard p99 {out['hard_p99']:.3f} | "
          f"R@1%FMR {out['recall_1fmr']:.2f} hard {out['recall_1fmr_hard']:.2f}")
    return out


results = []

# --- v1 baseline: SigLIP2 ONNX (CPU) ---
sys.path.insert(0, "/Users/dylanc/Documents/GitHub/edge-mission-control")
from app.encoder import get_encoder  # noqa: E402
enc = get_encoder()
enc.load()
results.append(eval_model("siglip2-base (v1)",
                          lambda ims: np.stack([enc.encode_image(im) for im in ims])))

# --- fastembed models (ONNX CPU) ---
from fastembed import ImageEmbedding  # noqa: E402
for fe_name in ["Qdrant/clip-ViT-B-32-vision", "Qdrant/Unicom-ViT-B-32", "Qdrant/Unicom-ViT-B-16"]:
    fe = ImageEmbedding(fe_name)
    results.append(eval_model(fe_name.split("/")[1],
                              lambda ims, fe=fe: np.stack(list(fe.embed(ims, batch_size=16)))))

# --- DINOv2-small CLS (torch MPS) ---
import torch  # noqa: E402
from transformers import AutoImageProcessor, AutoModel  # noqa: E402
dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small", use_fast=True)
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()


def dino_embed(ims):
    out = []
    with torch.inference_mode():
        for k in range(0, len(ims), 16):
            batch = proc(images=ims[k:k + 16], return_tensors="pt").to(dev)
            out.append(dino(**batch).last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(out)


results.append(eval_model("dinov2-small", dino_embed))

(CROPS / "bench_results.json").write_text(json.dumps(results, indent=1))
print("saved ->", CROPS / "bench_results.json")
