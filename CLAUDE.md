# Fleet Memory

Shared object memory on Qdrant Edge — a demo. Webcam → class-agnostic
detection → **names come from vector search only** (humans teach unknowns) →
local Edge shards sync with a central fleet collection (Qdrant Cloud or
Docker) so every device knows what any device learned. The repo is named
`hive-mind` for historical reasons; the demo's display name is **Fleet
Memory** and the shared server is "the fleet" everywhere in code and UI.

**Read `PLAN.md` before doing anything.** It is the build contract: decided
semantics, architecture, spike evidence, phases. If reality contradicts it,
stop and flag — don't silently patch.

## Context you don't otherwise have

- This repo was planned in a session inside `../edge-mission-control` (v1 of
  this demo, branch `live-camera`). Consult it as *reference*, port ideas —
  never copy its registry/threading design (PLAN.md §12 lists the lessons).
  Its `PROJECT_STATE.md` is the v1 autopsy.
- Spike scripts + measured results live in `docs/spikes/` (sync round-trip,
  detector shootout, embedding benchmark, 100k-scale latency). Rerun-able.
- Test footage: `../edge-mission-control/footage/*.mp4`.

## Environment gotchas (hard-won)

- **Port 8000 is usually taken** by the old demo's uvicorn on this machine —
  default this app to another port and never assume 8000 is free.
- `qdrant-edge-py` is pinned (beta; API drifts between minors). The sync test
  is the canary when bumping.
- Edge API: `count()` takes a `CountRequest`; snapshot downloads must use
  `requests` `iter_content` (chunked transfer — `r.raw` corrupts the tar).
- Snapshot endpoints: `GET /collections/{c}/shards/{id}/snapshot`,
  `POST .../snapshot/partial/create` (send `snapshot_manifest()` as JSON).
- Model weights: ultralytics auto-downloads `yoloe-11l-seg-pf.pt` to the repo
  root (gitignore it); fastembed caches under the system temp dir.
- ONNX/torch runs: YOLOE on MPS, embedders per PLAN.md §9.2.
- More Edge 0.7.2 quirks (found during the build): `EdgeShard.create` needs the
  directory to already exist; `UpdateOperation.set_payload(payload=…,
  point_ids=…)` is kwargs-only; `IsNullCondition` doesn't take `is_null=` (we
  filter payloads in-process instead); `scroll` returns `(records, offset)`.
- Ultralytics tracking needs `lap` pinned explicitly — we set
  `YOLO_AUTOINSTALL=false` so it can't pip-install at runtime.
- **torch-MPS leaks ~80 MB/min** in pure-Python inference loops (autoreleased
  Metal objects, invisible to `torch.mps` accounting). Every detector call must
  stay wrapped in `objc.autorelease_pool()` (`docs/spikes/spike_mps_leak.py`).
  When measuring memory, use current RSS via `ps`, never `ru_maxrss` — the
  high-water mark hides creep under early spikes.
- An empty Edge-created shard CAN seed straight from a partial snapshot
  (manifest of empty shard → `partial/create` → `update_from_snapshot`), so
  seed and pull share one code path — verified in `tests/sync/`.

## Working rules (repo-specific)

- Qdrant is a **vector search engine** — never "vector database" in any
  user-visible string, doc, or commit.
- Demo-first quality bar: what an audience sees in 3 minutes wins over
  completeness. No feature not in PLAN.md without asking Dylan.
- Tests are in-tree; no CI (cut 2026-07-01 — single-builder repo). Run lint +
  `tests/gate` + `tests/sync` before every commit instead (`tests/gate` must
  stay fast and deterministic; sync needs `make fleet-up`).
