# Fleet Memory

Shared object memory on Qdrant Edge — a demo. Webcam → class-agnostic
detection → **names come from vector search only** (humans teach unknowns) →
local Edge shards sync with a central fleet collection (Qdrant Cloud or
Docker) so every device knows what any device learned. The repo is named
`hive-mind` for historical reasons; the demo's display name is **Fleet
Memory** and the shared server is "the fleet" everywhere in code and UI.

**Read `PLAN.md` before doing anything.** It is the build contract: decided
semantics, architecture, spike evidence. If reality contradicts it, stop and
flag — don't silently patch. Deviations decided since the plan are listed
below and win over the plan text.

**Keep this file clean and updated.** Whenever a decision is made, a gotcha is
earned, or the project state changes, update the relevant section here in the
same working session — this file is the project's memory between sessions.

## Current state (2026-07-01)

All six PLAN.md §8 phases are built and green. The app works end-to-end:
`make run` → http://127.0.0.1:8765, live teach/recognize, curated push/pull
against Dylan's Qdrant Cloud cluster (his `.env`, collection `fleet`), 100k
scale stunt (S key, after `make demo-scale`). Two feedback rounds from live
use are folded in. Not yet done: golden demo state (`make demo-save` after a
real teaching session), README demo-script rehearsal, second-laptop test.

## Layout

- `fleetmemory/` — `config` (env), `perception/` (detector, crops, embedder,
  cadence), `memory/` (store, matcher, core = single-threaded heart),
  `sync/` (fleet client + manager), `server/` (FastAPI + WS + pipeline).
- `static/` — vanilla-JS UI (no build step), Qdrant-branded mission-control
  theme; brand SVGs in `static/brand/`.
- `tests/gate` (deterministic, mock geometry, real Edge shards, run before
  every commit), `tests/sync` (needs `make fleet-up`), `tests/smoke` +
  `tests/drive` (real models, local), `scripts/` (soak, scale, demo-check).

## Decisions since PLAN.md (these override the plan)

- **No CI** (Dylan): single-builder repo. Run lint + `tests/gate` +
  `tests/sync` before every commit instead.
- **The fleet is Qdrant Cloud, full stop — no Docker anywhere** (Dylan,
  2026-07-01). Local-first: unreachable fleet = "FLEET OFFLINE" pill, silent
  auto-reconnect, status events on transitions only. `tests/sync` runs
  against the Cloud cluster in `.env` (throwaway `fm-test-*` collections,
  deleted in teardown; skipped when no `.env`). Cloud requires payload
  indexes for filtered scrolls (`label_key`, keyword).
- **Any local edit clears `t_sync`** (Codex review, 2026-07-01): a pushed
  object edited locally is dirty again, or the next pull's dedup would
  delete the edit. `MarkPushed` skips stamping if the point's fingerprint
  changed mid-push. Fleet label matching is case-insensitive via `label_key`.
- **INSTANCE identity, not name==identity** (Dylan, 2026-07-01 — REPLACES
  PLAN §2's rule): a point is ONE physical thing (≤24 views of it); the
  label is a display name and may repeat. Teach folds into a same-name
  object only when the view is ≥ s_suggest to its rows (`_fold_target`);
  otherwise it's a new instance. Same rule at push time (fleet fold is
  similarity-gated; same-id always folds). Rename conflicts no longer
  exist. Search/inventory show each instance separately — thumbnails
  differentiate. Hive scale = unlimited instance points, never
  melting-pot multivectors.
- **Person/body-part suppression** (Dylan): detector class names ARE consulted
  — solely to drop people/hands/faces proposals (`PERSON_WORDS` in
  `detector.py`). Names still come from vector search only.
- **MAX_AREA tightened** 0.55 → 0.20: live desks produced quarter-screen
  phantom proposals; demo objects are hand-held scale.
- **Departed unknowns stay teachable**: unnamed dead tracks are archived
  (cap 12) in the core; Teach/Dismiss on a dead (tid, epoch) hits the archive.
- **Ignored items are curate-able**: blocklist entries show in inventory with
  thumbnails + per-vector prune; "unignore" = Forget on the blocklist point.
- **Hybrid search (Dylan, "we're Qdrant")** = miniCOIL sparse + bge-small
  dense over LABELS (`memory/labels.py`), two prefetch legs fused with RRF
  (k=2, Qdrant's default). Edge 0.7.2 exports Prefetch/Fusion but doesn't
  consume them, so the RRF step runs app-side. Substring pass for partial
  words; dense visual expansion from the top hit ("looks similar"). Dense
  leg floor 0.6 (bge scores everything). Fallback: no LabelEmbedder (gate,
  sync tests) → on-device BM25, same sparse field. Schema: `label_dense`
  384-d added; old shards migrate in place (`create_dense_vector`) and old
  labels re-embed at boot (`label_v` marker). Note: label semantic search ≠
  image-text search — Unicom still has no text tower (PLAN §3.3 final).
- **Sightings**: core counts recognitions per object (`sightings`, `t_seen`
  payload on mutable objects, session-only for fleet ones). `upsert_object`
  takes `base_payload` so re-upserts don't wipe auxiliary payload keys —
  every re-upsert call site must pass it.
- **Camera lifecycle**: capture runs only while a browser is connected AND
  the UI toggle is on. The Python process owns the camera (it IS the edge
  device); the browser is a dashboard.
- **Suppression is two-tier** (Dylan, 2026-07-01: ignored doors/hair kept
  returning): ≥ S_ignore hard-suppresses; ≥ S_IGNORE_SOFT (0.65) suppresses
  UNLESS a taught object outranks it. Soft-suppressed boxes stay faint and
  rescuable. Repeated ignores fold views into ONE blocklist entry
  (IGNORE_FOLD). VIEW_CAP raised 12 → 24; at cap, a human view replaces the
  most redundant auto view so confirms never stop teaching.
- **Unknowns carry guesses**: top-3 nearest memories (teach-fold one-click)
  + YOLOE class names (the ONLY other use of detector labels — hints, never
  auto-naming).
- **Label-fold push rewrites the local point under the fleet id** so the
  §3.3 id-present dedup applies verbatim on the next pull.
- **Empty-delta pulls are skipped** (zero-byte body or tar without
  `segments/`); a genuinely corrupt pull rebuilds the mirror (it's a
  disposable replica) and re-seeds on the next pull.

## Environment gotchas (hard-won)

- **Port 8000 is usually taken** on this machine — default is 8765, never
  assume 8000 is free.
- `qdrant-edge-py` is pinned (beta; API drifts between minors). The sync test
  is the canary when bumping.
- Edge API: `count()` takes a `CountRequest`; snapshot downloads must use
  `requests` `iter_content` (chunked — `r.raw` corrupts the tar);
  `EdgeShard.create` needs the directory to already exist;
  `UpdateOperation.set_payload(payload=…, point_ids=…)` is kwargs-only;
  `scroll` returns `(records, offset)`; `IsNullCondition` doesn't take
  `is_null=` (we filter payloads in-process).
- Snapshot endpoints: `GET /collections/{c}/shards/{id}/snapshot`,
  `POST .../snapshot/partial/create` (send `snapshot_manifest()` as JSON;
  Cloud auth via `api-key` header). An empty Edge-created shard CAN seed
  straight from a partial snapshot — seed and pull share one code path.
- **torch-MPS leaks ~80 MB/min** in pure-Python inference loops (autoreleased
  Metal objects, invisible to `torch.mps` accounting). Every detector call
  must stay wrapped in `objc.autorelease_pool()`
  (`docs/spikes/spike_mps_leak.py`). When measuring memory, use current RSS
  via `ps`, never `ru_maxrss` — the high-water mark hides creep.
- Ultralytics tracking needs `lap` pinned explicitly (`YOLO_AUTOINSTALL`
  is off). Model weights: `yoloe-11l-seg-pf.pt` in repo root (gitignored);
  fastembed caches under the system temp dir. YOLOE on MPS, Unicom on CPU.
- UI: never re-render a drawer per frame — it steals input focus and eats
  clicks. Gate re-renders on a content signature (see `renderUnknowns`).

## Working rules (repo-specific)

- Qdrant is a **vector search engine** — never "vector database" in any
  user-visible string, doc, or commit.
- Demo-first quality bar: what an audience sees in 3 minutes wins over
  completeness. No feature not in PLAN.md without asking Dylan.
- Tests are in-tree; no CI (cut 2026-07-01). Run lint + `tests/gate` +
  `tests/sync` before every commit (`tests/gate` must stay fast and
  deterministic; sync needs `make fleet-up`).
- Commits: subject-only, imperative, 5–10 words; commit freely for
  snapshots/rollbacks (Dylan, 2026-07-01) — but never push unasked.
