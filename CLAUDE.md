# Fleet Memory

Shared object memory on Qdrant Edge — a demo. Webcam → class-agnostic
detection → **names come from vector search only** (humans teach unknowns) →
local Edge shards sync with a central fleet collection in Qdrant Cloud so
every device knows what any device learned. The repo is `fleet-memory`
(renamed from `hive-mind`, 2026-07-02; the Python package stays
`fleetmemory`); the demo's display name is **Fleet Memory** and the shared
server is "the fleet" everywhere in code and UI.

**Keep this file clean and updated.** Whenever a decision is made, a gotcha is
earned, or the project state changes, update the relevant section here in the
same working session — this file is the project's memory between sessions.

## Current state (2026-07-02)

The app works end-to-end: `make run` → http://127.0.0.1:8765, live
teach/recognize with picture-pill suggestions, hybrid search, curated
push/pull against the Qdrant Cloud cluster in `.env` (the `fleet` collection
is auto-created), 300k-vector scale stunt (S key, after `make demo-scale`).
2026-07-02 cleanup for public release: tests, PLAN.md, and docs/spikes removed
from git (single-builder repo, demo-first); drive mode and the old shard
migrations deleted with them; comments swept to describe code, not history;
README rewritten for the community. Not yet done: golden demo state
(`make demo-save` after a real teaching session), README demo-script
rehearsal, second-laptop test, and rehearsing the two demo beats (Wi-Fi kill
mid-demo; teach on unit A / recognize on unit B).

## Layout

- `fleetmemory/` — `config` (env), `perception/` (detector, crops, embedder,
  cadence), `memory/` (store, matcher, core = single-threaded heart),
  `sync/` (fleet client + manager), `server/` (FastAPI + WS + pipeline).
- `static/` — vanilla-JS UI (no build step), Qdrant-branded mission-control
  theme; brand SVGs in `static/brand/`.
- `scripts/` — `demo_check.py` (offline model preflight), `preload_scale.py`
  (builds the stunt shard).

## Semantics (the rules the code implements)

- **The fleet is Qdrant Cloud, full stop — no Docker anywhere.** Local-first:
  unreachable fleet = "FLEET OFFLINE" pill, silent auto-reconnect, status
  events on transitions only. Cloud requires payload indexes for filtered
  scrolls (`label_key`, keyword).
- **INSTANCE identity, not name==identity**: a point is ONE physical thing
  (≤24 views of it); the label is a display name and may repeat. Teach folds
  into a same-name object only when the view is ≥ s_suggest to its rows
  (`_fold_target`); otherwise it's a new instance. Same rule at push time
  (fleet fold is similarity-gated; same-id always folds). Search/inventory
  show each instance separately — thumbnails differentiate. Scale = unlimited
  instance points, never melting-pot multivectors.
- **Any local edit clears `t_sync`**: a pushed object edited locally is dirty
  again, or the next pull's dedup would delete the edit. `MarkPushed` skips
  stamping if the point's fingerprint changed mid-push. Fleet label matching
  is case-insensitive via `label_key`.
- **Fleet objects hydrate copy-on-write**: a HUMAN teach signal (confirm,
  same-label teach fold ≥ s_suggest, archived teach) aimed at a mirror object
  reads the mirror copy and writes the merged result into the MUTABLE shard
  under the SAME id, dirty — it shadows the mirror (mutable wins ties),
  survives pull dedup, and same-id-folds back into the fleet point on the
  next push. Auto-captured views never hydrate (every unit would dirty every
  object it sees). The mirror is never written. Curation shows fleet objects'
  views read-only (no prune); per-view thumbnails exist only on the unit that
  saw them.
- **Push folds SAME-ID FIRST** (`client.get_point`) before the label lookup —
  a renamed local copy would miss `find_by_label`, and a plain upsert would
  clobber views other units folded in; the fold also carries `label`, not
  just `label_key`, so renames propagate. **Label-fold push rewrites the
  local point under the fleet id** so id-present dedup applies verbatim on
  the next pull.
- **Empty-delta pulls are skipped** (zero-byte body or tar without
  `segments/`); a genuinely corrupt pull rebuilds the mirror (it's a
  disposable replica) and re-seeds on the next pull.
- **Person/body-part suppression**: detector class names ARE consulted —
  solely to drop people/hands/faces proposals (`PERSON_WORDS` in
  `detector.py`, derived from the model's actual 4,585-class vocab: wig,
  ponytail, eyebrow, etc.). Suppression is STICKY per track: the classifier
  flickers (hair reads "hair" one frame, "wig"/"fur" the next), so a tid
  that ever looked person-like stays suppressed for its lifetime
  (`_person_tids`, cleared on tracker reset). The person check runs before
  the area band so oversized face boxes still poison their tid. Names still
  come from vector search only. MAX_AREA is tight (0.20): live desks
  produce quarter-screen phantom proposals; demo objects are hand-held
  scale.
- **Suppression is two-tier**: ≥ S_ignore hard-suppresses; ≥ S_IGNORE_SOFT
  (0.65) suppresses UNLESS a taught object outranks it. Soft-suppressed boxes
  stay faint and rescuable. Repeated ignores fold views into ONE blocklist
  entry (IGNORE_FOLD). At VIEW_CAP (24), a human view replaces the most
  redundant auto view so confirms never stop teaching.
- Ignoring a RECOGNIZED box ignores the bound OBJECT (label + views move to
  the blocklist), not a one-view phantom. Ignored items are curate-able:
  blocklist entries show in inventory with thumbnails + per-vector prune;
  "unignore" = Forget on the blocklist point.
- **Merge is same-kind only** (object+object or ignored+ignored) and
  preserves `kind` — a re-upsert without `kind`/`base_payload` silently
  corrupts points; every re-upsert call site must pass both. `upsert_object`
  takes `base_payload` so re-upserts don't wipe auxiliary payload keys
  (e.g. `sightings`, `t_seen`).
- **Departed unknowns stay teachable**: unnamed dead tracks are archived
  (cap 12) in the core; Teach/Dismiss on a dead (tid, epoch) hits the archive.
- **Guesses are picture pills**: unknown and suggest tracks carry the top-3
  nearest memories (object_id + label + score + thumb). At suggest tier a
  pill CONFIRMS that specific instance; at unknown tier it teaches that
  name; ignored entries ride along red-flagged (click = fold into the
  blocklist). YOLOE class names appear as extra teach hints — the only
  other use of detector labels, never auto-naming.
- **Hybrid search** = miniCOIL sparse + bge-small dense over LABELS
  (`memory/labels.py`), two prefetch legs fused with RRF (k=2). Edge 0.7.2
  exports Prefetch/Fusion but doesn't consume them, so the RRF step runs
  app-side. Substring pass for partial words; dense visual expansion from
  the top hit ("looks similar"). Dense leg floor 0.6 (bge scores everything).
  Fallback: no LabelEmbedder → on-device BM25, same sparse field. Label
  semantic search ≠ image-text search — Unicom has no text tower.
- **"Memories" = exemplar VECTORS, not points**: HUD/metrics/searched all
  report `store.vector_count()` (lazy recount after mutations; the scale
  shard contributes 3×count by construction). "Objects" = named instances.
- **Camera lifecycle**: capture runs only while a browser is connected AND
  the UI toggle is on. The Python process owns the camera (it IS the edge
  device); the browser is a dashboard.
- **Live ingest is paced; video is decoupled**: a grabber thread owns the
  camera (720p — 1080p drags the sensor to ~20 fps for nothing) and streams
  JPEG at ≤30 fps; the detect thread runs YOLOE at TARGET_FPS=8 on the
  latest frame and emits boxes-only messages; the client eases boxes between
  ticks (~90 ms). Measured 25 fps video / 8 Hz detect. Detection cadence is
  a live TUNING dial (`target_fps` cmd); the ticker shows video fps only.
- **Demo beats**: (1) kill Wi-Fi mid-demo — everything keeps working, FLEET
  OFFLINE pill, reconnect syncs; (2) teach on unit A, recognize on unit B.
  Rejected ideas: TTS voice, live fleet-feed ticker, leaderboards,
  glasses/robot hardware pivots.
- **UI**: sans for prose, mono for telemetry. Latency lives in the rail's
  RECOGNITION QUERY panel — an "Edge band" under the video was tried and
  reverted (camera must be full width). Memory map is points-only (labels on
  hover) with wheel zoom + drag pan, ⌂/double-click resets. Search results
  show last-seen time + device name — the unit IS the location (name a unit
  after its place). No em dashes in UI strings (Qdrant copy rule). Feed
  click hit-region includes the chip strip + a pad and the union of
  eased/latest box — at 25 fps video a moving object visibly outruns its
  8 Hz box, so tight hit-tests miss.

## Environment gotchas (hard-won)

- **macOS on Apple Silicon only** (decided 2026-07-02: Windows support
  skipped for now — too much surface). YOLOE on MPS, Unicom on CPU.
- **Port 8000 is usually taken** on this machine — default is 8765, never
  assume 8000 is free.
- `qdrant-edge-py` is pinned (beta; API drifts between minors). Bump with
  care — exercise a full push/pull cycle against the Cloud cluster after.
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
  must stay wrapped in `objc.autorelease_pool()`. When measuring memory, use
  current RSS via `ps`, never `ru_maxrss` — the high-water mark hides creep.
- Ultralytics tracking needs `lap` pinned explicitly (`YOLO_AUTOINSTALL`
  is off). Model weights: `yoloe-11l-seg-pf.pt` in repo root (gitignored);
  fastembed caches under the system temp dir.
- UI: never re-render a drawer per frame — it steals input focus and eats
  clicks. Gate re-renders on a content signature (see `renderUnknowns`).
- **GIL vs the camera**: ultralytics' Python-side glue holds the GIL in
  chunks; at the default 5 ms switch interval the grabber thread misses
  frames (AVFoundation keeps only the latest) and video halves to ~15 fps.
  `pipeline.py` sets `sys.setswitchinterval(0.002)` — measured 25 fps.
  Related: pace gates must be slot accumulators, not `now` stamps — a
  stamped 24 fps gate against a 29 fps camera beats down to every other
  frame (~15 fps).
- Headless UI screenshot without deps: `"/Applications/Google
  Chrome.app/Contents/MacOS/Google Chrome" --headless=new --screenshot=…
  --window-size=1512,900 --virtual-time-budget=9000 http://127.0.0.1:8765`.

## Working rules (repo-specific)

- Qdrant is a **vector search engine** — never "vector database" in any
  user-visible string, doc, or commit.
- Demo-first quality bar: what an audience sees in 3 minutes wins over
  completeness. No new features without asking Dylan.
- **No tests in-tree** (removed 2026-07-02 for the public release;
  single-builder repo). Before every commit: `make lint`, boot the app
  (`make run`, load the UI, teach/recognize one object), and for sync
  changes exercise a push/pull against the Cloud cluster in `.env`.
- Comments describe what the code does and its constraints — never project
  history, names, dates, or decision narration. That context lives here.
- Commits: subject-only, imperative, 5–10 words; commit freely for
  snapshots/rollbacks — but never push unasked.
