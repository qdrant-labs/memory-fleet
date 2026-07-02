# Fleet Memory

Shared object memory on Qdrant Edge — a demo. Webcam → class-agnostic
detection → **names come from vector search only** (humans teach unknowns) →
local Edge shards sync with a central fleet collection in Qdrant Cloud so
every device knows what any device learned. The repo is named `hive-mind`
for historical reasons; the demo's display name is **Fleet Memory** and the
shared server is "the fleet" everywhere in code and UI.

**Read `PLAN.md` before doing anything.** It is the build contract: decided
semantics, architecture, spike evidence. If reality contradicts it, stop and
flag — don't silently patch. Deviations decided since the plan are listed
below and win over the plan text.

**Keep this file clean and updated.** Whenever a decision is made, a gotcha is
earned, or the project state changes, update the relevant section here in the
same working session — this file is the project's memory between sessions.

## Current state (2026-07-02)

All six PLAN.md §8 phases are built and green, plus five rounds of feedback
from Dylan's live use. The app works end-to-end: `make run` →
http://127.0.0.1:8765, live teach/recognize with picture-pill suggestions,
hybrid search, curated push/pull against Dylan's Qdrant Cloud cluster
(`.env`; the `fleet` collection is auto-created), 300k-memory scale stunt
(S key, after `make demo-scale`). 2026-07-02: video decoupled from detection
(25 fps video / 8 Hz detect) and a UI pass (Edge band, map zoom, contrast).
Not yet done: golden demo state (`make demo-save` after a real teaching
session), README demo-script rehearsal, second-laptop test, and rehearsing
the two approved demo beats (Wi-Fi kill + second unit, below).

## Layout

- `fleetmemory/` — `config` (env), `perception/` (detector, crops, embedder,
  cadence), `memory/` (store, matcher, core = single-threaded heart),
  `sync/` (fleet client + manager), `server/` (FastAPI + WS + pipeline).
- `static/` — vanilla-JS UI (no build step), Qdrant-branded mission-control
  theme; brand SVGs in `static/brand/`.
- `tests/gate` (deterministic, mock geometry, real Edge shards, run before
  every commit), `tests/sync` (runs against the Cloud cluster in `.env`),
  `tests/smoke` + `tests/drive` (real models, local), `scripts/` (soak,
  scale, demo-check).

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
- **Guesses are picture pills**: unknown and suggest tracks carry the top-3
  nearest memories (object_id + label + score + thumb). At suggest tier a
  pill CONFIRMS that specific instance; at unknown tier it teaches that
  name; ignored entries ride along red-flagged (click = fold into the
  blocklist). YOLOE class names appear as extra teach hints — the only
  other use of detector labels, never auto-naming.
- **"Memories" = exemplar VECTORS, not points** (Dylan): HUD/metrics/searched
  all report `store.vector_count()` (lazy recount after mutations; the scale
  shard contributes 3×count by construction). "Objects" = named instances.
- **Live ingest is paced; video is decoupled** (Dylan, 2026-07-02: 8 fps
  video too choppy; unpaced MPS ran the M5 hot): a grabber thread owns the
  camera (set to 720p — 1080p drags the sensor to ~20 fps for nothing) and
  streams JPEG at ≤30 fps; the detect thread runs YOLOE at TARGET_FPS=8 on
  the latest frame and emits boxes-only messages; the client eases boxes
  between ticks (~90 ms). Measured 25 fps video / 8 Hz detect; heat profile
  unchanged. Drive mode keeps the synchronous single-thread path (boxes ride
  frame messages) so tests stay deterministic.
- **Demo script beats (Dylan, 2026-07-02)**: (1) kill Wi-Fi mid-demo —
  everything keeps working, FLEET OFFLINE pill, reconnect syncs; (2) teach
  on unit A, recognize on unit B (PLAN §4.5 — still unrehearsed). Rejected:
  TTS voice, live fleet-feed ticker (demos rarely run concurrently),
  leaderboards, glasses/robot hardware pivots.
- **UI pass (Dylan, 2026-07-02: "not very pretty, low contrast")**: the
  on-device search latency is the hero — an Edge band under the video (hero
  µs figure + latency sparkline + memories/objects) absorbs the leftover
  viewport height. Sans for prose, mono for telemetry; brighter contrast
  tokens. Memory map is points-only (labels moved to hover) with wheel zoom
  + drag pan, ⌂/double-click resets. Search results show last-seen time +
  device name — the unit IS the location (no GPS on laptops; name a unit
  after its place). Panel is "SEARCH", not "SEARCH THE MEMORY". Em dashes
  swept from UI strings (Qdrant copy rule).
- **Merge is same-kind only** (object+object or ignored+ignored) and
  preserves `kind` — a re-upsert without `kind`/`base_payload` silently
  corrupts points; every re-upsert call site must pass both.
- Ignoring a RECOGNIZED box ignores the bound OBJECT (label + views move to
  the blocklist), not a one-view phantom.
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
  completeness. No feature not in PLAN.md without asking Dylan.
- Tests are in-tree; no CI (cut 2026-07-01). Run lint + `tests/gate` +
  `tests/sync` before every commit (`tests/gate` must stay fast and
  deterministic; sync needs the Cloud `.env` and skips without it).
- Commits: subject-only, imperative, 5–10 words; commit freely for
  snapshots/rollbacks (Dylan, 2026-07-01) — but never push unasked.
