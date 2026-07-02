# Fleet Memory — Build Plan

> A collective object memory for edge devices, built on Qdrant Edge.
> Written 2026-07-01 from a full review of `edge-mission-control` (v1), the
> decisions Dylan locked in, and measured spikes. This document is the contract
> for the build session: follow it, and flag (don't silently patch) anything
> that turns out wrong.

---

## 1. What this is

A live demo for **Qdrant Edge** showing two things:

1. **Speed** — vector search running *inside* the app process, on-device,
   answering recognition queries in microseconds-to-milliseconds, visible on
   screen with every frame.
2. **Sync / collective memory** — what one device learns, every device knows.
   Local shards synchronize with a central Qdrant server (the shared **fleet memory**, "the fleet") using
   Edge's native snapshot sync; the demo laptop stands in for robots and
   wearables.

**The loop:** a webcam watches a desk. An object detector proposes regions —
*it assigns no labels and has no vocabulary*. Each stable region is embedded
and looked up in Qdrant. A match shows the remembered name plus the live query
latency. No match shows "unknown — teach me"; a human names it once, and it's
remembered — locally at first, fleet-wide after curation.

**The pitch line:** *the model only sees shapes; the memory knows what things
are — and the memory is shared.*

### Non-goals

- Not a product; a demo. Optimize for what an audience sees in 3 minutes.
- No auto-labeling of any kind. Names enter the system through humans only.
- No multi-user isolation, auth, or privacy story beyond "sync is explicit."

---

## 2. Decisions already made (Dylan, 2026-07-01 session)

| Decision | Detail |
|---|---|
| Labels from vector search only | Detector is class-agnostic. New items require a human name. |
| Two-tier matching | High sim → recognized. Medium sim → *suggest* the nearest label, human confirms with one tap. Below → unknown. Thresholds live-tunable. |
| Fleet sync is opt-in | No `QDRANT_URL`/`QDRANT_API_KEY` in `.env` → fully local demo, no sync UI beyond a "fleet offline" hint. With env → pull/seed available, push gated behind curation. |
| Curation-gated push | Nothing reaches the fleet uncurated. The curation panel is the gate; per-view pruning (drop a single bad exemplar from an object) is a required feature, carried over from v1. |
| Keep from v1 scope | Text search over memory (BM25 on human labels), 2D memory map, curation panel. |
| Cut from v1 | Florence-2 captioning, caption BM25, whole-frame scene memory, class-level suppression (no classes exist), auto-label correction machinery. |
| Name | **Fleet Memory** (decided 2026-07-01). Subtitle: "shared object memory on Qdrant Edge". Repo stays `hive-mind`; package `fleetmemory/`; the central server is "the fleet". |

### Standing v1 policies that carry over

- **name == identity** for human corrections: teaching a name that already
  exists folds the item into that object. Kills duplicates; accepted cost:
  two intentionally-distinct items can't share a name (use "mug A"/"mug B").
- **Masked crops**: flatten background to neutral gray inside the segmentation
  polygon before embedding. This is what makes recognition survive background
  and hand changes (v1-proven).
- **Asymmetric suppression UX**: a false "ignored" is visible and rescuable;
  a false "tracked" is one tap to ignore.

---

## 3. Architecture

### 3.1 Layers (own package each, no cross-imports except downward)

```
fleetmemory/
  perception/   detector (class-agnostic boxes+masks), tracker glue,
                crop+mask, embedder. Pure: frame in -> proposals out.
  memory/       the core. Two Edge shards (mutable + immutable mirror),
                matcher (two-tier), object records, verbs (teach/confirm/
                reject/merge/forget/ignore/prune), event emission.
  sync/         fleet client: curated push (upload queue -> qdrant-client
                upsert), partial-snapshot pull, seed-on-boot. All .env-gated.
  server/       FastAPI + one WebSocket; serializes UI commands into the
                core's queue; broadcasts events. No logic.
  ui/           static vanilla JS: live overlay + latency HUD, unknowns
                queue, teach popover, inventory, curation panel, map,
                search. No build step.
```

### 3.2 Single-threaded core (v1's #1 lesson)

All memory mutations happen on **one thread** consuming a command queue:

```
ingest tick (proposals+embeddings) ─┐
UI verbs (teach/confirm/merge/...) ─┼──> queue ──> core thread ──> events out
sync events (pull complete, ...)   ─┘
```

Embedding/detection run in worker threads (they're pure), but their *results*
enter the core as messages. No locks. v1 spent days on lock races that this
design makes impossible by construction.

**Staleness is still real** (queues fix races, not time): a worker result or
UI command can arrive after its track died or re-bound. Every track carries an
**incarnation id** (`tid`, `epoch`); every message referencing a track carries
the incarnation it saw; the core drops mismatches. One rule, applied at the
single place messages are consumed — v1 needed the same guard scattered
across eight methods under locks.

### 3.3 Storage — Qdrant Edge, two shards (the documented sync pattern)

Verified working end-to-end in the sync spike (§9.3): server 1.18.2 ↔ Edge 0.7.2.

- **`mutable/`** — everything taught locally. Also the only shard written
  during offline demos.
- **`immutable/`** — a mirror of the fleet collection, updated by **native
  partial snapshots** (`snapshot_manifest` → POST
  `/collections/fleet/shards/0/snapshot/partial/create` → `update_from_snapshot`).
- Recognition queries **fan out to both shards**, results merged, deduped by
  point id (mutable wins ties).
- Dedup after a pull — **stricter than the docs' pattern.** The docs dual-write
  *everything* and delete mutable points by a blanket `t_sync <= pull_time`
  filter; our push is *curated*, so unpushed local objects must survive. Rule:
  delete a mutable point only when (a) its `t_sync` was stamped at push AND
  (b) its id is now present in the freshly-pulled immutable mirror. Never by
  bare timestamp.

**Point schema** (one point = one remembered object; identical on Edge and the fleet collection):

- id: UUID (globally unique across devices — fleet requirement)
- payload `kind`: `"object"` | `"ignored"` (keyword-indexed; blocklist entries
  are points too, **local-only — never pushed to the fleet**)
- vector `exemplars`: **multivector, MAX_SIM comparator** — one row per stored
  view (cap ~12, diversity-gated like v1). Recognition = one native MAX_SIM query.
- vector `label` (sparse, BM25/IDF): the human label, embedded on-device via
  `qdrant_edge.Bm25` — powers text search with zero extra models.
- payload: `label`, `device`, `event`, `t_created`, `t_sync`,
  `thumb` (ONE tiny portrait JPEG base64, so pulled objects render),
  `views` row-aligned metadata: `[{view_id, human: bool}, ...]`,
  `neg` (negative exemplar vectors, app-side veto — v2 may move to FormulaQuery).
- **Per-view thumbnails live on local disk** (keyed by `view_id`), like v1 —
  they exist for the local prune UI only. Shipping 12 base64 images inside
  every synced point would bloat each snapshot pull for a feature the fleet
  never renders.

**No separate `vision` portrait vector** (v1 had one for dense text search and
as a stable anchor). Text search is BM25-on-labels; the stable anchor is the
first human-taught row (provenance-marked). Resolved by §9.2: Unicom has no
text tower, so dense text search is not coming back — this is final, not
conditional.

### 3.4 Recognition flow (per stable track)

```
proposal stable for K frames (K≈3)
  └─ embed masked crop (worker) ──> core:
       MAX_SIM query on both shards (this latency is the HUD number)
         ├─ ignored-match ≥ S_ignore        → faint box, never stored
         ├─ best ≥ S_same                   → recognized: bind, show name+score
         │                                    (accrete view if diverse enough)
         ├─ best ≥ S_suggest (< S_same)     → "looks like «label» — same?"
         │                                    [yes → confirm-teach] [no → negative + unknown]
         └─ else                            → unknown: outline + unknowns queue
```

- Tracker: whatever ships with the chosen detector via ultralytics `track()`
  (BoT-SORT), used ONLY for frame-to-frame continuity — never for identity.
  Re-embedding+re-query happens on a slow cadence per track (~every 2s) so a
  wrong early bind self-corrects; every query updates the HUD.
- Negatives: per-object "not me" vectors vetoing a match (v1-proven); an
  app-side rerank of the top-k with the veto + human-provenance bonus. Small,
  bounded, and this time it's the *whole* rerank (no color signatures, no
  class gates — they existed to patch detector-label problems we no longer have).
- Thresholds `S_same`/`S_suggest`/`S_ignore` start from the embedding spike's
  measured operating points (§9.2) and stay live-tunable in the UI.

### 3.5 The verbs (complete list — v1's 12 collapse to 8)

| Verb | From | Effect |
|---|---|---|
| `teach(track, label)` | popover | new object (or fold into existing same-label object); triggers a ~3 s multi-angle capture burst (§4 step 2) |
| `confirm(track, object)` | suggestion chip | human-vouched view accretes to object |
| `reject(track, object)` | suggestion chip | negative on object, track stays unknown |
| `ignore(track\|object)` | popover/panel | blocklist entry (own exemplars); never auto-stored again |
| `forget(object)` | panel | delete outright (may be re-taught later) |
| `merge(a, b)` | panel | fold duplicates (name==identity makes this rare) |
| `rename(object, label)` | panel | relabel a local object; renaming onto an existing label prompts a merge |
| `prune(object, view_id)` | panel | drop one exemplar row (+ its thumb) |

Two details v1 paid for and this table bakes in: `reject` is available on
**recognized boxes too** (click a bound box → "not «label»" → negative +
back to unknown) — a false high-confidence match must be recoverable in one
tap, not only at suggest tier. And every exemplar row carries a stable
`view_id` in its row-aligned metadata, so `prune` can't hit the wrong row
when views accrete mid-curation (v1's row-shift hazard, solved by
identity-by-thumbnail hacks there).

All verbs are one queue message each; all are covered by the deterministic gate.

**Verbs vs fleet-pulled objects (the immutable mirror is read-only):**
`reject` on a fleet object keeps the negative **in core memory only** (a
session-scoped `fleet_point_id → [neg vecs]` map consulted at rerank). Not persisted:
a rejection that matters across restarts will simply be re-taught in one tap,
and this deletes an entire persistence mechanism (an earlier draft had
`kind="shadow"` points — cut as over-engineering for a demo). `ignore` works
normally (the blocklist is local). `merge`/`prune`/`forget`/`rename` on fleet
objects are **out of scope** — fleet content is edited by pushing a better
version from curation, not in place.

### 3.6 Sync lifecycle

```
boot:  .env has fleet?  ── no ──> local mode (mutable shard only)
        └─ yes ─> ensure collection (schema §3.3) ─> full-snapshot seed or
                  partial-snapshot catch-up into immutable/
loop:  partial-snapshot pull every ~30s (and on-demand "pull now" button)
push:  ONLY from curation panel: selected objects -> qdrant-client upsert
       (marks t_sync; next pull dedups them out of mutable/)
```

Threading of the pull: download + unpack happen on the sync worker;
`update_from_snapshot` is applied **as a queued core message** (partial
snapshots at demo scale measured in the hundreds of KiB — a brief pause, not
a freeze). If phase 5 measures otherwise, escalate to building the updated
shard in a second directory off-core and swapping atomically — decided by
measurement, not now.

- Fleet target is any Qdrant server: Qdrant Cloud cluster (primary) or the
  bundled `docker-compose.yml` (offline/conference fallback). Same code path.
- Push-time dedup is just **name==identity extended to the fleet**: pushing an
  object whose label already exists there folds its views into the existing
  fleet point (one upsert) instead of creating a sibling. No modal, no
  similarity prompt — the standing policy already accepts the cost (two
  distinct items can't share a name; use distinct names). An earlier draft
  had a "fleet already knows a mug — merge or push anyway?" dialog; cut as
  over-engineering that second-guessed a decided policy.
- **✅ Verified on Dylan's Qdrant Cloud cluster (2026-07-01, v1.17.1):** both
  `GET /collections/{c}/shards/{id}/snapshot` and
  `POST .../snapshot/partial/create` work end-to-end into an Edge shard with
  MAX_SIM intact (`docs/spikes/spike_sync_cloud.py`). Cloud is the primary
  fleet target; Docker remains the offline fallback.

---

## 4. The demo (what the audience sees)

1. **Cold open, local:** camera on a desk of objects. Boxes appear with names
   and a per-query latency readout ("recognized `stapler` · 0.4 ms · 3,412
   memories searched"). Unknown items carry a quiet "?".
2. **Teach:** pick an unknown, type "my badge", then a ~3-second **capture
   burst** — a progress ring while you rotate the item, harvesting diverse
   views into the multivector. (One view does NOT generalize to far angles —
   same-item p10 is 0.47, §9.2 — the burst is what makes "now it knows it
   from any side" true, and it's good theater.)
3. **Show the speed:** the HUD's running strip of recognition-query
   latencies — sub-millisecond at taught scale, a few ms at stunt scale
   (claim exactly what §9.4 measured).
4. **Scale stunt:** "what if this robot had been running for a year?" — swap
   in the **prebuilt** 100k-memory shard (`make demo-scale`; building it live
   takes ~1 min, so it's staged, swapped on a keypress); latency HUD barely
   moves (numbers in §9.4).
5. **The fleet:** open curation, review today's teachings (prune a bad view,
   merge a dup), push. Second device (or relaunched instance) pulls — and
   recognizes everything taught on the first. *"Every robot you ship knows
   what any of them ever learned."*
6. Optional persistent-fleet kicker at events: the demo recognizes objects
   taught at a previous conference. **A bonus beat, not an acceptance
   criterion** — never rehearse the demo to depend on it.

Demo insurance: everything works with zero network (local mode); the fleet server
runs in Docker on the same laptop if venue wifi dies. The cold open starts
from a **bundled "yesterday's memory" shard** (known-good, versioned), so
step 1 has names to show; `make demo-restore` resets to it in seconds if
live teaching/curation pollutes the stage state; `make demo-check` also
verifies every model loads **from local cache with networking disabled**
(first-run downloads at a venue are a classic demo killer).

---

## 5. UI

Evolve v1's dark mission-control look into a fleet-ops identity (Qdrant brand
accents; devices as "units", the shared memory as "the fleet"). Vanilla JS + canvas overlay,
no build step. Panels:

- **Live view**: boxes (recognized = solid + name + score; suggested = dashed
  + chip; unknown = dotted "?"; ignored = faint), latency HUD (last query µs/ms,
  rolling sparkline, memory count, fleet status dot).
- **Unknowns drawer**: hidden by default — a toggle button with a count badge
  ("? 3") opens a side drawer listing current unknowns (ranked by track
  stability); click one → teach popover (name field + "same as…" suggestions
  with thumbnails). Unknown boxes on the live feed stay directly clickable
  too — the drawer is a finder, not the only path.
- **Inventory**: recognized objects with portraits, view counts, last-seen.
- **Curation panel**: v1's review grid, leaner: rename / merge / forget /
  ignore / per-view prune (thumbnails per row), then "⛟ push to fleet"
  (same-label objects fold into the existing fleet point — §3.6).
- **Map**: 2D projection of object embeddings; excludes `synthetic: true`
  points (the scale stunt must not detonate it) and renders from snapshots
  off the hot path — decorative, never load-bearing. Projection method is
  the builder's choice (v1's projector is a reference); don't gold-plate it.
- **Search**: one box; BM25 over labels, results highlight in inventory + map.
  Phase-gated: if the sparse-vector plumbing costs more than half a day,
  ship payload substring match — indistinguishable on stage at demo scale.

---

## 6. Testing (in-tree from day one — v1's dead-gate lesson)

- `tests/gate/` — **deterministic core gate**: mock embedder (controlled
  geometry) + mock detector driving the real core through scripted scenarios:
  teach/recognize/suggest/reject/ignore/merge/prune/forget, two-shard merge
  dedup, sync dedup-after-pull, unknown surfacing. Runs in CI in seconds.
  Port the *scenario ideas* from v1's `sim_reid.py` (25 scenarios), not the code.
- `tests/smoke/` — real models + real Edge shard over a bundled 10s clip:
  boot → detect → teach (scripted) → persist → reload → re-recognize.
  Marked slow; run before demos and on demand (models are GBs — local only,
  not CI).
- `tests/drive/` — **headless WS drive** (v1's most valuable harness, `ws_drive`
  reborn in-tree): boots the real server on a temp port + temp data dir, feeds
  recorded frames over the WebSocket, scripts teach/confirm/reject/curate/push
  over the wire, and asserts the emitted events. This is the only layer that
  proves server+core+events end-to-end without a browser — the UI itself gets
  no automated tests (vanilla JS, demo-grade; the drive harness plus eyeballs
  is the deliberate trade).
- `tests/sync/` — against Docker Qdrant (spike script §9.3 grows into this):
  schema round-trip, partial-snapshot pull, curated push, and **explicitly**:
  the id-present dedup rule (§3.3 — pushed points leave mutable, **unpushed
  local objects survive a pull**), and a two-device scenario (A teaches+pushes,
  B pulls+recognizes, both push same-label objects → one fleet point with the
  views of both, per §3.6 label-fold).
  The §9.3 spike verified the docs' blanket pattern; the curated variant is
  OUR design delta and gets its own test before phase 5 is called done.
- A `make demo-check` target chaining smoke + sync + offline-model-cache
  verification = pre-stage ritual; `make demo-restore` = golden-state reset.

---

## 7. Repo layout & tooling

```
hive-mind/
  pyproject.toml        # uv; deps: qdrant-edge-py, fastembed (Unicom + BM25),
                        # ultralytics, fastapi, uvicorn, opencv-python, numpy
  Makefile              # setup / run / run-b (second instance) / reset /
                        # fleet-up (docker) / demo-check / demo-restore /
                        # demo-scale (prebuild stunt shard) / test
  docker-compose.yml    # the local fleet server
  .env.example          # QDRANT_URL, QDRANT_API_KEY, DEVICE_NAME, EVENT_TAG
  fleetmemory/             # §3.1 layers
  static/
  tests/
  scripts/              # preload_scale.py (stunt shard builder)
  docs/spikes/          # the five spike scripts + results (evidence)
  PLAN.md               # this file
  CLAUDE.md             # build-session context (below)
```

Python 3.12, `uv` managed, `ruff` + `pytest` in CI (GitHub Actions). CI runs
lint + the deterministic gate + the sync tests (random vectors + dockerized
Qdrant — no model downloads); smoke and drive stay local `make` targets
because they need the real models.

> **Amendment (2026-07-01, Dylan):** GitHub Actions CI cut — single-builder
> repo. The same suites (lint + gate + sync) run locally before every commit
> instead; §6's dead-gate protection now lives in that commit ritual.

---

## 8. Build phases (each lands green with its tests)

1. **Scaffold**: repo layout, uv, CI, Makefile, .env plumbing, empty layers.
2. **Perception**: detector + tracker glue + masked crops + embedder, golden
   fixtures; measure fps to size the ingest cadence. **Exit criterion: a
   10-minute live-webcam soak of the integrated pipeline** (detect + track +
   embed-on-cadence + query) holding ≥2 fps ingest with stable memory — the
   spikes measured components in isolation (87 ms/frame detect, 7.4 ms/crop
   embed, ~22 proposals/frame); the embed *cadence* (new/stale tracks only,
   never all proposals every frame) is what makes the budget close, and it
   gets proven here, not assumed. This soak also validates the detector's
   live tracking stability (the spike used `predict` on 24 sampled frames).
3. **Memory core**: two-shard store, matcher, verbs, events; the full
   deterministic gate. (Biggest phase — the heart.)
4. **Server + live UI**: WS, overlay, HUD, teach flow, unknowns queue,
   inventory. *At the end of this phase the local demo is showable.*
5. **Sync**: fleet client, seed/pull/push with label-fold, curation panel,
   docker-compose, sync tests, second-instance support (`EDGE_DATA_DIR`-style
   override was v1's trick; here a first-class `--instance` flag).
6. **Polish**: map, search, scale-stunt script, identity/brand pass, README
   with demo script, `make demo-check`.

Phases 2 and 3 can proceed in parallel after 1 (perception is pure).

---

## 9. Spike evidence (measured 2026-07-01, this laptop, MPS/CPU)

### 9.1 Detector — class-agnostic proposals ✅ DECIDED: YOLOE-11L prompt-free

24 frames, 2 clips, MPS, imgsz 640, area-band filtered:

| Model | ms/frame (med, p90) | proposals/frame (med) | Visual quality |
|---|---|---|---|
| **yoloe-11l-seg-pf** | **87, 137** | **22** | Object-level proposals (plant, chairs, lamps, vases, rug, art), tight masks. Some furniture sub-parts — the stability gate's job. |
| yoloe-11s-seg-pf | 83, 92 | 18 | Hallucinates room-spanning mega-blobs; masks messy. Barely faster than 11l on MPS. |
| fastsam-s | 45, 61 | 50 | Segments *architecture* — walls, floor, ceiling fragments as giant regions. Would flood the unknowns queue. |
| fastsam-x | 62, 80 | 54 | Same failure mode, slower. |

Decision: **YOLOE-11L prompt-free** (built-in vocab drives detection; labels
discarded). ~11 fps ≫ 5 fps ingest target; same ultralytics `track()` /
BoT-SORT integration v1 already proved; spike conf=0.25 — expect to tune up
on real webcam scenes. Annotated frames: `docs/spikes/det/`.

### 9.2 Embedding — instance separation ✅ DECIDED: Unicom-ViT-B-32 (fastembed)

Protocol (self-verifying, no manual labels): 1,248 masked crops, 126 tracks,
3 clips, v1's exact crop path. Same-item pairs = within one track (5,987);
diff-item pairs = tracks coexisting in the same frame → provably distinct
(5,103); hard subset = coexisting same-class pairs (585). Script:
`docs/spikes/bench_embeddings.py`, raw numbers `docs/spikes/bench_results.json`.

| Model | dim | ms/crop | AUC | hard AUC | same med/p10 | diff p90/p99 | R@1%FMR all/hard |
|---|---|---|---|---|---|---|---|
| siglip2-base (v1) | 768 | 30.2 (CPU) | 0.951 | 0.841 | 0.931 / 0.836 | 0.817 / 0.947 | 0.36 / 0.15 |
| clip-ViT-B-32 | 512 | 7.5 (CPU) | 0.923 | 0.832 | 0.929 / 0.835 | 0.860 / 0.947 | 0.34 / 0.18 |
| **Unicom-ViT-B-32** | **512** | **7.4 (CPU)** | **0.951** | **0.856** | 0.789 / 0.469 | **0.411** / 0.832 | **0.39 / 0.20** |
| Unicom-ViT-B-16 | 768 | 26.9 (CPU) | 0.951 | 0.851 | 0.746 / 0.436 | 0.373 / 0.800 | 0.37 / 0.14 |
| dinov2-small | 384 | 4.3 (MPS) | 0.947 | 0.830 | 0.772 / 0.446 | 0.378 / 0.843 | 0.32 / 0.14 |

Why Unicom-B/32: the semantic models (SigLIP2/CLIP) compress all scores into
~0.82–0.95 — same-item *median* sits BELOW different-item p99, i.e. v1's "no
clean threshold exists" ceiling, now measured. Unicom (trained for instance
retrieval) spreads the scale: **working margin same-med↔diff-p90 of 0.38 vs
SigLIP2's 0.11**, best hard-pair AUC, best recall at fixed false-match rate,
4× faster than SigLIP2, 512-d (halves shard size; §9.4 was measured at 512-d),
ONNX CPU (leaves MPS to the detector), and it's a fastembed model — the
Qdrant-ecosystem story ("detection by YOLOE, embeddings by FastEmbed,
memory by Qdrant Edge").

Starting thresholds — derived from measured operating curves
(`docs/spikes/unicom_thresholds.json`, single-view pairs):

| thr | TPR | FMR | hard-FMR |
|---|---|---|---|
| 0.60 | 0.81 | 3.3% | 20.5% |
| 0.70 | 0.68 | 2.5% | 15.2% |
| **0.80** | 0.47 | **1.4%** | 8.0% |
| **0.90** | 0.17 | **0.2%** | 0.9% |

`S_same = 0.80` (auto-recognition; 1.4% single-view false-match, and the
negative-veto rerank sits behind it), `S_suggest = 0.55–0.60` (a suggestion
costs one tap, so ~3–4% false-suggest is acceptable for ~0.85 TPR),
`S_ignore = 0.90` (false-suppression is the costly error; v1's strictness
vindicated). Three caveats that all *raise* live TPR relative to this table:
matching is MAX_SIM against up to 12 accreted views (best-of-N, not one
pair); teaching captures a multi-angle burst (§3.4); and hard-FMR here is
dominated by literal twin furniture. All three sliders stay live-tunable and
must be re-derived on real demo objects (§11).

Caveats: (a) the hard-pair tail (p99 0.83) includes literal twin furniture —
matching chair sets coexisting in frame — indistinguishable by appearance for
ANY model; personal demo objects are friendlier. (b) No embedder gives a
human-free threshold — the two-tier suggest+confirm UX and negative exemplars
are load-bearing, not nice-to-haves. (c) **This benchmark ranks models; it
does not predict live rates.** Same-pairs trust tracker id purity, and
diff-pairs (coexisting only) cannot represent the "one object splits into two
tracks over time" failure mode — the very case re-id must fix. Both biases
apply equally to all five models (ranking robust), but absolute
thresholds/rates must be re-measured on domain-matched footage — handheld
objects at webcam distance, identity human-verified (§11 defines the
calibration protocol; the demo is open-world, so calibrate the domain, not
specific objects).

### 9.3 Sync — Edge ↔ server round-trip ✅ PASSED
Server 1.18.2 (Docker) ↔ qdrant-edge-py 0.7.2, with the §3.3 schema (MAX_SIM
multivector + BM25 sparse): collection → full snapshot → `unpack_snapshot` →
`EdgeShard.load` → native MAX_SIM query ✓; late server-side points delivered
via `snapshot_manifest` → partial snapshot → `update_from_snapshot` ✓;
mutable-shard dual-write + `t_sync`-filtered dedup delete ✓.
Gotcha for the build: download snapshots with `iter_content` (chunked
transfer); `count()` takes a `CountRequest`. Script: `docs/spikes/spike_sync.py`.

### 9.4 Scale — the 100k stunt ✅ GO

Edge shard, 512-d, every point = 1 single vector + 3-row MAX_SIM multivector
(so 100k points = 300k exemplar vectors), query top-10, this laptop. (The
spike schema carried an extra single vector the final schema drops, and real
objects store up to 12 rows vs 3 here — preload synthetics at ~3 rows keeps
the stunt honest while real taught objects number only in the hundreds.)

| Points | MAX_SIM med (raw → optimized) | single-vector med (opt) |
|---|---|---|
| 1k | 0.09 ms | 0.04 ms |
| 10k | 0.89 → 0.44 ms | 0.15 ms |
| 50k | 4.3 → 1.45 ms | 0.51 ms |
| 100k | 6.5 → **2.7 ms** | 0.90 ms |

Insert of 100k ≈ 30 s; `optimize()` at 100k ≈ 26 s — so the stunt shard is
**prebuilt** and swapped in on stage (§4), never built live. Shard on disk
≈ 2.4 GB (synthetic preload skips thumbnails; points carry `synthetic: true`
so the map and inventory exclude them). Demo line holds: *"a year of robot
memories, recognized in under 3 ms, no server."* The HUD shows count +
latency. Note: this measured ONE shard; the live HUD number is a two-shard
fanout + rerank — at demo scale the second shard is small, but the honest
end-to-end HUD number is what `tests/smoke` measures and what we quote.

---

## 10. Name — DECIDED: Fleet Memory

Criterion (Dylan): this is a demo, not a launched product — the name should be
**self-descriptive**, understood from a booth screen with zero explanation.
Whichever wins, the subtitle does the rest of the work:
*"shared object memory on Qdrant Edge."*

| Name | Case |
|---|---|
| **Fleet Memory** ✅ DECIDED | Says the collective feature in the audience's own language — "every robot in your fleet remembers what one learned." Fleet is real robotics/ops vocabulary, matching the stated use cases (robots, wearables). |
| Hive Memory | Self-descriptive AND keeps the repo/bee identity (honeycomb visuals, "the fleet" for the server). "One hive memory, many eyes." |
| Collective Memory | The most literal statement of the feature; slightly long on a slide. |
| Swarm Memory | Swarm-robotics flavor; punchier than Collective, nerdier than Fleet. |
| Edge Memory | Nods straight at Qdrant Edge; risk: reads like a Qdrant product/feature name rather than a demo, which could confuse. |

(Repo stays `hive-mind` regardless; the name is the demo's display identity.)

---

## 11. Risks & mitigations

- **Instance discrimination is still the hard problem.** v1's ceiling was the
  embedding, not the code. The spike picks the best available; the mitigations
  are: thresholds tunable live, suggest-tier catches borderline cases with a
  human tap, merge/name==identity recovers duplicates. **Threshold calibration
  is per-DOMAIN, not per-object** — the demo is open-world (any person, venue,
  object every time), so don't chase "the" demo items. What must match is the
  *kind* of data: handheld personal items, webcam distance, masked crops.
  Best source: a ~10-minute self-recorded webcam session with any assortment
  of desk objects (conditions match exactly; extraction + benchmark scripts
  are reusable as-is). Internet videos work as a supplement IF domain-matched
  — unboxing/EDC/product-review footage where hands rotate objects at the
  camera — NOT room-tour stock (that's the furniture domain §9.2 already
  measured). The live sliders absorb residual venue-to-venue drift.
- **Class-agnostic detectors propose everything** (wall art, shadows).
  Gates: box-area band, K-frame stability, ignore verb, per-track query cadence.
  If the unknowns queue is still spammy on real scenes, add an objectness/size
  prior — decided during phase 2 with real webcam data, not guessed now.
- **Tracker churn duplicates** (v1's dominant duplicate source): slow re-query
  cadence + easy merge + name==identity; accepted residual.
- **Two instances on one laptop** may contend for MPS. Plan A: second laptop
  at the booth. Plan B: one instance + relaunch-as-pull demo. Sizing measured
  in phase 2.
- ~~Qdrant Cloud partial-snapshot availability~~ — **resolved**: verified on
  the real cluster (§3.6); Docker fleet server remains the offline fallback.
- **Edge is beta** — pin `qdrant-edge-py==0.7.2`, keep the sync test as the
  canary when bumping.

---

## 12. v1 lessons the builder must not re-learn (from PROJECT_STATE.md)

1. No god-objects: the registry split (§3.1/3.5) is the design, not a wish.
2. No hand-rolled lock choreography: one core thread, commands in, events out.
3. The shard is the source of truth for vectors; in-memory copies are caches.
4. Tests live in-tree and run in CI from phase 1, or they die silently.
5. Never trust the tracker for identity; never trust the detector for meaning.
6. Measure detector/tracker config changes in separate fresh processes
   (warm-state artifacts produced a fake win in v1).
7. Exemplar metadata travels WITH its vector row as one object (v1's
   parallel-array desyncs), including through sync.
