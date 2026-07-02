# Fleet Memory

Shared object memory on Qdrant Edge.

A webcam watches a desk. A class-agnostic detector proposes regions: it assigns
no labels and has no vocabulary. Each stable region is embedded on-device and
looked up in [Qdrant Edge](https://qdrant.tech/edge/), a vector search engine
running inside the app process. A match shows the remembered name plus the live
query latency. No match shows "unknown, teach me": a human names it once, and
every device in the fleet can know it after curation.

The model only sees shapes. The memory knows what things are, and the memory is
shared.

## How It Works

- **Names come from vector search only.** The detector (YOLOE-11L prompt-free)
  proposes boxes and masks; its labels are discarded. Recognition is one native
  MAX_SIM query over multivector points (up to 12 views per object), embedded
  with Unicom-ViT-B-32 via FastEmbed. Sub-millisecond at taught scale, under
  3 ms against 100k memories (measured in `docs/spikes/`).
- **Two-tier matching.** High similarity binds and shows the name. Borderline
  similarity asks: "looks like «mug», same?" One tap confirms or rejects.
  Rejections become negative exemplars that veto future false matches.
- **Local first, fleet by curation.** Every device runs two Edge shards: a
  mutable shard for local teachings and an immutable mirror of the central
  fleet collection, synced by native partial snapshots. Nothing reaches the
  fleet uncurated: review, prune bad views, then push. Same-name objects from
  different devices fold into one fleet point.
- **No fleet configured, no problem.** Without `QDRANT_URL` the demo runs fully
  local. The fleet target is any Qdrant server: Qdrant Cloud or the bundled
  Docker compose file.

## Quickstart

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and a webcam.

```bash
make setup          # install (first run downloads model weights, ~500 MB)
make run            # http://127.0.0.1:8765
```

Optional fleet sync:

```bash
make fleet-up       # local Qdrant in Docker (or point .env at Qdrant Cloud)
cp .env.example .env  # set QDRANT_URL (+ QDRANT_API_KEY for Cloud)
make run
make run-b          # second "device" on the same laptop (own port + data dir)
```

## Demo Script (3 Minutes)

1. **Cold open.** Camera on a desk. Known objects carry solid boxes, names, and
   a live latency readout. Unknowns carry a quiet "?".
2. **Teach.** Click an unknown, type a name, rotate the item through the
   3-second capture burst. It now recognizes the item from any side.
3. **Speed.** The HUD strip shows every recognition query: microseconds to
   low milliseconds, on-device, no server.
4. **Scale.** Press `S` to swap in the prebuilt 100k-memory shard ("what if
   this robot had been running for a year?"). Watch the latency barely move.
   Build it once with `make demo-scale`.
5. **The fleet.** Open inventory, prune a bad view, push. On the second
   device: pull, and it recognizes everything the first device taught.
   Every robot you ship knows what any of them ever learned.

Before going on stage: `make demo-check` verifies models load offline and runs
the full test chain. `make demo-restore` resets to the saved golden state.

## Architecture

```
webcam -> detector (boxes+masks, no labels) -> masked crops -> embedder (512-d)
   -> single-threaded memory core -> two Edge shards (mutable + fleet mirror)
   -> events over one WebSocket -> vanilla JS overlay
fleet: curated push via qdrant-client; pull via native partial snapshots (~30 s)
```

Layer map in `PLAN.md` (the build contract), spike evidence in `docs/spikes/`.

## Tests

```bash
make test    # deterministic core gate: mock geometry, real Edge shards
make smoke   # real models over a bundled clip
make soak    # 10-minute live pipeline soak, >= 2 fps required
uv run pytest tests/sync  # fleet round-trip against Docker Qdrant
uv run pytest tests/drive # headless WebSocket drive of the real server
```
