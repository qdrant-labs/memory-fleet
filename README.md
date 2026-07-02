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
  proposes boxes and masks; recognition is one native MAX_SIM query over
  multivector points, embedded with Unicom-ViT-B-32 via FastEmbed.
  Sub-millisecond at taught scale, under 3 ms against 300k memories
  (measured in `docs/spikes/`). Detector class names appear only as
  teach-time suggestion chips — a human always does the naming.
- **An object is one physical thing.** Each point is one item with up to 24
  views of it; the label is a display name and may repeat. Teach two
  different watches as "watch" and you get two clean objects that both
  answer to "watch" — re-teach the same watch and it folds into itself.
- **Two-tier matching.** High similarity binds and shows the name. Borderline
  similarity asks: "looks like «mug», same?" One tap confirms or rejects.
  Rejections become negative exemplars that veto future false matches, and
  ignored looks are suppressed without ever silently eating a taught object.
- **Hybrid search over everything learned.** miniCOIL sparse + dense text
  embeddings, fused with reciprocal rank fusion, all on-device — with the
  engine latency on screen. "Cup" finds the coffee mug.
- **Local first, fleet by curation.** Every device runs two Edge shards: a
  mutable shard for local teachings and an immutable mirror of the central
  fleet collection, synced by Edge's native partial snapshots. Nothing
  reaches the fleet uncurated: review, prune bad views, then push. The same
  item pushed from two devices folds into one fleet point.
- **The fleet is a bonus, never a dependency.** Without `QDRANT_URL` (or
  without wifi) everything runs fully on-device; the Qdrant Cloud fleet
  reconnects on its own when reachable.

## Quickstart

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and a webcam.

```bash
make setup          # install (first run downloads model weights, ~500 MB)
make run            # http://127.0.0.1:8765
```

Optional fleet sync (local-first: an unreachable fleet degrades gracefully):

```bash
cp .env.example .env  # set QDRANT_URL + QDRANT_API_KEY (Qdrant Cloud)
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
4. **Scale.** Press `S` to attach the prebuilt stunt shard: 300,000 memories
   ("what if this robot had been running for a year?"). Watch the latency
   barely move. Build it once with `make demo-scale`.
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
make test-sync            # fleet round-trip against the Cloud cluster in .env
uv run pytest tests/drive # headless WebSocket drive of the real server
```
