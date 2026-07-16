# Fleet Memory

Shared object memory on Qdrant Edge: every device remembers what any device learned.

![Fleet Memory mission control: live feed with recognized objects, memory map, and fleet search](docs/screenshot-ui.png)

## The Idea

Show a device an object once, give it a name, and every device in the fleet can
recognize it. No model is trained or fine-tuned at any point: recognizing
something is a vector search over the fleet's shared memory, and teaching
something new is adding vectors to it.

You teach by typing a name or by speaking it ("this is my mug"). Asking where an
object was last seen ("where did I leave my keys?") is another vector search: the
fleet answers with the device that saw it last and when.

Each device learns locally and works fully offline. When a connection is
available, devices share what they learned through a central collection in
Qdrant Cloud. One unit learns, all units know.

## Why Qdrant Edge

The demo runs on [Qdrant Edge](https://qdrant.tech/edge/), the embedded build of
the Qdrant vector search engine:

- **On-device search.** The engine runs inside the app process, against shards
  on local disk. Recognition never leaves the device and needs no server.
- **Fully offline.** Without a fleet configured, or with the network down,
  everything keeps working. Sync resumes on its own when the fleet is reachable.
- **Native Cloud sync.** Local shards synchronize with a Qdrant Cloud collection
  through [Edge synchronization](https://qdrant.tech/documentation/edge/edge-synchronization-guide/):
  snapshot-based pulls, automatic batched pushes.

## Architecture and Stack

1. **Capture.** A grabber thread owns the webcam and streams video (~25 fps).
2. **Detect.** YOLOE proposes and tracks regions at ~8 Hz. Its class labels are
   discarded; people and body parts are filtered out.
3. **Embed.** Stable regions are cropped to their segmentation mask and embedded
   into 512-d vectors on-device.
4. **Match.** A vector search over the local shards decides: similarity ≥ 0.80
   recognizes, ≥ 0.55 suggests a confirmation, below that the object is unknown.
5. **Teach.** A human names unknowns and confirms suggestions, by typing or by
   voice. A memory is one physical thing with up to 24 views; the same name can
   cover several objects.
6. **Sync.** Taught objects go to the fleet automatically; the fleet's memory
   flows back.

| Component        | Choice                                                      |
|------------------|-------------------------------------------------------------|
| On-device search | Qdrant Edge 0.7.2, embedded shards                          |
| Fleet            | Qdrant Cloud, one shared `fleet` collection                 |
| Detector         | YOLOE-11L-seg prompt-free (ultralytics) + BoT-SORT tracking |
| Image embedding  | Unicom ViT-B/32, 512-d, via fastembed (ONNX, CPU)           |
| Text search      | miniCOIL sparse + bge-small dense, fused with RRF           |
| Speech           | whisper-base via onnx-asr (ONNX, CPU), on-device           |
| Server           | FastAPI + one WebSocket                                     |
| UI               | Vanilla JS, no build step                                   |

## How Fleet Sync Works

Each device keeps two local shards: a mutable one holding its own teachings and
a read-only mirror of the fleet collection. Recognition searches both.

- **Push is automatic.** Objects a human taught or confirmed upload in batches
  on the sync tick; anything taught offline goes up when the fleet is reachable
  again. Only vectors, one thumbnail, and metadata leave the device, never
  camera frames. An object that already exists on the fleet is merged into,
  not duplicated.
- **Pull is automatic.** The mirror updates from the fleet collection every
  ~30 seconds using
  [Qdrant's Edge synchronization](https://qdrant.tech/documentation/edge/edge-synchronization-guide/).
- **Local edits win.** Teach new views to a downloaded memory and your device
  keeps a local copy that overrides the mirror, then merges back into the fleet
  point on the next push. A sync never wipes something you taught.
- **The fleet sleeps.** `make fleet-sleep` folds duplicate instances of the
  same object into one memory and archives memories no device has seen in
  months, ranked with
  [Qdrant's decay functions](https://qdrant.tech/documentation/search/search-relevance/).
  The hot fleet stays small, so every device's mirror does too.

## Quickstart

Requirements: macOS on Apple Silicon, Python 3.12,
[uv](https://docs.astral.sh/uv/), and a webcam.

```bash
git clone https://github.com/qdrant-labs/memory-fleet.git && cd memory-fleet
make setup          # install dependencies
make run            # http://127.0.0.1:8765 (first run downloads ~500 MB of model weights)
```

Fleet sync is optional. Without a `.env`, the app runs fully local:

```bash
cp .env.example .env
# QDRANT_URL + QDRANT_API_KEY  -> a Qdrant Cloud cluster (the fleet)
# DEVICE_NAME                  -> names this unit in the fleet
# EVENT_TAG                    -> tag stamped on pushed objects
make run
```

Second device on the same laptop (port 8766, own data dir and name):

```bash
make run-b
```

Scale stunt: `make demo-scale` builds a synthetic shard of 300k vectors; press
`S` in the UI to attach it live and watch recognition latency barely move.
`make reset` wipes local memory.

Porting beyond a laptop (ballpark, not measured on this app):

- **Minimum:** 4-core CPU, 8 GB RAM, GTX 1650-class GPU, at a reduced
  detection rate (tunable live, 2-15 per second).
- **Recommended:** 16 GB RAM, RTX 3060 or Jetson Orin NX class.

Detection is the only heavy workload; on Nvidia hardware it needs a one-line
device change in `perception/detector.py`. A Jetson Orin Nano 8 GB can hold
the full detection rate if the detector is exported to TensorRT.

## Next Steps

- Two-tier memory: recognize against one compact prototype vector per object
  and keep the full view sets in Qdrant Cloud for confirmation. That is the
  path from hundreds of shared objects to hundreds of thousands.

## Applications

- Robot or drone fleets that share what they have seen without shipping raw video.
- Retail and warehouse device fleets learning a shared inventory from any unit.
- Wearables and smart cameras that recognize objects a peer taught them.
- Any edge fleet where devices must learn from each other while images stay on
  the device.

## Repo Layout

```
fleetmemory/
  config.py          # .env plumbing, fleet opt-in gate
  perception/        # detector, masked crops, embedder, speech, embed cadence
  memory/            # store (two shards), matcher, labels, core
  sync/              # fleet client, sync manager, fleet-sleep job
  server/            # FastAPI app, WebSocket, capture/detect pipeline
static/              # vanilla-JS UI + brand assets
scripts/             # preload_scale.py (stunt shard), demo_check.py (offline preflight)
```
