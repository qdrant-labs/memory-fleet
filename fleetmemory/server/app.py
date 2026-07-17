"""FastAPI + one WebSocket. Serializes UI commands into the core's queue,
broadcasts core events + frames. No logic here."""

import asyncio
import contextlib
import json
import logging
import threading
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from fleetmemory.config import Settings
from fleetmemory.memory import core as verbs
from fleetmemory.memory.store import Store
from fleetmemory.server.pipeline import CameraSource, Pipeline

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parents[2] / "static"


class Hub:
    """Fan events out to every connected WebSocket; safe to call from any thread."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: dict[WebSocket, asyncio.Queue] = {}

    def broadcast(self, event: dict):
        if self.loop is None or self.loop.is_closed():
            return  # worker threads may outlive the loop briefly at shutdown
        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(self._push, event)

    def _push(self, event: dict):
        is_frame = event.get("type") == "frame"
        for q in self.clients.values():
            if is_frame and q.qsize() > 4:
                continue  # shed frames for slow clients; verbs/events still queue
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(event)


def create_app(settings: Settings) -> FastAPI:
    hub = Hub()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        app.state.core.start()
        app.state.pipeline.start()

        # warm the label + speech models off-thread so the first teach/search/
        # voice command doesn't stall
        def warm_models():
            label_embedder.load()
            from fleetmemory.perception import asr

            asr.load()

        threading.Thread(target=warm_models, name="models-warm", daemon=True).start()
        if app.state.sync is not None:
            app.state.sync.start()
        yield
        if app.state.sync is not None:
            app.state.sync.stop()
        app.state.pipeline.stop()
        app.state.core.stop()
        app.state.store.close()

    app = FastAPI(title="Fleet Memory", lifespan=lifespan)

    from fleetmemory.memory.labels import LabelEmbedder

    label_embedder = LabelEmbedder()  # miniCOIL + dense; loads lazily, warmed at startup
    store = Store(
        settings.data_dir, with_immutable=settings.fleet_enabled, label_embedder=label_embedder
    )
    # the shards are core-thread-only, so _hello must not call into the store;
    # track the last known count from the event stream instead
    app.state.mem_count = store.vector_count()  # safe: core thread hasn't started yet
    app.state.disk_bytes = store.disk_bytes()

    def on_core_event(e: dict):
        if "memories" in e:
            app.state.mem_count = e["memories"]
        elif e.get("type") == "query":
            app.state.mem_count = e["searched"]
        if "disk_bytes" in e:
            app.state.disk_bytes = e["disk_bytes"]
        app.state.pipeline.note_event(e)
        hub.broadcast(e)

    core = verbs.Core(
        store,
        device_name=settings.device_name,
        event_tag=settings.event_tag,
        on_event=on_core_event,
    )
    source = CameraSource(settings.camera_index)
    pipeline = Pipeline(core, source, hub.broadcast, model=settings.detector_model)

    sync = None
    if settings.fleet_enabled:
        from fleetmemory.sync.client import FleetClient
        from fleetmemory.sync.manager import SyncManager

        client = FleetClient(
            settings.qdrant_url,
            settings.qdrant_api_key,
            dim=store.dim,
            label_embedder=label_embedder,
        )
        sync = SyncManager(core, client, on_event=hub.broadcast)

    app.state.settings, app.state.store, app.state.core = settings, store, core
    app.state.pipeline, app.state.hub, app.state.sync = pipeline, hub, sync

    # tuning dials persist next to the shards so a restart keeps them
    app.state.tuning_path = Path(settings.data_dir) / "tuning.json"
    app.state.tuning = _load_tuning(app.state.tuning_path, core, pipeline)

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/thumbs/{view_id}.jpg")
    async def thumb(view_id: str):
        p = core.thumbs_dir / f"{view_id}.jpg"
        if not p.is_file() or not view_id.isalnum():
            return Response(status_code=404)
        return FileResponse(p, media_type="image/jpeg")

    @app.post("/voice")
    async def voice(request: Request):
        """Browser mic (16 kHz mono WAV body) -> on-device whisper -> a verb.
        mode=teach names the popover's current unknown; mode=ask runs recall.
        Results flow back over the WS, same as a typed command."""
        import tempfile

        from fleetmemory.perception import asr

        data = await request.body()
        if not data:
            return {"transcript": "", "error": "no audio"}

        def work():
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                f.write(data)
                f.flush()
                return asr.transcribe(f.name)

        text = await asyncio.to_thread(work)
        if not text:
            return {"transcript": "", "error": "silence"}  # dead mic or no permission
        if request.query_params.get("mode") == "teach":
            try:
                tid = int(request.query_params["tid"])
                epoch = int(request.query_params["epoch"])
            except (KeyError, ValueError):
                return {"transcript": text, "error": "no target"}
            label = asr.parse_label(text)
            core.submit(verbs.Teach(tid=tid, epoch=epoch, label=label))
            return {"transcript": text, "label": label}
        core.submit(verbs.RecallRequest(text=text))
        return {"transcript": text}

    @app.post("/phrase")
    async def phrase(request: Request):
        """Optional garnish: a local LLM (Ollama) phrases the retrieved fact as
        one spoken sentence. It is handed ONLY label/when/where and told to
        invent nothing, so retrieval stays the source of truth. Ollama absent,
        slow, or erroring -> empty, and the UI keeps its grounded card."""
        import json as _json
        import os
        import urllib.request

        a = await request.json()
        label = str(a.get("label", ""))
        if not label:
            return {"sentence": ""}
        model = os.environ.get("FM_PHRASE_MODEL", "gemma2:2b")
        prompt = (
            "You are a home assistant. In ONE short spoken sentence, tell the user "
            "where their item is, using ONLY these facts and inventing nothing:\n"
            f"item: {label}\nlast seen: {a.get('when', 'recently')}\n"
            f"location: {a.get('where', 'unknown')}\nSentence:"
        )

        def work():
            body = _json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                body,
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=6) as r:  # noqa: S310 (localhost)
                return _json.loads(r.read()).get("response", "").strip()

        try:
            return {"sentence": await asyncio.to_thread(work)}
        except (OSError, TimeoutError, ValueError):  # Ollama absent/slow/bad reply
            return {"sentence": ""}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        hub.clients[sock] = q
        pipeline.set_active(True)  # first viewer turns the camera on
        await sock.send_text(json.dumps(_hello(app)))

        async def sender():
            while True:
                ev = await q.get()
                await sock.send_text(json.dumps(ev))

        send_task = asyncio.create_task(sender())
        try:
            while True:
                raw = await sock.receive_text()
                try:  # one malformed message must not tear down the connection
                    _dispatch(app, json.loads(raw))
                except WebSocketDisconnect:
                    raise
                except Exception:
                    logger.exception("bad ws message: %.120s", raw)
        except WebSocketDisconnect:
            pass
        finally:
            send_task.cancel()
            hub.clients.pop(sock, None)
            if not hub.clients:
                pipeline.set_active(False)  # last viewer left: release the camera

    return app


def _load_tuning(path: Path, core, pipeline) -> dict:
    """Apply persisted dials (must run before the core thread starts)."""
    try:
        raw = json.loads(path.read_text())
        tuning = {k: float(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}
    if not tuning:
        return {}
    d = pipeline.detector
    d.conf = tuning.get("conf", d.conf)
    d.max_area = tuning.get("max_area", d.max_area)
    pipeline.target_fps = max(2.0, min(15.0, tuning.get("target_fps", pipeline.target_fps)))
    t = core.thresholds
    core.thresholds = verbs.Thresholds(
        tuning.get("s_same", t.s_same),
        tuning.get("s_suggest", t.s_suggest),
        tuning.get("s_ignore", t.s_ignore),
    )
    return tuning


def _remember_tuning(app, **kv: float):
    app.state.tuning.update(kv)
    with contextlib.suppress(OSError):
        app.state.tuning_path.write_text(json.dumps(app.state.tuning))


def _hello(app) -> dict:
    core, settings = app.state.core, app.state.settings
    t = core.thresholds
    sync = app.state.sync
    return {
        "type": "hello",
        "device": settings.device_name,
        "fleet": settings.fleet_enabled,
        "fleet_online": bool(sync and sync.online),  # current state, not just transitions
        "memories": app.state.mem_count,
        "disk_bytes": app.state.disk_bytes,
        "thresholds": {"s_same": t.s_same, "s_suggest": t.s_suggest, "s_ignore": t.s_ignore},
        "detector_conf": app.state.pipeline.detector.conf,
        "detector_max_area": app.state.pipeline.detector.max_area,
        "target_fps": app.state.pipeline.target_fps,
        "camera": app.state.pipeline.user_enabled,
    }


def _dispatch(app, m: dict):
    core, pipeline = app.state.core, app.state.pipeline
    cmd = m.get("cmd")
    if cmd == "conf":
        pipeline.detector.conf = float(m["value"])
        _remember_tuning(app, conf=pipeline.detector.conf)
        return
    if cmd == "max_area":
        pipeline.detector.max_area = float(m["value"])
        _remember_tuning(app, max_area=pipeline.detector.max_area)
        return
    if cmd == "target_fps":
        pipeline.target_fps = max(2.0, min(15.0, float(m["value"])))
        _remember_tuning(app, target_fps=pipeline.target_fps)
        return
    if cmd == "camera":
        pipeline.set_user_enabled(bool(m.get("on")))
        return
    if cmd in ("pull_now", "push"):
        sync = app.state.sync
        if sync is None:
            app.state.hub.broadcast({"type": "fleet_error", "message": "fleet offline"})
        elif cmd == "pull_now":
            sync.request_pull()
        else:
            sync.request_push([str(i) for i in m.get("object_ids", [])])
        return
    if cmd == "thresholds":
        _remember_tuning(
            app,
            s_same=float(m["s_same"]),
            s_suggest=float(m["s_suggest"]),
            s_ignore=float(m["s_ignore"]),
        )
    msg = _to_message(m, cmd)
    if msg is not None:
        core.submit(msg)


def _to_message(m: dict, cmd: str):
    match cmd:
        case "teach":
            return verbs.Teach(tid=int(m["tid"]), epoch=int(m["epoch"]), label=str(m["label"]))
        case "confirm":
            return verbs.Confirm(
                tid=int(m["tid"]), epoch=int(m["epoch"]), object_id=str(m["object_id"])
            )
        case "reject":
            return verbs.Reject(
                tid=int(m["tid"]), epoch=int(m["epoch"]), object_id=str(m["object_id"])
            )
        case "ignore_track":
            return verbs.IgnoreTrack(
                tid=int(m["tid"]), epoch=int(m["epoch"]), label=str(m.get("label", ""))
            )
        case "dismiss_unknown":
            return verbs.DismissUnknown(tid=int(m["tid"]), epoch=int(m["epoch"]))
        case "ignore_object":
            return verbs.IgnoreObject(object_id=str(m["object_id"]))
        case "forget":
            return verbs.Forget(object_id=str(m["object_id"]))
        case "merge":
            return verbs.Merge(keep_id=str(m["keep_id"]), fold_id=str(m["fold_id"]))
        case "rename":
            return verbs.Rename(object_id=str(m["object_id"]), label=str(m["label"]))
        case "prune":
            return verbs.Prune(object_id=str(m["object_id"]), view_id=str(m["view_id"]))
        case "thresholds":
            return verbs.SetThresholds(
                s_same=float(m["s_same"]),
                s_suggest=float(m["s_suggest"]),
                s_ignore=float(m["s_ignore"]),
            )
        case "inventory":
            return verbs.InventoryRequest()
        case "search":
            return verbs.SearchRequest(text=str(m.get("text", "")))
        case "recall":
            return verbs.RecallRequest(text=str(m.get("text", "")))
        case "map":
            return verbs.MapRequest()
        case "scale":
            return verbs.ScaleStunt(on=bool(m.get("on")))
        case _:
            logger.warning("unknown command %r", cmd)
            return None
