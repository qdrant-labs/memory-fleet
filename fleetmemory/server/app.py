"""FastAPI + one WebSocket (PLAN.md §3.1). Serializes UI commands into the
core's queue, broadcasts core events + frames. No logic here."""

import asyncio
import contextlib
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from fleetmemory.config import Settings
from fleetmemory.memory import core as verbs
from fleetmemory.memory.store import Store
from fleetmemory.server.pipeline import CameraSource, DriveSource, Pipeline

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parents[2] / "static"


class Hub:
    """Fan events out to every connected WebSocket; safe to call from any thread."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: dict[WebSocket, asyncio.Queue] = {}

    def broadcast(self, event: dict):
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(self._push, event)

    def _push(self, event: dict):
        is_frame = event.get("type") == "frame"
        for q in self.clients.values():
            if is_frame and q.qsize() > 4:
                continue  # shed frames for slow clients; verbs/events still queue
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(event)


def create_app(settings: Settings, drive_mode: bool = False) -> FastAPI:
    hub = Hub()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        app.state.core.start()
        app.state.pipeline.start()
        if app.state.sync is not None:
            app.state.sync.start()
        yield
        if app.state.sync is not None:
            app.state.sync.stop()
        app.state.pipeline.stop()
        app.state.core.stop()
        app.state.store.close()

    app = FastAPI(title="Fleet Memory", lifespan=lifespan)

    store = Store(settings.data_dir, with_immutable=settings.fleet_enabled)
    core = verbs.Core(
        store,
        device_name=settings.device_name,
        event_tag=settings.event_tag,
        on_event=lambda e: (app.state.pipeline.note_event(e), hub.broadcast(e)),
    )
    source = DriveSource() if drive_mode else CameraSource(0)
    pipeline = Pipeline(core, source, hub.broadcast, drive_mode=drive_mode)

    sync = None
    if settings.fleet_enabled:
        from fleetmemory.sync.client import FleetClient
        from fleetmemory.sync.manager import SyncManager

        client = FleetClient(settings.qdrant_url, settings.qdrant_api_key, dim=store.dim)
        sync = SyncManager(core, client, on_event=hub.broadcast)

    app.state.settings, app.state.store, app.state.core = settings, store, core
    app.state.pipeline, app.state.hub, app.state.sync = pipeline, hub, sync

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/thumbs/{view_id}.jpg")
    async def thumb(view_id: str):
        p = core.thumbs_dir / f"{view_id}.jpg"
        if not p.is_file() or not view_id.isalnum():
            return Response(status_code=404)
        return FileResponse(p, media_type="image/jpeg")

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
                msg = json.loads(await sock.receive_text())
                _dispatch(app, msg)
        except WebSocketDisconnect:
            pass
        finally:
            send_task.cancel()
            hub.clients.pop(sock, None)
            if not hub.clients:
                pipeline.set_active(False)  # last viewer left: release the camera

    return app


def _hello(app) -> dict:
    core, settings = app.state.core, app.state.settings
    t = core.thresholds
    return {
        "type": "hello",
        "device": settings.device_name,
        "fleet": settings.fleet_enabled,
        "memories": app.state.store.count(),
        "thresholds": {"s_same": t.s_same, "s_suggest": t.s_suggest, "s_ignore": t.s_ignore},
        "detector_conf": app.state.pipeline.detector.conf,
        "camera": app.state.pipeline.user_enabled,
    }


def _dispatch(app, m: dict):
    core, pipeline = app.state.core, app.state.pipeline
    cmd = m.get("cmd")
    if cmd == "frame":  # drive mode: recorded frames over the wire (tests/drive)
        if isinstance(pipeline.source, DriveSource):
            import base64

            pipeline.source.push(base64.b64decode(m["jpg"]))
        return
    if cmd == "conf":
        pipeline.detector.conf = float(m["value"])
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
            return verbs.IgnoreTrack(tid=int(m["tid"]), epoch=int(m["epoch"]))
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
        case "map":
            return verbs.MapRequest()
        case "scale":
            return verbs.ScaleStunt(on=bool(m.get("on")))
        case _:
            logger.warning("unknown command %r", cmd)
            return None
