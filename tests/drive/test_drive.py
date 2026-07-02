"""Headless WS drive (PLAN.md §6): boots the real server app (drive mode, temp
data dir), feeds recorded frames over the WebSocket, scripts teach over the
wire, and asserts the emitted events. Real models — local only, not CI.
"""

import base64
import json
from pathlib import Path

import cv2
import pytest
from starlette.testclient import TestClient

from fleetmemory.config import Settings
from fleetmemory.server.app import create_app

CLIP = Path("tests/fixtures/clip.mp4")
pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def clip_jpegs():
    cap = cv2.VideoCapture(str(CLIP))
    frames = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % 6 == 0:  # ~5 fps, matches live ingest
            ok2, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok2:
                frames.append(base64.b64encode(buf).decode())
        i += 1
    cap.release()
    assert len(frames) >= 40
    return frames


class Driver:
    def __init__(self, ws):
        self.ws = ws
        self.events = []

    def send(self, **msg):
        self.ws.send_text(json.dumps(msg))

    def recv(self):
        ev = json.loads(self.ws.receive_text())
        self.events.append(ev)
        return ev

    def feed_until(self, frames, pred, max_frames=200, desc=""):
        """Feed frames one at a time (self-paced against broadcasts) until pred hits."""
        fed = 0
        for jpg in frames:
            if fed >= max_frames:
                break
            self.send(cmd="frame", jpg=jpg)
            fed += 1
            # drain everything the server produced for this frame
            got_frame = False
            while not got_frame:
                ev = self.recv()
                if pred(ev):
                    return ev
                got_frame = ev["type"] == "frame"
            # also check non-frame events that raced ahead
            hit = next((e for e in self.events if pred(e)), None)
            if hit:
                return hit
        pytest.fail(f"drive: condition never met ({desc}; fed {fed} frames)")

    def last_boxes(self):
        fr = next((e for e in reversed(self.events) if e["type"] == "frame"), None)
        return fr["boxes"] if fr else []


def test_drive_teach_recognize_over_the_wire(tmp_path, clip_jpegs):
    settings = Settings(
        qdrant_url=None,
        qdrant_api_key=None,
        device_name="drive-unit",
        event_tag="drive",
        port=0,
        data_dir=tmp_path / "data",
    )
    app = create_app(settings, drive_mode=True)
    loop = clip_jpegs * 10  # loop the clip if conditions need more frames

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        d = Driver(ws)
        hello = d.recv()
        assert hello["type"] == "hello" and hello["fleet"] is False

        # 1. a stable track surfaces as unknown
        ev = d.feed_until(
            loop,
            lambda e: e["type"] == "track_update" and e["state"] == "unknown",
            desc="unknown track",
        )
        tid = ev["tid"]

        # 2. teach it over the wire -> object created, burst runs, track binds
        box = next(b for b in d.last_boxes() if b["tid"] == tid)
        d.send(cmd="teach", tid=tid, epoch=box["epoch"], label="drive thing")
        ev = d.feed_until(loop, lambda e: e["type"] == "object_created", desc="object_created")
        assert ev["label"] == "drive thing"
        d.feed_until(
            loop,
            lambda e: (
                e["type"] == "track_update" and e["tid"] == tid and e["state"] == "recognized"
            ),
            desc="burst ends recognized",
        )

        # 3. the HUD latency stream is real (post-teach queries search >= 1 memory)
        d.feed_until(
            loop,
            lambda e: e["type"] == "query" and e["ms"] > 0 and e["searched"] >= 1,
            desc="post-teach query",
        )

        # 4. inventory over the wire shows the taught object
        d.send(cmd="inventory")
        ev = d.feed_until(loop, lambda e: e["type"] == "inventory", desc="inventory")
        labels = [o["label"] for o in ev["items"]]
        assert "drive thing" in labels

        # 5. forget it -> deletion event arrives
        oid = next(o["object_id"] for o in ev["items"] if o["label"] == "drive thing")
        d.send(cmd="forget", object_id=oid)
        d.feed_until(loop, lambda e: e["type"] == "object_deleted", desc="object_deleted")
