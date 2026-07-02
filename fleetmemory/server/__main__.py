"""Run the Fleet Memory server: uv run python -m fleetmemory.server"""

import argparse
import logging
from dataclasses import replace

import uvicorn

from fleetmemory.config import load_settings
from fleetmemory.server.app import create_app


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--drive", action="store_true", help="frames over WS instead of camera")
    ap.add_argument(
        "--instance",
        default=None,
        help="second device on one laptop: own data dir + port offset (e.g. 'b')",
    )
    args = ap.parse_args()

    settings = load_settings()
    port = args.port or settings.port
    if args.instance:
        settings = replace(
            settings,
            data_dir=settings.data_dir.with_name(f"{settings.data_dir.name}-{args.instance}"),
            device_name=f"{settings.device_name}-{args.instance}",
        )
        if args.port is None:
            port += 1
    app = create_app(settings, drive_mode=args.drive)
    print(f"\n  Fleet Memory · unit {settings.device_name}")
    print(f"  → http://127.0.0.1:{port}\n", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
