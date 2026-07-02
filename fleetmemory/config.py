"""Runtime configuration: .env plumbing and the fleet opt-in gate.

Fleet sync is opt-in: no QDRANT_URL in the environment means a fully local
demo — no sync UI beyond a "fleet offline" hint.
"""

import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# Port 8000 is habitually taken on the dev machine — never assume it's free.
DEFAULT_PORT = 8765


def load_env_file(path: str | Path = ".env") -> dict[str, str]:
    """Parse a KEY=VALUE .env file. Missing file -> empty dict."""
    path = Path(path)
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if not (value.startswith('"') or value.startswith("'")):
            value = value.split(" #")[0].strip()  # drop inline comments
        values[key] = value.strip("'\"")
    return values


@dataclass(frozen=True)
class Settings:
    qdrant_url: str | None
    qdrant_api_key: str | None
    device_name: str
    event_tag: str
    port: int
    data_dir: Path

    @property
    def fleet_enabled(self) -> bool:
        return bool(self.qdrant_url)


def load_settings(
    env_file: str | Path = ".env",
    environ: Mapping[str, str] = os.environ,
) -> Settings:
    """Build Settings from a .env file merged with the environment (environ wins)."""
    merged = {**load_env_file(env_file), **environ}
    try:
        port = int(merged.get("FM_PORT") or DEFAULT_PORT)
    except ValueError:
        port = DEFAULT_PORT
    return Settings(
        qdrant_url=merged.get("QDRANT_URL") or None,
        qdrant_api_key=merged.get("QDRANT_API_KEY") or None,
        device_name=merged.get("DEVICE_NAME") or socket.gethostname(),
        event_tag=merged.get("EVENT_TAG") or "dev",
        port=port,
        data_dir=Path(merged.get("FM_DATA_DIR") or "edge-data"),
    )
