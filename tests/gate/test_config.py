"""Phase 1 gate: .env plumbing and the fleet opt-in gate.

Every test passes an explicit environ and env_file so the repo's real .env
(which carries live fleet credentials) can never leak into the gate.
"""

from fleetmemory.config import DEFAULT_PORT, load_env_file, load_settings


def test_local_mode_without_fleet_env(tmp_path):
    s = load_settings(env_file=tmp_path / "absent.env", environ={})
    assert s.fleet_enabled is False
    assert s.qdrant_url is None
    assert s.qdrant_api_key is None


def test_fleet_enabled_with_url(tmp_path):
    s = load_settings(
        env_file=tmp_path / "absent.env",
        environ={"QDRANT_URL": "https://fleet.example", "QDRANT_API_KEY": "k"},
    )
    assert s.fleet_enabled is True
    assert s.qdrant_url == "https://fleet.example"


def test_env_file_parsed_with_comments_and_quotes(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# fleet\nQDRANT_URL=https://fleet.example\n\nDEVICE_NAME='unit-a'\nnot a pair\n"
    )
    assert load_env_file(env) == {
        "QDRANT_URL": "https://fleet.example",
        "DEVICE_NAME": "unit-a",
    }


def test_environ_wins_over_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("DEVICE_NAME=from-file\nEVENT_TAG=conf-2026\n")
    s = load_settings(env_file=env, environ={"DEVICE_NAME": "from-env"})
    assert s.device_name == "from-env"
    assert s.event_tag == "conf-2026"


def test_default_port_is_never_8000(tmp_path):
    s = load_settings(env_file=tmp_path / "absent.env", environ={})
    assert s.port == DEFAULT_PORT
    assert s.port != 8000


def test_port_and_data_dir_overrides(tmp_path):
    s = load_settings(
        env_file=tmp_path / "absent.env",
        environ={"FM_PORT": "9111", "FM_DATA_DIR": str(tmp_path / "d")},
    )
    assert s.port == 9111
    assert s.data_dir == tmp_path / "d"
