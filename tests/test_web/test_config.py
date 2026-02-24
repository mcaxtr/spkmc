"""Tests for web configuration management."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_config(tmp_path):
    """Return a WebConfig instance whose CONFIG_FILE lives in tmp_path."""
    from spkmc.web.config import WebConfig

    cfg = WebConfig()
    cfg.CONFIG_FILE = tmp_path / "web_config.json"
    cfg.config = WebConfig.DEFAULTS.copy()
    return cfg


# ── Defaults ──────────────────────────────────────────────────────────────────


def test_web_config_defaults():
    """WebConfig initializes with expected default values."""
    from spkmc.web.config import WebConfig

    config = WebConfig()

    assert config.get("data_directory") == "data"
    assert config.get("experiments_directory") == "experiments"
    assert config.get("default_nodes") == 1000


def test_all_defaults_are_present():
    """Every key in DEFAULTS must be accessible via get()."""
    from spkmc.web.config import WebConfig

    config = WebConfig()
    for key in WebConfig.DEFAULTS:
        assert config.get(key) is not None or WebConfig.DEFAULTS[key] is None


# ── Save / load round-trip ────────────────────────────────────────────────────


def test_save_and_load_round_trip(tmp_path):
    """A value written via set() must survive a reload from disk."""
    cfg = _make_config(tmp_path)
    cfg.set("my_key", "my_value")

    cfg2 = _make_config(tmp_path)
    cfg2.load()

    assert cfg2.get("my_key") == "my_value"


def test_update_persists_multiple_keys(tmp_path):
    """update() must persist all supplied keys to disk."""
    cfg = _make_config(tmp_path)
    cfg.update({"key1": "v1", "key2": 42})

    cfg2 = _make_config(tmp_path)
    cfg2.load()

    assert cfg2.get("key1") == "v1"
    assert cfg2.get("key2") == 42


# ── Type coercion ─────────────────────────────────────────────────────────────


def test_json_integer_coerced_to_float_when_default_is_float(tmp_path):
    """
    JSON may deserialize 10.0 as int 10.
    WebConfig must coerce it back to float to avoid Streamlit type errors.
    """
    cfg = _make_config(tmp_path)
    # Write a file where a float default is stored as JSON integer
    data = {**cfg.config, "default_k_avg": 10}  # int instead of float
    tmp_path.joinpath("web_config.json").write_text(json.dumps(data))

    cfg.load()

    value = cfg.get("default_k_avg")
    assert isinstance(value, float), f"Expected float, got {type(value)}"
    assert value == 10.0


def test_json_float_coerced_to_int_when_default_is_int(tmp_path):
    """
    If a default is int and the file stores a float, coerce back to int.
    """
    cfg = _make_config(tmp_path)
    data = {**cfg.config, "default_nodes": 1000.0}  # float instead of int
    tmp_path.joinpath("web_config.json").write_text(json.dumps(data))

    cfg.load()

    value = cfg.get("default_nodes")
    assert isinstance(value, int), f"Expected int, got {type(value)}"
    assert value == 1000


# ── Resilience ────────────────────────────────────────────────────────────────


def test_corrupted_config_file_falls_back_to_defaults(tmp_path):
    """A malformed JSON config must not crash — fall back to DEFAULTS."""
    from spkmc.web.config import WebConfig

    config_file = tmp_path / "web_config.json"
    config_file.write_text("{this is not valid json}")

    cfg = _make_config(tmp_path)
    cfg.load()

    assert cfg.get("data_directory") == WebConfig.DEFAULTS["data_directory"]


def test_missing_key_returns_provided_default(tmp_path):
    """get() must return the caller-supplied default for absent keys."""
    cfg = _make_config(tmp_path)
    assert cfg.get("nonexistent_key", "fallback") == "fallback"


def test_missing_key_returns_none_by_default(tmp_path):
    """get() must return None (not raise) for an absent key."""
    cfg = _make_config(tmp_path)
    assert cfg.get("nonexistent_key") is None


def test_load_merges_file_with_defaults(tmp_path):
    """
    A config file that only contains some keys must be merged with DEFAULTS
    so all expected keys remain present.
    """
    from spkmc.web.config import WebConfig

    partial = {"data_directory": "custom_data"}
    tmp_path.joinpath("web_config.json").write_text(json.dumps(partial))

    cfg = _make_config(tmp_path)
    cfg.load()

    # Custom value was kept
    assert cfg.get("data_directory") == "custom_data"
    # Default for a key absent from the file is still present
    assert cfg.get("default_nodes") == WebConfig.DEFAULTS["default_nodes"]


# ── Path helpers ──────────────────────────────────────────────────────────────


def test_get_data_path_returns_path_instance(tmp_path):
    cfg = _make_config(tmp_path)
    result = cfg.get_data_path()
    assert isinstance(result, Path)


def test_get_data_path_reflects_configured_value(tmp_path):
    cfg = _make_config(tmp_path)
    cfg.set("data_directory", "my_data")
    assert cfg.get_data_path() == Path("my_data")


def test_get_experiments_path_returns_path_instance(tmp_path):
    cfg = _make_config(tmp_path)
    result = cfg.get_experiments_path()
    assert isinstance(result, Path)


def test_get_experiments_path_reflects_configured_value(tmp_path):
    cfg = _make_config(tmp_path)
    cfg.set("experiments_directory", "my_experiments")
    assert cfg.get_experiments_path() == Path("my_experiments")


# ── OpenAI secrets ────────────────────────────────────────────────────────────


def test_set_and_read_openai_api_key_round_trip(tmp_path, monkeypatch):
    """
    set_openai_api_key writes to .streamlit/secrets.toml inside tmp_path.
    The subsequent read must return the same key.
    """
    from unittest.mock import MagicMock, patch

    from spkmc.web.config import WebConfig

    monkeypatch.chdir(tmp_path)

    WebConfig.set_openai_api_key("sk-test-abc123")

    secrets_file = tmp_path / ".streamlit" / "secrets.toml"
    assert secrets_file.exists()

    content = secrets_file.read_text()
    assert "sk-test-abc123" in content


def test_set_openai_api_key_preserves_existing_secrets(tmp_path, monkeypatch):
    """
    Writing a new API key must not clobber other secrets already in the file.
    """
    from spkmc.web.config import WebConfig

    monkeypatch.chdir(tmp_path)
    secrets_dir = tmp_path / ".streamlit"
    secrets_dir.mkdir()
    (secrets_dir / "secrets.toml").write_text('OTHER_SECRET = "keep_me"\n')

    WebConfig.set_openai_api_key("sk-new-key")

    content = (secrets_dir / "secrets.toml").read_text()
    assert "keep_me" in content
    assert "sk-new-key" in content


def test_set_openai_api_key_preserves_toml_structure(tmp_path, monkeypatch):
    """Structured TOML (sections, typed values, comments) must survive API key update."""
    from spkmc.web.config import WebConfig

    monkeypatch.chdir(tmp_path)
    secrets_dir = tmp_path / ".streamlit"
    secrets_dir.mkdir()

    original = (
        "# Top comment\n"
        'OPENAI_API_KEY = "sk-old"\n'
        "\n"
        "[database]\n"
        'host = "localhost"\n'
        "port = 5432\n"
    )
    (secrets_dir / "secrets.toml").write_text(original)

    WebConfig.set_openai_api_key("sk-new")

    content = (secrets_dir / "secrets.toml").read_text()
    # New key must be present, old value gone
    assert "sk-new" in content
    assert "sk-old" not in content
    # TOML structure must be preserved verbatim
    assert "[database]" in content
    assert 'host = "localhost"' in content
    assert "port = 5432" in content
    assert "# Top comment" in content


def test_set_openai_api_key_appends_when_key_absent(tmp_path, monkeypatch):
    """When OPENAI_API_KEY is not yet in the file, append without disturbing content."""
    from spkmc.web.config import WebConfig

    monkeypatch.chdir(tmp_path)
    secrets_dir = tmp_path / ".streamlit"
    secrets_dir.mkdir()

    original = "[other]\nfoo = true\n"
    (secrets_dir / "secrets.toml").write_text(original)

    WebConfig.set_openai_api_key("sk-appended")

    content = (secrets_dir / "secrets.toml").read_text()
    assert "sk-appended" in content
    assert "[other]" in content
    assert "foo = true" in content


def test_get_openai_api_key_returns_override_after_set(tmp_path, monkeypatch):
    """After set_openai_api_key(), get_openai_api_key() returns the new value."""
    import spkmc.web.config as config_mod
    from spkmc.web.config import WebConfig

    monkeypatch.chdir(tmp_path)
    # Reset module-level override to isolate from other tests
    monkeypatch.setattr(config_mod, "_api_key_override", None)

    WebConfig.set_openai_api_key("sk-first")
    assert WebConfig.get_openai_api_key() == "sk-first"

    WebConfig.set_openai_api_key("sk-second")
    assert WebConfig.get_openai_api_key() == "sk-second"


# ── atomic_json_write tests ──────────────────────────────────────────────────


class TestAtomicJsonWrite:
    """Tests for the atomic_json_write helper."""

    def test_writes_valid_json(self, tmp_path):
        from spkmc.web import atomic_json_write

        path = tmp_path / "test.json"
        data = {"key": "value", "num": 42}
        atomic_json_write(path, data)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_overwrites_existing_file(self, tmp_path):
        from spkmc.web import atomic_json_write

        path = tmp_path / "test.json"
        atomic_json_write(path, {"v": 1})
        atomic_json_write(path, {"v": 2})

        with open(path) as f:
            loaded = json.load(f)
        assert loaded == {"v": 2}

    def test_no_temp_file_left_on_success(self, tmp_path):
        from spkmc.web import atomic_json_write

        path = tmp_path / "test.json"
        atomic_json_write(path, {"a": 1})

        tmp_file = path.with_suffix(".json.tmp")
        assert not tmp_file.exists()

    def test_preserves_original_on_failure(self, tmp_path):
        from spkmc.web import atomic_json_write

        path = tmp_path / "test.json"
        original = {"original": True}
        atomic_json_write(path, original)

        # Attempt to write non-serializable data
        class BadObj:
            pass

        with pytest.raises(TypeError):
            atomic_json_write(path, {"bad": BadObj()})

        # Original should still be intact
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == original

        # Temp file should be cleaned up
        assert not path.with_suffix(".json.tmp").exists()
