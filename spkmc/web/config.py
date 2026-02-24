"""
Web interface configuration management.

Handles loading and saving web preferences (stored as JSON) and reading secrets
from Streamlit's secrets.toml file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, cast

import streamlit as st

# In-memory override for the API key, used to bypass st.secrets caching.
# st.secrets is process-cached and has no public invalidation API; writing
# to secrets.toml does not update the in-memory singleton.  After
# set_openai_api_key(), all subsequent reads in this process see the new
# value via this override.  On process restart the override resets to None
# and the freshly-loaded st.secrets provides the correct value from disk.
_api_key_override: Optional[str] = None


class WebConfig:
    """Manages web interface configuration and secrets."""

    CONFIG_FILE: Path = Path(
        os.environ.get(
            "SPKMC_WEB_CONFIG_FILE",
            str(Path.home() / ".spkmc" / "web_config.json"),
        )
    )

    # Default configuration values
    DEFAULTS = {
        "data_directory": "data",
        "experiments_directory": "experiments",
        "theme": "light",
        "chart_height": 500,
        "chart_color_s": "#4477AA",
        "chart_color_i": "#EE6677",
        "chart_color_r": "#228833",
        "chart_template": "plotly_white",
        "default_network_type": "er",
        "default_distribution": "gamma",
        "default_nodes": 1000,
        "default_k_avg": 10.0,
        "default_samples": 50,
        "default_num_runs": 2,
        "default_initial_perc": 0.01,
        "default_t_max": 10.0,
        "default_steps": 100,
        "default_shape": 2.0,
        "default_scale": 1.0,
        "default_mu": 1.0,
        "default_lambda": 1.0,
        "default_exponent": 2.5,
        "ai_model": "gpt-4o-mini",
    }

    def __init__(self) -> None:
        """Initialize configuration manager."""
        self.config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from JSON file, creating with defaults if not found."""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, "r") as f:
                    loaded = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    merged = {**self.DEFAULTS, **loaded}
                    # Coerce types to match defaults (JSON may deserialize
                    # 10.0 as int 10, which causes StreamlitMixedNumericTypesError)
                    for key, default_val in self.DEFAULTS.items():
                        if key in merged:
                            if isinstance(default_val, float) and isinstance(merged[key], int):
                                merged[key] = float(merged[key])
                            elif isinstance(default_val, int) and isinstance(merged[key], float):
                                merged[key] = int(merged[key])
                    self.config = merged
            except (json.JSONDecodeError, IOError):
                # If file is corrupted, start with defaults
                self.config = self.DEFAULTS.copy()
        else:
            # Create config directory if it doesn't exist
            self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.config = self.DEFAULTS.copy()
            self.save()

    def save(self) -> None:
        """Save current configuration to JSON file."""
        from spkmc.web import atomic_json_write

        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(self.CONFIG_FILE, self.config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value and save."""
        self.config[key] = value
        self.save()

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values at once."""
        self.config.update(updates)
        self.save()

    @staticmethod
    def get_openai_api_key() -> Optional[str]:
        """
        Get OpenAI API key.

        Returns the in-memory override (set by ``set_openai_api_key``) if
        present, otherwise falls back to ``st.secrets``.

        Returns:
            API key if found, None otherwise
        """
        if _api_key_override is not None:
            return _api_key_override
        try:
            return cast(Optional[str], st.secrets.get("OPENAI_API_KEY", None))
        except (FileNotFoundError, KeyError):
            return None

    @staticmethod
    def set_openai_api_key(api_key: str) -> None:
        """
        Set OpenAI API key in Streamlit secrets and update in-memory cache.

        Writes to ``.streamlit/secrets.toml`` for persistence across restarts,
        and updates ``_api_key_override`` so subsequent reads in this process
        see the new value immediately (st.secrets is process-cached with no
        public invalidation API).

        Args:
            api_key: The OpenAI API key to save
        """
        global _api_key_override
        _api_key_override = api_key
        secrets_file = Path(".streamlit") / "secrets.toml"
        secrets_file.parent.mkdir(exist_ok=True)

        # Escape the value for TOML (double-quote string)
        escaped_value = api_key.replace("\\", "\\\\").replace('"', '\\"')
        new_line = f'OPENAI_API_KEY = "{escaped_value}"'

        # Pattern that matches an existing OPENAI_API_KEY assignment
        key_pattern = re.compile(r"^OPENAI_API_KEY\s*=\s*.*$", re.MULTILINE)

        if secrets_file.exists():
            content = secrets_file.read_text()
            if key_pattern.search(content):
                # Replace existing key in-place, preserving all other content
                content = key_pattern.sub(new_line, content)
            else:
                # Append to end, ensuring a leading newline
                if content and not content.endswith("\n"):
                    content += "\n"
                content += new_line + "\n"
            secrets_file.write_text(content)
        else:
            # Create new file with just this key
            secrets_file.write_text(
                "# Streamlit secrets for SPKMC web interface\n" + new_line + "\n"
            )

    def get_data_path(self) -> Path:
        """Get the data directory path."""
        return Path(self.get("data_directory", "data"))

    def get_experiments_path(self) -> Path:
        """Get the experiments directory path."""
        return Path(self.get("experiments_directory", "experiments"))
