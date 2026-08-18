"""Central configuration for the energy-trading package.

All runtime parameters are loaded from environment variables (prefixed ``ET_``)
or from a ``.env`` file in the project root.  Defaults are chosen so the
pipeline runs out-of-the-box without any manual configuration.

Example .env::

    ET_HUBS=["HB_NORTH","HB_SOUTH","HB_WEST","HB_HOUSTON"]
    ET_DATA_DIR=/abs/path/to/data
    ET_LOG_LEVEL=DEBUG
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Project root the directory that contains ``pyproject.toml``.
# This is resolved at import time so paths work regardless of the caller's cwd.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application-wide configuration.

    All fields can be overridden via environment variables with the ``ET_``
    prefix or via a ``.env`` file located at the project root.
    """

    model_config = SettingsConfigDict(
        env_prefix="ET_",
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_parse_none_str="null",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Market configuration
    # ------------------------------------------------------------------
    hubs: list[str] = Field(
        default=["HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON"],
        description="ERCOT settlement-point hubs to include in all datasets.",
    )

    # ------------------------------------------------------------------
    # File-system paths
    # ------------------------------------------------------------------
    data_dir: Path = Field(
        default=_PROJECT_ROOT / "data",
        description="Root directory for raw and processed data files.",
    )

    # ------------------------------------------------------------------
    # External API keys
    # ------------------------------------------------------------------
    eia_api_key: str | None = Field(
        default=None,
        description=(
            "EIA Open Data API key.  Required for wind/solar generation and "
            "gas price features.  Register free at https://www.eia.gov/opendata/"
        ),
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: Annotated[str, Field(description="Loguru log level.")] = "INFO"

    # ------------------------------------------------------------------
    # Data-quality thresholds
    # ------------------------------------------------------------------
    price_floor: float = Field(
        default=-500.0,
        description="Minimum plausible LMP ($/MWh).  Prices below this trigger a warning.",
    )
    price_cap: float = Field(
        default=10_000.0,
        description="Maximum plausible LMP ($/MWh).  Prices above this trigger a warning.",
    )
    max_missing_hours_pct: float = Field(
        default=0.01,
        description="Maximum fraction of hourly observations allowed to be missing.",
    )

    # ------------------------------------------------------------------
    # Derived paths (not configurable directly)
    # ------------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        """Directory for unmodified downloaded data."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Directory for cleaned and feature-engineered data."""
        return self.data_dir / "processed"

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        valid = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {value!r}")
        return upper

    @field_validator("hubs")
    @classmethod
    def _validate_hubs(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("hubs list must contain at least one entry.")
        return [h.upper() for h in value]


# ---------------------------------------------------------------------------
# Module-level singleton import ``settings`` everywhere instead of
# constructing a new Settings() object per call.
# ---------------------------------------------------------------------------
settings = Settings()
