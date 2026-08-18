"""Data quality validation for ERCOT market data.

All public functions follow a **raise-on-failure** contract: they return
``None`` on success and raise ``DataValidationError`` on the first detected
issue.  Callers that want a full summary of issues should use
``run_all_checks``, which collects all errors before raising.

Design principles
-----------------
* Functions are pure and stateless they accept a DataFrame and return or
  raise.  No I/O side effects.
* Every check is individually testable with synthetic data.
* Thresholds come from the ``Settings`` object, not hard-coded constants,
  so they can be adjusted without changing library code.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from loguru import logger

from energy_trading.config import Settings, settings as default_settings

# Column name constants keep in sync with data/ercot.py.
_COL_TIME = "interval_start_utc"
_COL_HUB = "hub"
_COL_LMP = "lmp"


class DataValidationError(ValueError):
    """Raised when one or more data quality checks fail."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        bullet_list = "\n  • ".join(errors)
        super().__init__(f"Data validation failed with {len(errors)} error(s):\n  • {bullet_list}")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_not_empty(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """Raise if ``df`` contains no rows.

    Args:
        df: DataFrame to inspect.
        label: Human-readable name used in the error message.

    Raises:
        DataValidationError: If ``df`` is empty.
    """
    if df.empty:
        raise DataValidationError([f"{label} is empty."])


def check_required_columns(df: pd.DataFrame, required: list[str], label: str = "") -> None:
    """Raise if any column in ``required`` is absent from ``df``.

    Args:
        df: DataFrame to inspect.
        required: Expected column names.
        label: Human-readable name used in the error message.

    Raises:
        DataValidationError: If one or more required columns are missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        prefix = f"{label}: " if label else ""
        raise DataValidationError([f"{prefix}Missing columns: {missing}"])


def check_no_duplicate_timestamps(
    df: pd.DataFrame,
    group_col: str | None = None,
    label: str = "",
) -> None:
    """Raise if any timestamp appears more than once within a group.

    Args:
        df: DataFrame with a DatetimeIndex.
        group_col: Optional column to group by before checking duplicates
            (e.g. ``"hub"`` for per-hub uniqueness checks).
        label: Human-readable name used in the error message.

    Raises:
        DataValidationError: If duplicates are found.
    """
    if group_col is not None and group_col in df.columns:
        errors: list[str] = []
        for group_value, group_df in df.groupby(group_col):
            dupes = group_df.index[group_df.index.duplicated(keep=False)]
            if not dupes.empty:
                errors.append(
                    f"{label or 'DataFrame'} [{group_col}={group_value!r}] has "
                    f"{len(dupes)} duplicate timestamp(s), e.g. {dupes[0]}"
                )
        if errors:
            raise DataValidationError(errors)
    else:
        dupes = df.index[df.index.duplicated(keep=False)]
        if not dupes.empty:
            prefix = f"{label}: " if label else ""
            raise DataValidationError(
                [
                    f"{prefix}{len(dupes)} duplicate timestamp(s) found, "
                    f"e.g. {dupes[0]}"
                ]
            )


def check_hourly_coverage(
    df: pd.DataFrame,
    start: date,
    end: date,
    max_missing_pct: float | None = None,
    cfg: Settings | None = None,
    label: str = "",
) -> None:
    """Raise if too many hourly timestamps are absent between ``start`` and ``end``.

    Args:
        df: DataFrame with a DatetimeIndex (UTC, hourly).
        start: Expected first date (inclusive).
        end: Expected last date (inclusive).
        max_missing_pct: Maximum fraction of missing hours before failing.
            Defaults to ``cfg.max_missing_hours_pct``.
        cfg: Settings instance.  Defaults to the module-level singleton.
        label: Human-readable name used in the error message.

    Raises:
        DataValidationError: If the fraction of missing hours exceeds the
            threshold.
    """
    cfg = cfg or default_settings
    threshold = max_missing_pct if max_missing_pct is not None else cfg.max_missing_hours_pct

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(
            [
                f"{label + ': ' if label else ''}Index is not a DatetimeIndex "
                f"(got {type(df.index).__name__}); cannot check hourly coverage."
            ]
        )

    expected = pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC") + timedelta(hours=23),
        freq="h",
    )
    actual_hours = df.index.floor("h").unique()
    missing = expected.difference(actual_hours)
    missing_pct = len(missing) / len(expected)

    if missing_pct > threshold:
        prefix = f"{label}: " if label else ""
        raise DataValidationError(
            [
                f"{prefix}{len(missing)} missing hours out of {len(expected)} "
                f"({missing_pct:.1%} > threshold {threshold:.1%}).  "
                f"First missing: {missing[0] if len(missing) else 'N/A'}"
            ]
        )

    if missing_pct > 0:
        logger.warning(
            "{}{} missing hour(s) out of {} ({:.2%}) within tolerance.",
            f"{label}: " if label else "",
            len(missing),
            len(expected),
            missing_pct,
        )


def check_required_hubs(
    df: pd.DataFrame,
    required_hubs: list[str] | None = None,
    hub_col: str = _COL_HUB,
    cfg: Settings | None = None,
    label: str = "",
) -> None:
    """Raise if any expected hub is absent from the data.

    Args:
        df: LMP DataFrame containing a hub column.
        required_hubs: Expected hub names.  Defaults to ``cfg.hubs``.
        hub_col: Column name that holds hub identifiers.
        cfg: Settings instance.  Defaults to the module-level singleton.
        label: Human-readable name used in the error message.

    Raises:
        DataValidationError: If one or more hubs are missing.
    """
    cfg = cfg or default_settings
    expected = set(required_hubs or cfg.hubs)

    if hub_col not in df.columns:
        raise DataValidationError(
            [f"{label or 'DataFrame'}: hub column {hub_col!r} not found."]
        )

    present = set(df[hub_col].unique())
    missing = expected - present
    if missing:
        prefix = f"{label}: " if label else ""
        raise DataValidationError([f"{prefix}Missing hubs: {sorted(missing)}"])


def check_price_bounds(
    df: pd.DataFrame,
    lmp_col: str = _COL_LMP,
    cfg: Settings | None = None,
    label: str = "",
) -> None:
    """Log a warning if any LMP is outside plausible ERCOT price bounds.

    ERCOT's theoretical bounds are roughly -$150/MWh to $9,000/MWh, but
    historical scarcity events (e.g. Winter Storm Uri) have pushed prices
    to the administrative cap.  This function warns on extreme values rather
    than failing the pipeline, since such prices are real market events.

    Args:
        df: DataFrame containing an LMP column.
        lmp_col: Column name for LMP values.
        cfg: Settings instance.  Defaults to the module-level singleton.
        label: Human-readable name used in the error message.

    Raises:
        DataValidationError: If ``lmp_col`` is absent from ``df``.
    """
    cfg = cfg or default_settings

    if lmp_col not in df.columns:
        raise DataValidationError(
            [f"{label or 'DataFrame'}: LMP column {lmp_col!r} not found."]
        )

    prices = df[lmp_col].dropna()

    below_floor = prices[prices < cfg.price_floor]
    above_cap = prices[prices > cfg.price_cap]

    prefix = f"{label}: " if label else ""
    if not below_floor.empty:
        logger.warning(
            "{}{}  price(s) below floor of ${:.0f}/MWh. Min: ${:.2f}",
            prefix,
            len(below_floor),
            cfg.price_floor,
            below_floor.min(),
        )
    if not above_cap.empty:
        logger.warning(
            "{}{}  price(s) above cap of ${:.0f}/MWh. Max: ${:.2f}",
            prefix,
            len(above_cap),
            cfg.price_cap,
            above_cap.max(),
        )


def check_no_all_nan_columns(df: pd.DataFrame, label: str = "") -> None:
    """Raise if any column is entirely NaN.

    Args:
        df: DataFrame to inspect.
        label: Human-readable name used in the error message.

    Raises:
        DataValidationError: If one or more columns are all-NaN.
    """
    all_nan = [c for c in df.columns if df[c].isna().all()]
    if all_nan:
        prefix = f"{label}: " if label else ""
        raise DataValidationError([f"{prefix}Columns with all NaN values: {all_nan}"])


# ---------------------------------------------------------------------------
# Composite validators
# ---------------------------------------------------------------------------


def validate_lmp_dataframe(
    df: pd.DataFrame,
    start: date,
    end: date,
    cfg: Settings | None = None,
    label: str = "LMP",
) -> None:
    """Run the full suite of LMP quality checks.

    Checks are collected and all failures are reported in a single
    ``DataValidationError`` rather than stopping at the first issue.

    Args:
        df: LMP DataFrame returned by ``ERCOTClient.get_dam_prices`` or
            ``get_rtm_prices``.
        start: Expected start date of the data range.
        end: Expected end date of the data range.
        cfg: Settings instance.  Defaults to the module-level singleton.
        label: Human-readable label for error messages (e.g. ``"DAM"``).

    Raises:
        DataValidationError: If any check fails.
    """
    cfg = cfg or default_settings
    errors: list[str] = []

    def _run(check_fn, *args, **kwargs) -> None:
        try:
            check_fn(*args, **kwargs)
        except DataValidationError as exc:
            errors.extend(exc.errors)

    _run(check_not_empty, df, label)
    _run(check_required_columns, df, [_COL_HUB, _COL_LMP], label)
    _run(check_no_duplicate_timestamps, df, _COL_HUB, label)
    _run(check_required_hubs, df, cfg=cfg, label=label)
    _run(check_hourly_coverage, df, start, end, cfg=cfg, label=label)
    _run(check_no_all_nan_columns, df, label)
    _run(check_price_bounds, df, cfg=cfg, label=label)

    if errors:
        raise DataValidationError(errors)

    logger.info("{} validation passed ({} rows).", label, len(df))
