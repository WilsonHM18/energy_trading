#!/usr/bin/env python3
"""Generate all portfolio plots for the ERCOT DA/RT spread trading project.

Reads pre-computed parquet files from data/processed/ and saves publication-
quality PNG figures to docs/plots/.

Usage
-----
    uv run python scripts/generate_plots.py

Output files
------------
docs/plots/equity_curve.png          Cumulative P&L per model (HB_SOUTH OOS)
docs/plots/walk_forward_performance.png Direction accuracy by fold & model
docs/plots/regime_decomposition.png  Sharpe ratio by market regime
docs/plots/feature_importance.png    Top-15 LightGBM feature importances
docs/plots/tc_sensitivity.png        Sharpe & P&L vs transaction cost
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
PLOT_DIR = ROOT / "docs" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)

MODEL_COLORS = {
    "linear": "#2196F3",   # blue
    "ridge":  "#4CAF50",   # green
    "lasso":  "#FF9800",   # orange
    "lgbm":   "#7B1FA2",   # purple (best model)
}
MODEL_LABELS = {
    "linear": "OLS Linear",
    "ridge":  "Ridge (L2)",
    "lasso":  "Lasso (L1)",
    "lgbm":   "LightGBM",
}


# ---------------------------------------------------------------------------
# Helper: load parquet (fix interval_start_utc stored-as-column)
# ---------------------------------------------------------------------------

def _load(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    if "interval_start_utc" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("interval_start_utc")
    return df


# ---------------------------------------------------------------------------
# 1. Equity curve
# ---------------------------------------------------------------------------

def plot_equity_curves() -> None:
    """Cumulative net P&L for all 4 models on HB_SOUTH (2021-2024 OOS)."""
    print("  [1/5] Equity curves …", end="", flush=True)

    from energy_trading.backtest import run_backtest

    oos_df = _load("oos_predictions")
    hub = "HB_SOUTH"
    models = ["linear", "ridge", "lasso", "lgbm"]

    fig, ax = plt.subplots(figsize=(13, 5))

    trade_by_model: dict = {}
    for model in models:
        trade_df, _ = run_backtest(oos_df, model=model, hub=hub)
        trade_by_model[model] = trade_df
        cum_pnl = trade_df["cumulative_pnl"]
        ax.plot(
            cum_pnl.index,
            cum_pnl.values,
            color=MODEL_COLORS[model],
            linewidth=1.8 if model == "lgbm" else 1.2,
            alpha=1.0 if model == "lgbm" else 0.7,
            label=MODEL_LABELS[model],
            zorder=4 if model == "lgbm" else 3,
        )

    # Drawdown shading for LightGBM (reuse result from loop above)
    cum = trade_by_model["lgbm"]["cumulative_pnl"]
    hwm = cum.cummax()
    ax.fill_between(
        cum.index,
        hwm.values,
        cum.values,
        where=(cum < hwm),
        color=MODEL_COLORS["lgbm"],
        alpha=0.12,
        label="LightGBM drawdown",
        zorder=2,
    )

    # Horizontal zero line
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    # Shade Winter Storm Uri
    uri_start = pd.Timestamp("2021-02-10", tz="UTC")
    uri_end = pd.Timestamp("2021-02-21", tz="UTC")
    ax.axvspan(uri_start, uri_end, color="#FF5722", alpha=0.08, label="Winter Storm Uri")

    ax.set_title("Cumulative Net P&L ERCOT Virtual Spread Strategy (HB_SOUTH, 1 MWh, tc=0)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "equity_curve.png")
    plt.close(fig)
    print(" saved.")


# ---------------------------------------------------------------------------
# 2. Walk-forward fold performance
# ---------------------------------------------------------------------------

def plot_walk_forward_performance() -> None:
    """Direction accuracy by fold and model for HB_SOUTH."""
    print("  [2/5] Walk-forward performance …", end="", flush=True)

    wf = pd.read_parquet(DATA_DIR / "walk_forward_metrics.parquet")
    hub_df = wf[wf["hub"] == "HB_SOUTH"].copy()

    test_years = sorted(hub_df["test_year"].unique())
    model_order = ["linear", "ridge", "lasso", "lgbm"]
    n_models = len(model_order)
    n_years = len(test_years)

    x = np.arange(n_years)
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Direction accuracy ---
    ax = axes[0]
    for i, model in enumerate(model_order):
        mdf = hub_df[hub_df["model"] == model].sort_values("test_year")
        vals = mdf["direction_accuracy"].values * 100
        bars = ax.bar(
            x + (i - n_models / 2 + 0.5) * width,
            vals,
            width=width * 0.9,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            alpha=0.88,
        )

    ax.axhline(50, color="black", linewidth=1.0, linestyle="--", alpha=0.5, label="Random (50%)")
    ax.set_title("Direction Accuracy by Fold HB_SOUTH")
    ax.set_xlabel("Test Year")
    ax.set_ylabel("Direction Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in test_years])
    ax.set_ylim(40, 75)
    ax.legend(loc="upper left", fontsize=9)
    # Annotate Uri 2021 fold
    ax.annotate(
        "Uri\nstorm",
        xy=(0, 50.5),
        xytext=(0.15, 42),
        fontsize=8,
        color="#FF5722",
        arrowprops=dict(arrowstyle="->", color="#FF5722", lw=1),
    )

    # --- RMSE ---
    ax = axes[1]
    for i, model in enumerate(model_order):
        mdf = hub_df[hub_df["model"] == model].sort_values("test_year")
        vals = mdf["rmse"].values
        ax.bar(
            x + (i - n_models / 2 + 0.5) * width,
            vals,
            width=width * 0.9,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            alpha=0.88,
        )

    ax.set_title("RMSE by Fold HB_SOUTH")
    ax.set_xlabel("Test Year")
    ax.set_ylabel("RMSE ($/MWh)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in test_years])
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Walk-Forward Cross-Validation: Out-of-Sample Model Performance", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "walk_forward_performance.png")
    plt.close(fig)
    print(" saved.")


# ---------------------------------------------------------------------------
# 3. Regime decomposition
# ---------------------------------------------------------------------------

def plot_regime_decomposition() -> None:
    """Horizontal bar chart of Sharpe ratio by market regime."""
    print("  [3/5] Regime decomposition …", end="", flush=True)

    regime_df = pd.read_parquet(DATA_DIR / "risk_regime_metrics.parquet")

    # Ordered subset for clarity
    regime_order = [
        "uri_storm", "low_vol", "peak", "low_wind", "low_gas",
        "all",
        "high_gas", "high_wind", "off_peak", "high_vol", "non_uri",
    ]
    regime_labels = {
        "all":       "All hours",
        "peak":      "Peak hours (HE 7–22 weekdays)",
        "off_peak":  "Off-peak hours",
        "uri_storm": "Winter Storm Uri (Feb 2021)",
        "non_uri":   "Non-Uri hours",
        "high_vol":  "High volatility (σ ≥ median)",
        "low_vol":   "Low volatility (σ < median)",
        "high_wind": "High wind (≥ median)",
        "low_wind":  "Low wind (< median)",
        "high_gas":  "High gas price (≥ median)",
        "low_gas":   "Low gas price (< median)",
    }

    # Filter to available regimes in order
    available = [r for r in regime_order if r in regime_df["regime"].values]
    plot_df = regime_df.set_index("regime").loc[available]

    sharpe_vals = plot_df["sharpe"].values
    labels = [regime_labels.get(r, r) for r in available]

    # Color by above/below "all" benchmark
    all_sharpe = float(regime_df.loc[regime_df["regime"] == "all", "sharpe"].iloc[0])
    colors = ["#7B1FA2" if v >= all_sharpe else "#90A4AE" for v in sharpe_vals]
    # Uri gets its own highlight
    if "uri_storm" in available:
        uri_idx = available.index("uri_storm")
        colors[uri_idx] = "#FF5722"

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(available))
    bars = ax.barh(y_pos, sharpe_vals, color=colors, alpha=0.88, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, sharpe_vals):
        ax.text(
            max(val + 0.05, 0.1),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.axvline(all_sharpe, color="black", linewidth=1.2, linestyle="--", alpha=0.5,
               label=f"Overall Sharpe = {all_sharpe:.2f}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Annualised Sharpe Ratio")
    ax.set_title("P&L Performance by Market Regime LightGBM / HB_SOUTH (OOS 2021–2024)")
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "regime_decomposition.png")
    plt.close(fig)
    print(" saved.")


# ---------------------------------------------------------------------------
# 4. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance() -> None:
    """Top-15 LightGBM feature importances (trained on 2020-2023 HB_SOUTH)."""
    print("  [4/5] Feature importance …", end="", flush=True)

    from energy_trading.features.engineering import get_feature_columns
    from energy_trading.models.train import build_models, get_feature_importance

    features_df = _load("features_dataset")

    hub = "HB_SOUTH"
    hub_df = features_df[features_df["hub"] == hub].copy()

    # Train on 2020-2023, leaving 2024 as unseen
    train_mask = hub_df.index.year <= 2023
    train_df = hub_df[train_mask].dropna()

    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols].values
    y_train = train_df["spread"].values

    models = build_models()
    lgbm = models["lgbm"]
    lgbm.fit(X_train, y_train)

    importance = get_feature_importance(lgbm, feature_cols).head(15)

    # Friendly display names
    name_map = {
        "spread_lag_1d":           "Spread lag 1 day",
        "spread_lag_7d":           "Spread lag 7 days",
        "spread_lag_1w":           "Spread lag 1 week",
        "spread_lag_2w":           "Spread lag 2 weeks",
        "spread_lag_4w":           "Spread lag 4 weeks",
        "spread_lag_2d":           "Spread lag 2 days",
        "spread_vol_24h":          "Spread volatility 24h",
        "spread_vol_168h":         "Spread volatility 168h",
        "dam_lmp_lag_1d":          "DAM LMP lag 1 day",
        "dam_lmp_lag_7d":          "DAM LMP lag 7 days",
        "rtm_lmp_lag_1d":          "RTM LMP lag 1 day",
        "rtm_lmp_lag_7d":          "RTM LMP lag 7 days",
        "is_peak":                 "Is peak hour",
        "hour_cpt":                "Hour of day (CPT)",
        "hour_utc":                "Hour of day (UTC)",
        "dow":                     "Day of week",
        "month":                   "Month",
        "load_mw":                 "Load (MW)",
        "load_mw_lag1h":           "Load lag 1h",
        "load_mw_lag1d":           "Load lag 24h",
        "load_ramp_1h":            "Load ramp 1h",
        "load_ramp_24h":           "Load ramp 24h",
        "wind_actual_mw":          "Wind generation (MW)",
        "wind_actual_mw_lag1d":    "Wind lag 1 day",
        "solar_actual_mw":         "Solar generation (MW)",
        "solar_actual_mw_lag1d":   "Solar lag 1 day",
        "gas_price_mmbtu":         "Henry Hub gas price",
        "gas_price_lag1d":         "Gas price lag 1 day",
        "gas_elec_spread":         "Gas-to-power spread",
        "load_forecast_mw":        "Load forecast (MW)",
        "load_forecast_error":     "Load forecast error",
        "west_north_spread_diff":  "West–North spread diff",
        "sin_hour":                "sin(hour)",
        "cos_hour":                "cos(hour)",
        "sin_month":               "sin(month)",
        "cos_month":               "cos(month)",
        "quarter":                 "Quarter",
        "is_weekend":              "Is weekend",
    }

    labels = [name_map.get(f, f) for f in importance.index]

    # Colour bars by feature group
    group_colors = {
        "Spread lag": "#7B1FA2",
        "Spread vol": "#9C27B0",
        "Price level": "#2196F3",
        "Calendar": "#4CAF50",
        "Load": "#FF9800",
        "Wind": "#00BCD4",
        "Solar": "#FFEB3B",
        "Gas": "#F44336",
        "Forecast": "#009688",
        "Other": "#90A4AE",
    }

    def _group(name: str) -> str:
        if "lag" in name.lower() and "spread" in name.lower():
            return "Spread lag"
        if "spread vol" in name.lower() or "volatility" in name.lower():
            return "Spread vol"
        if "lmp lag" in name.lower() or "price level" in name.lower():
            return "Price level"
        if any(w in name.lower() for w in ("hour", "day of week", "month", "quarter", "weekend", "peak", "sin", "cos")):
            return "Calendar"
        if "load forecast error" in name.lower():
            return "Forecast"
        if "load" in name.lower():
            return "Load"
        if "wind" in name.lower():
            return "Wind"
        if "solar" in name.lower():
            return "Solar"
        if "gas" in name.lower():
            return "Gas"
        return "Other"

    bar_colors = [group_colors[_group(lbl)] for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, importance.values, color=bar_colors, alpha=0.88,
                   edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("Top-15 LightGBM Feature Importances HB_SOUTH (Trained 2020–2023)")
    ax.invert_yaxis()

    # Legend for groups
    seen_groups: set[str] = set()
    for lbl, color in zip(labels, bar_colors):
        grp = _group(lbl)
        if grp not in seen_groups:
            ax.bar(0, 0, color=color, label=grp, alpha=0.88)
            seen_groups.add(grp)
    ax.legend(loc="lower right", fontsize=9, title="Feature group")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "feature_importance.png")
    plt.close(fig)
    print(" saved.")


# ---------------------------------------------------------------------------
# 5. Transaction-cost sensitivity
# ---------------------------------------------------------------------------

def plot_tc_sensitivity() -> None:
    """Dual-axis: Sharpe ratio and total P&L vs transaction cost ($/MWh)."""
    print("  [5/5] TC sensitivity …", end="", flush=True)

    tc_df = pd.read_parquet(DATA_DIR / "risk_tc_sensitivity.parquet")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    color_sharpe = MODEL_COLORS["lgbm"]
    color_pnl = "#FF9800"

    ax1.plot(tc_df["tc"], tc_df["sharpe"], color=color_sharpe, linewidth=2.2,
             marker="o", markersize=7, label="Sharpe ratio (left)")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax1.set_xlabel("Round-Trip Transaction Cost ($/MWh)")
    ax1.set_ylabel("Annualised Sharpe Ratio", color=color_sharpe)
    ax1.tick_params(axis="y", colors=color_sharpe)

    ax2.plot(tc_df["tc"], tc_df["total_pnl"] / 1_000, color=color_pnl,
             linewidth=2.0, marker="s", markersize=6, linestyle="--",
             label="Total P&L (right, $k)")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.3)
    ax2.set_ylabel("Total P&L ($k)", color=color_pnl)
    ax2.tick_params(axis="y", colors=color_pnl)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    ax1.set_title("Transaction-Cost Sensitivity LightGBM / HB_SOUTH (OOS 2021–2024)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tc_sensitivity.png")
    plt.close(fig)
    print(" saved.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Add project src to path so imports work when called as a script
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    print(f"Generating plots -> {PLOT_DIR}")
    plot_equity_curves()
    plot_walk_forward_performance()
    plot_regime_decomposition()
    plot_feature_importance()
    plot_tc_sensitivity()
    print(f"\nDone. All 5 plots saved to {PLOT_DIR}")
