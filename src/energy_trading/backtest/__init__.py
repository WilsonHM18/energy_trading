"""Backtest engine and risk analysis for ERCOT DA/RT virtual spread trading.

Public API
----------
from energy_trading.backtest import (
    generate_signals,
    compute_hourly_pnl,
    run_backtest,
    run_all_backtests,
    compute_pnl_metrics,
    identify_drawdown_periods,
    compute_regime_metrics,
    compute_tail_risk,
    tc_sensitivity,
    compute_rolling_sharpe,
    parameter_robustness,
)
"""

from energy_trading.backtest.engine import (
    compute_hourly_pnl,
    generate_signals,
    run_all_backtests,
    run_backtest,
)
from energy_trading.backtest.metrics import compute_pnl_metrics
from energy_trading.backtest.risk import (
    compute_regime_metrics,
    compute_rolling_sharpe,
    compute_tail_risk,
    identify_drawdown_periods,
    parameter_robustness,
    tc_sensitivity,
)

__all__ = [
    "generate_signals",
    "compute_hourly_pnl",
    "run_backtest",
    "run_all_backtests",
    "compute_pnl_metrics",
    "identify_drawdown_periods",
    "compute_regime_metrics",
    "compute_tail_risk",
    "tc_sensitivity",
    "compute_rolling_sharpe",
    "parameter_robustness",
]
