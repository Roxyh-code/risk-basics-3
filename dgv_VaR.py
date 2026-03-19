#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class DgvVarConfig:
    spx_symbol: str = "^GSPC"
    vix_symbol: str = "^VIX"
    start_date: str = "2023-11-09"
    end_date_exclusive: str = "2024-11-09"
    current_spx_level: float = 5995.0
    strike: float = 5995.0
    risk_free_rate: float = 0.05
    current_iv: float = 0.15
    days_to_maturity: int = 28
    tail_probability: float = 0.01


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def black_scholes_call_greeks(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    time_to_maturity: float,
) -> tuple[float, float, float]:
    if sigma <= 0 or time_to_maturity <= 0:
        raise ValueError("Sigma and time_to_maturity must be positive.")

    d1 = (
        log(spot / strike)
        + (rate + 0.5 * sigma * sigma) * time_to_maturity
    ) / (sigma * sqrt(time_to_maturity))

    delta = normal_cdf(d1)
    gamma = normal_pdf(d1) / (spot * sigma * sqrt(time_to_maturity))
    vega = spot * normal_pdf(d1) * sqrt(time_to_maturity)

    return delta, gamma, vega


def load_close_series(symbol: str, start_date: str, end_date_exclusive: str) -> pd.Series:
    close_series = yf.download(
        symbol,
        start=start_date,
        end=end_date_exclusive,
        progress=False,
        auto_adjust=False,
    )["Close"]

    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]

    close_series = close_series.dropna()

    if close_series.empty:
        raise ValueError(f"No close data returned for {symbol}.")

    close_series.name = "close"
    return close_series


def compute_daily_returns(close_series: pd.Series) -> pd.Series:
    daily_returns = close_series.pct_change().dropna()
    daily_returns.name = "daily_return"
    return daily_returns


def build_dgv_scenario_table(config: DgvVarConfig) -> tuple[pd.DataFrame, float, float, float]:
    spx_close = load_close_series(
        symbol=config.spx_symbol,
        start_date=config.start_date,
        end_date_exclusive=config.end_date_exclusive,
    )
    vix_close = load_close_series(
        symbol=config.vix_symbol,
        start_date=config.start_date,
        end_date_exclusive=config.end_date_exclusive,
    )

    spx_return = compute_daily_returns(spx_close).rename("spx_return")
    vix_return = compute_daily_returns(vix_close).rename("vix_return")

    scenario_table = pd.concat(
        [spx_close.rename("spx_close"), vix_close.rename("vix_close"), spx_return, vix_return],
        axis=1,
    ).dropna()

    time_to_maturity = config.days_to_maturity / 365.0

    delta, gamma, vega = black_scholes_call_greeks(
        spot=config.current_spx_level,
        strike=config.strike,
        rate=config.risk_free_rate,
        sigma=config.current_iv,
        time_to_maturity=time_to_maturity,
    )

    scenario_table["simulated_spx_level"] = config.current_spx_level * (1.0 + scenario_table["spx_return"])
    scenario_table["simulated_iv"] = config.current_iv * (1.0 + scenario_table["vix_return"])

    scenario_table["delta_s"] = scenario_table["simulated_spx_level"] - config.current_spx_level
    scenario_table["delta_sigma"] = scenario_table["simulated_iv"] - config.current_iv

    scenario_table["dgv_pnl"] = (
        delta * scenario_table["delta_s"]
        + 0.5 * gamma * scenario_table["delta_s"] ** 2
        + vega * scenario_table["delta_sigma"]
    )

    return scenario_table, delta, gamma, vega


def compute_var_from_pnl(pnl_series: pd.Series, tail_probability: float) -> float:
    left_tail_pnl = np.quantile(
        pnl_series.dropna().to_numpy(),
        tail_probability,
        method="lower",
    )
    return float(-left_tail_pnl)


def print_summary(
    config: DgvVarConfig,
    scenario_count: int,
    delta: float,
    gamma: float,
    vega: float,
    var_value: float,
) -> None:
    confidence_level = 1.0 - config.tail_probability

    print("SPX Option 1-Day Delta-Gamma-Vega VaR")
    print("-" * 40)
    print(f"SPX symbol         : {config.spx_symbol}")
    print(f"VIX symbol         : {config.vix_symbol}")
    print(f"Start date         : {config.start_date}")
    print(f"End date           : {config.end_date_exclusive} (exclusive)")
    print(f"Current SPX        : {config.current_spx_level:.2f}")
    print(f"Strike             : {config.strike:.2f}")
    print(f"Current IV         : {config.current_iv:.4f}")
    print(f"Risk-free rate     : {config.risk_free_rate:.4f}")
    print(f"Days to maturity   : {config.days_to_maturity}")
    print(f"Confidence level   : {confidence_level:.2%}")
    print(f"Scenario count     : {scenario_count}")
    print(f"Delta              : {delta:.6f}")
    print(f"Gamma              : {gamma:.8f}")
    print(f"Vega               : {vega:.6f}")
    print(f"1-day DGV VaR      : {var_value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute 1-day 99% Delta-Gamma-Vega VaR for an ATM SPX call option."
    )
    parser.add_argument(
        "--output-file",
        default="spx_option_dgv_var.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()

    config = DgvVarConfig()

    scenario_table, delta, gamma, vega = build_dgv_scenario_table(config)
    var_value = compute_var_from_pnl(
        pnl_series=scenario_table["dgv_pnl"],
        tail_probability=config.tail_probability,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_table.to_csv(output_path, index=True)

    print_summary(
        config=config,
        scenario_count=len(scenario_table),
        delta=delta,
        gamma=gamma,
        vega=vega,
        var_value=var_value,
    )
    print(f"\nSaved output to: {output_path}")


if __name__ == "__main__":
    main()
