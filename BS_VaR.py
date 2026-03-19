#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class BsVarConfig:
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


def black_scholes_call_price(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    time_to_maturity: float,
) -> float:
    if sigma <= 0 or time_to_maturity <= 0:
        return max(spot - strike, 0.0)

    d1 = (
        log(spot / strike)
        + (rate + 0.5 * sigma * sigma) * time_to_maturity
    ) / (sigma * sqrt(time_to_maturity))
    d2 = d1 - sigma * sqrt(time_to_maturity)

    return (
        spot * normal_cdf(d1)
        - strike * exp(-rate * time_to_maturity) * normal_cdf(d2)
    )


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


def build_bs_repricing_scenario_table(
    config: BsVarConfig,
) -> tuple[pd.DataFrame, float]:
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
        [
            spx_close.rename("spx_close"),
            vix_close.rename("vix_close"),
            spx_return,
            vix_return,
        ],
        axis=1,
    ).dropna()

    current_time_to_maturity = config.days_to_maturity / 365.0
    next_day_time_to_maturity = max((config.days_to_maturity - 1) / 365.0, 1e-8)

    current_option_price = black_scholes_call_price(
        spot=config.current_spx_level,
        strike=config.strike,
        rate=config.risk_free_rate,
        sigma=config.current_iv,
        time_to_maturity=current_time_to_maturity,
    )

    scenario_table["simulated_spx_level"] = (
        config.current_spx_level * (1.0 + scenario_table["spx_return"])
    )
    scenario_table["simulated_iv"] = (
        config.current_iv * (1.0 + scenario_table["vix_return"])
    ).clip(lower=1e-8)

    scenario_table["simulated_option_price"] = scenario_table.apply(
        lambda row: black_scholes_call_price(
            spot=row["simulated_spx_level"],
            strike=config.strike,
            rate=config.risk_free_rate,
            sigma=row["simulated_iv"],
            time_to_maturity=next_day_time_to_maturity,
        ),
        axis=1,
    )

    scenario_table["option_pnl"] = (
        scenario_table["simulated_option_price"] - current_option_price
    )

    return scenario_table, current_option_price


def compute_var_from_pnl(pnl_series: pd.Series, tail_probability: float) -> float:
    left_tail_pnl = np.quantile(
        pnl_series.dropna().to_numpy(),
        tail_probability,
        method="lower",
    )
    return float(-left_tail_pnl)


def print_summary(
    config: BsVarConfig,
    scenario_count: int,
    current_option_price: float,
    var_value: float,
) -> None:
    confidence_level = 1.0 - config.tail_probability

    print("SPX Option 1-Day Black-Scholes Repricing VaR")
    print("-" * 46)
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
    print(f"Current option px  : {current_option_price:.4f}")
    print(f"1-day BS VaR       : {var_value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute 1-day 99% Black-Scholes repricing VaR for an ATM SPX call option."
    )
    parser.add_argument(
        "--output-file",
        default="spx_option_bs_var.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()

    config = BsVarConfig()

    scenario_table, current_option_price = build_bs_repricing_scenario_table(config)
    var_value = compute_var_from_pnl(
        pnl_series=scenario_table["option_pnl"],
        tail_probability=config.tail_probability,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_table.to_csv(output_path, index=True)

    print_summary(
        config=config,
        scenario_count=len(scenario_table),
        current_option_price=current_option_price,
        var_value=var_value,
    )
    print(f"\nSaved output to: {output_path}")


if __name__ == "__main__":
    main()
