#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class VarRunConfig:
    symbol: str = "^GSPC"
    start_date: str = "2023-11-09"
    end_date_exclusive: str = "2024-11-09"
    current_level: float = 5995.0
    tail_probability: float = 0.01


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


def compute_historical_var(
    daily_returns: pd.Series,
    current_level: float,
    tail_probability: float,
) -> float:
    left_tail_return = np.quantile(
        daily_returns.to_numpy(),
        tail_probability,
        method="lower",
    )
    return float(-current_level * left_tail_return)


def build_result_table(
    close_series: pd.Series,
    daily_returns: pd.Series,
) -> pd.DataFrame:
    result = pd.DataFrame(index=close_series.index)
    result["close"] = close_series
    result["daily_return"] = daily_returns.reindex(close_series.index)
    return result


def print_summary(config: VarRunConfig, scenario_count: int, var_value: float) -> None:
    confidence_level = 1.0 - config.tail_probability

    print("SPX 1-Day Historical Simulation VaR")
    print("-" * 38)
    print(f"Symbol            : {config.symbol}")
    print(f"Start date        : {config.start_date}")
    print(f"End date          : {config.end_date_exclusive} (exclusive)")
    print(f"Current level     : {config.current_level:.2f}")
    print(f"Confidence level  : {confidence_level:.2%}")
    print(f"Scenario count    : {scenario_count}")
    print(f"1-day VaR         : {var_value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute 1-day historical simulation VaR for SPX."
    )
    parser.add_argument(
        "--output-file",
        default="spx_historical_var.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()

    config = VarRunConfig()

    close_series = load_close_series(
        symbol=config.symbol,
        start_date=config.start_date,
        end_date_exclusive=config.end_date_exclusive,
    )
    daily_returns = compute_daily_returns(close_series)
    var_value = compute_historical_var(
        daily_returns=daily_returns,
        current_level=config.current_level,
        tail_probability=config.tail_probability,
    )

    result_table = build_result_table(
        close_series=close_series,
        daily_returns=daily_returns,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(output_path, index=True)

    print_summary(
        config=config,
        scenario_count=len(daily_returns),
        var_value=var_value,
    )
    print(f"\nSaved output to: {output_path}")


if __name__ == "__main__":
    main()
