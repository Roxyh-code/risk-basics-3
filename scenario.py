import math
import pandas as pd
import yfinance as yf


CURRENT_SPX = 5995.0
STRIKE = 5995.0
CURRENT_VIX = 15.0
RISK_FREE_RATE = 0.05
DAYS_TO_MATURITY = 28
TIME_TO_MATURITY = DAYS_TO_MATURITY / 365.0

SCENARIOS = [
    ("COVID-19 Crash", "Rapid Market Decline", "2020-02-20", "2020-03-23"),
    ("Global Financial Crisis", "Extended Bear Market", "2008-09-15", "2009-03-09"),
    ("Dot-com Bubble Burst", "Extended Bear Market", "2000-03-10", "2002-10-09"),
    ("August 5, 2024 Shock", "Rapid Market Decline", "2024-08-05", "2024-08-05"),
]


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes_call(spot, strike, rate, vol, maturity):
    if spot <= 0 or strike <= 0 or vol <= 0 or maturity <= 0:
        raise ValueError("Invalid Black-Scholes input.")

    d1 = (math.log(spot / strike) + (rate + 0.5 * vol ** 2) * maturity) / (vol * math.sqrt(maturity))
    d2 = d1 - vol * math.sqrt(maturity)

    return spot * normal_cdf(d1) - strike * math.exp(-rate * maturity) * normal_cdf(d2)


def get_close_value(df, position):
    value = df["Close"].iloc[position]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return float(value)


def get_scenario_data(start_date, end_date):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if start_ts == end_ts:
        download_start = (start_ts - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        download_end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        spx = yf.download("^GSPC", start=download_start, end=download_end, progress=False)
        vix = yf.download("^VIX", start=download_start, end=download_end, progress=False)

        if spx.empty or vix.empty:
            raise ValueError("Missing market data for scenario period.")
        if len(spx) < 2 or len(vix) < 2:
            raise ValueError("Not enough data points for single-day scenario.")

        spx_start = get_close_value(spx, -2)
        spx_end = get_close_value(spx, -1)
        vix_start = get_close_value(vix, -2)
        vix_end = get_close_value(vix, -1)

        actual_start = spx.index[-2].strftime("%Y-%m-%d")
        actual_end = spx.index[-1].strftime("%Y-%m-%d")
        duration = (spx.index[-1] - spx.index[-2]).days

    else:
        download_end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        spx = yf.download("^GSPC", start=start_date, end=download_end, progress=False)
        vix = yf.download("^VIX", start=start_date, end=download_end, progress=False)

        if spx.empty or vix.empty:
            raise ValueError("Missing market data for scenario period.")

        spx_start = get_close_value(spx, 0)
        spx_end = get_close_value(spx, -1)
        vix_start = get_close_value(vix, 0)
        vix_end = get_close_value(vix, -1)

        actual_start = spx.index[0].strftime("%Y-%m-%d")
        actual_end = spx.index[-1].strftime("%Y-%m-%d")
        duration = (spx.index[-1] - spx.index[0]).days

    spx_return = spx_end / spx_start - 1
    vix_return = vix_end / vix_start - 1

    return {
        "actual_start": actual_start,
        "actual_end": actual_end,
        "duration": duration,
        "spx_return": spx_return,
        "vix_return": vix_return,
    }


def run_scenario_analysis():
    initial_vol = CURRENT_VIX / 100
    initial_option_price = black_scholes_call(
        CURRENT_SPX, STRIKE, RISK_FREE_RATE, initial_vol, TIME_TO_MATURITY
    )

    results = []

    for event_name, scenario_type, start_date, end_date in SCENARIOS:
        data = get_scenario_data(start_date, end_date)

        scenario_spx = CURRENT_SPX * (1 + data["spx_return"])
        scenario_vix = CURRENT_VIX * (1 + data["vix_return"])
        scenario_vol = max(scenario_vix / 100, 1e-6)

        scenario_option_price = black_scholes_call(
            scenario_spx, STRIKE, RISK_FREE_RATE, scenario_vol, TIME_TO_MATURITY
        )

        pnl = scenario_option_price - initial_option_price

        results.append({
            "Scenario Type": scenario_type,
            "Event": event_name,
            "Start Date": data["actual_start"],
            "End Date": data["actual_end"],
            "Duration (days)": data["duration"],
            "SPX % Change": data["spx_return"] * 100,
            "VIX % Change": data["vix_return"] * 100,
            "Initial Option Price": initial_option_price,
            "Scenario Option Price": scenario_option_price,
            "Total PnL": pnl
        })

        print("=" * 80)
        print(f"Scenario: {event_name}")
        print(f"Scenario Type: {scenario_type}")
        print(f"Period: {data['actual_start']} to {data['actual_end']}")
        print(f"Duration: {data['duration']} days")
        print(f"SPX Change: {data['spx_return'] * 100:.2f}%")
        print(f"VIX Change: {data['vix_return'] * 100:.2f}%")
        print(f"Initial Option Price: {initial_option_price:.4f}")
        print(f"Scenario Option Price: {scenario_option_price:.4f}")
        print(f"Total PnL: {pnl:.4f}")
        print("=" * 80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["Scenario Type", "Total PnL"]).reset_index(drop=True)

    results_df = results_df[[
        "Scenario Type",
        "Event",
        "Start Date",
        "End Date",
        "Duration (days)",
        "SPX % Change",
        "VIX % Change",
        "Initial Option Price",
        "Scenario Option Price",
        "Total PnL"
    ]]

    results_df = results_df.round({
        "SPX % Change": 2,
        "VIX % Change": 2,
        "Initial Option Price": 4,
        "Scenario Option Price": 4,
        "Total PnL": 4
    })

    results_df.to_csv("scenario_analysis_results.csv", index=False)

    print("\nFinal Scenario Comparison")
    print(results_df.to_string(index=False))
    print("\nCSV file saved as: scenario_analysis_results.csv")

    return results_df


if __name__ == "__main__":
    run_scenario_analysis()
