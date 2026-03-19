import math
import pandas as pd
import matplotlib.pyplot as plt


CURRENT_SPX = 5995.0
STRIKE = 5995.0
CURRENT_VIX = 15.0
RISK_FREE_RATE = 0.05
DAYS_TO_MATURITY = 28
TIME_TO_MATURITY = DAYS_TO_MATURITY / 365.0

SPX_CHANGES = [-0.30, -0.15, -0.10, -0.05, -0.02, 0.00, 0.02, 0.05, 0.10, 0.15, 0.30]
VIX_CHANGES = [-0.50, -0.20, -0.10, -0.05, 0.00, 0.05, 0.20, 0.50, 1.00, 2.00, 3.00]


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes_call(spot, strike, rate, vol, maturity):
    if spot <= 0 or strike <= 0 or vol <= 0 or maturity <= 0:
        raise ValueError("Invalid Black-Scholes input.")

    d1 = (math.log(spot / strike) + (rate + 0.5 * vol ** 2) * maturity) / (vol * math.sqrt(maturity))
    d2 = d1 - vol * math.sqrt(maturity)

    return spot * normal_cdf(d1) - strike * math.exp(-rate * maturity) * normal_cdf(d2)


def run_sensitivity_analysis():
    initial_vol = CURRENT_VIX / 100
    initial_option_price = black_scholes_call(
        CURRENT_SPX, STRIKE, RISK_FREE_RATE, initial_vol, TIME_TO_MATURITY
    )

    matrix = []

    for spx_change in SPX_CHANGES:
        row = []
        for vix_change in VIX_CHANGES:
            new_spx = CURRENT_SPX * (1 + spx_change)
            new_vix = CURRENT_VIX * (1 + vix_change)
            new_vol = max(new_vix / 100, 1e-6)

            new_option_price = black_scholes_call(
                new_spx, STRIKE, RISK_FREE_RATE, new_vol, TIME_TO_MATURITY
            )

            option_pct_change = (new_option_price - initial_option_price) / initial_option_price * 100
            row.append(option_pct_change)
        matrix.append(row)

    row_labels = [f"{x * 100:.0f}%" for x in SPX_CHANGES]
    col_labels = [f"{x * 100:.0f}%" for x in VIX_CHANGES]

    matrix_df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    matrix_df.to_csv("sensitivity_matrix.csv")

    print(f"Initial Option Price: {initial_option_price:.4f}")
    print("\nSensitivity Matrix (% Change in Option Price)")
    print(matrix_df.round(2).to_string())

    return matrix_df, initial_option_price


def plot_heatmap(matrix_df):
    values = matrix_df.values
    abs_max = max(abs(values.min()), abs(values.max()))

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(values, cmap="RdYlGn", vmin=-abs_max, vmax=abs_max, aspect="auto")

    ax.set_xticks(range(len(matrix_df.columns)))
    ax.set_yticks(range(len(matrix_df.index)))
    ax.set_xticklabels(matrix_df.columns)
    ax.set_yticklabels(matrix_df.index)

    ax.set_xlabel("VIX Change")
    ax.set_ylabel("SPX Change")
    ax.set_title("Option Sensitivity Matrix (% Change in Option Price)")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(matrix_df.index)):
        for j in range(len(matrix_df.columns)):
            value = values[i, j]
            text_color = "white" if abs(value) > abs_max * 0.45 else "black"
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=text_color, fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Option Price Change (%)")

    plt.tight_layout()
    plt.savefig("option_sensitivity_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nCSV file saved as: sensitivity_matrix.csv")
    print("Heatmap saved as: option_sensitivity_heatmap.png")


if __name__ == "__main__":
    matrix_df, initial_option_price = run_sensitivity_analysis()
    plot_heatmap(matrix_df)
