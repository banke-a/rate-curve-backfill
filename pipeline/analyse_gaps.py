"""
Stage 3 — Analyse Gaps and Simulate Missing Data
Characterises the data and introduces realistic missing value patterns
that mirror production scenarios: random gaps, tenor-specific outages,
and stress-period concentration.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

TENOR_COLUMNS = [
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2",
    "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"
]

TENOR_LABELS = {
    "DGS1MO": "1M", "DGS3MO": "3M", "DGS6MO": "6M",
    "DGS1": "1Y", "DGS2": "2Y", "DGS3": "3Y",
    "DGS5": "5Y", "DGS7": "7Y", "DGS10": "10Y",
    "DGS20": "20Y", "DGS30": "30Y"
}

# Peak stress window
STRESS_START = "2008-09-01"
STRESS_END = "2008-12-31"

# Gap simulation parameters
RANDOM_GAP_PCT = 0.10        # 10% random masking across all tenors
STRESS_GAP_PCT = 0.20        # 20% additional masking in stress period
TENOR_OUTAGE_TENORS = ["DGS7", "DGS20"]  # Tenors simulating illiquidity
TENOR_OUTAGE_WINDOW = ("2008-10-01", "2008-10-31")  # October 2008 outage


def simulate_gaps(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Introduce realistic missing value patterns into the clean DataFrame.

    Three gap types:
    1. Random gaps — 10% of observations across all tenors
    2. Stress period concentration — additional 20% masking Sep-Dec 2008
    3. Tenor-specific outage — DGS7 and DGS20 missing for October 2008

    Args:
        df: Clean business-day DataFrame from validate stage.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (df_with_gaps, mask) where mask is True where values were removed.
    """
    rng = np.random.default_rng(seed)
    df_gaps = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    # 1. Random gaps across all tenors
    for col in TENOR_COLUMNS:
        if col not in df.columns:
            continue
        n_mask = int(len(df) * RANDOM_GAP_PCT)
        indices = rng.choice(len(df), size=n_mask, replace=False)
        mask.iloc[indices, mask.columns.get_loc(col)] = True
        df_gaps.iloc[indices, df_gaps.columns.get_loc(col)] = np.nan
        
    # 2. Stress period concentration
    stress_idx = df.index[(df.index >= STRESS_START) & (df.index <= STRESS_END)]
    for col in TENOR_COLUMNS:
        if col not in df.columns:
            continue
        n_stress = int(len(stress_idx) * STRESS_GAP_PCT)
        indices = rng.choice(len(stress_idx), size=n_stress, replace=False)
        stress_dates = stress_idx[indices]
        mask.loc[stress_dates, col] = True
        df_gaps.loc[stress_dates, col] = np.nan

    # 3. Tenor-specific outage — illiquid tenors during October 2008
    outage_start, outage_end = TENOR_OUTAGE_WINDOW
    outage_idx = df.index[(df.index >= outage_start) & (df.index <= outage_end)]
    for col in TENOR_OUTAGE_TENORS:
        if col in df.columns:
            mask.loc[outage_idx, col] = True
            df_gaps.loc[outage_idx, col] = np.nan

    total_masked = mask.sum().sum()
    total_obs = df.size
    logger.info(
        f"Gap simulation complete — {total_masked:,} observations masked "
        f"({total_masked/total_obs:.1%} of total)"
    )
    logger.info(f"Missing values per tenor:\n{df_gaps.isna().sum().to_string()}")

    return df_gaps, mask


def plot_missing_heatmap(df_gaps: pd.DataFrame, output_dir: str | Path) -> Path:
    """
    Generate a heatmap showing the location of missing values across dates and tenors.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = df_gaps.isna().astype(int)

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        missing.T,
        cmap=["#EBF5FB", "#1A5276"],
        cbar=False,
        ax=ax,
        yticklabels=[TENOR_LABELS.get(c, c) for c in missing.columns],
    )

    # Mark stress period
    stress_start_pos = missing.index.searchsorted(pd.Timestamp(STRESS_START))
    stress_end_pos = missing.index.searchsorted(pd.Timestamp(STRESS_END))
    ax.axvline(x=stress_start_pos, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axvline(x=stress_end_pos, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.text(
        (stress_start_pos + stress_end_pos) / 2, -0.5,
        "Peak Stress", ha="center", va="bottom", color="red", fontsize=9
    )

    ax.set_title("Simulated Missing Data — US Treasury Yield Curve (2007—2009)", fontsize=13, pad=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Tenor", fontsize=10)

    # Thin out x-axis labels
    n_ticks = 12
    tick_positions = np.linspace(0, len(missing) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [missing.index[i].strftime("%b %Y") for i in tick_positions],
        rotation=45, ha="right", fontsize=8
    )

    plt.tight_layout()
    path = output_dir / "missing_data_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Missing data heatmap saved to {path}")
    return path


def plot_curve_shapes(df_clean: pd.DataFrame, output_dir: str | Path) -> Path:
    """
    Plot yield curve shapes on selected significant dates during the GFC.
    """
    output_dir = Path(output_dir)

    significant_dates = {
        "2007-06-01": ("Pre-crisis baseline", "#2ECC71"),
        "2008-01-02": ("Early 2008 — Fed cutting", "#F39C12"),
        "2008-09-15": ("Lehman bankruptcy", "#E74C3C"),
        "2008-11-20": ("Crisis peak", "#8E44AD"),
        "2009-03-18": ("Fed QE1 announcement", "#2980B9"),
        "2009-12-31": ("End of window", "#1A5276"),
    }

    tenors_numeric = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
    tenor_cols = TENOR_COLUMNS

    fig, ax = plt.subplots(figsize=(12, 6))

    for date_str, (label, color) in significant_dates.items():
        date = pd.Timestamp(date_str)
        # Find nearest available date
        available = df_clean.index[df_clean.index >= date]
        if len(available) == 0:
            continue
        actual_date = available[0]
        rates = df_clean.loc[actual_date, tenor_cols].values

        ax.plot(tenors_numeric, rates, marker="o", label=f"{actual_date.strftime('%d %b %Y')} — {label}",
                color=color, linewidth=2, markersize=5)

    ax.set_xlabel("Tenor (years)", fontsize=11)
    ax.set_ylabel("Yield (%)", fontsize=11)
    ax.set_title("US Treasury Yield Curve — Selected GFC Dates", fontsize=13, pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xscale("log")
    ax.set_xticks(tenors_numeric)
    ax.set_xticklabels(
        [TENOR_LABELS[c] for c in tenor_cols],
        fontsize=9
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "curve_shapes_gfc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Curve shape plot saved to {path}")
    return path
