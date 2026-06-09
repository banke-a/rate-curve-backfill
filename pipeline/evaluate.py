"""
Stage 5 — Evaluation
Masking evaluation: compare KNN reconstruction against baselines.
Focus on peak stress period (Sep—Dec 2008).
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

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

STRESS_START = "2008-09-01"
STRESS_END = "2008-12-31"

# Success thresholds (basis points)
MAE_FULL_THRESHOLD = 10
MAE_STRESS_THRESHOLD = 15
MAX_ERROR_THRESHOLD = 50


def compute_metrics(
    df_clean: pd.DataFrame,
    df_filled: pd.DataFrame,
    mask: pd.DataFrame,
    label: str = "Method",
) -> dict:
    """
    Compute error metrics for a single imputation method.

    Args:
        df_clean: Ground truth DataFrame.
        df_filled: Imputed DataFrame.
        mask: Boolean mask — True where values were removed.
        label: Method name for logging.

    Returns:
        Dictionary of error metrics.
    """
    actual = df_clean.values[mask.values]
    predicted = df_filled.values[mask.values]
    errors = np.abs(actual - predicted) * 100  # basis points

    # Stress period metrics
    stress_mask = mask.copy()
    non_stress = (df_clean.index < STRESS_START) | (df_clean.index > STRESS_END)
    stress_mask.loc[non_stress] = False

    actual_stress = df_clean.values[stress_mask.values]
    predicted_stress = df_filled.values[stress_mask.values]
    errors_stress = np.abs(actual_stress - predicted_stress) * 100

    metrics = {
        "method": label,
        "mae_bp": round(float(np.mean(errors)), 2),
        "rmse_bp": round(float(np.sqrt(np.mean(errors ** 2))), 2),
        "max_error_bp": round(float(np.max(errors)) if len(errors) > 0 else 0.0, 2),
        "mae_stress_bp": round(float(np.mean(errors_stress)), 2) if len(errors_stress) > 0 else None,
        "rmse_stress_bp": round(float(np.sqrt(np.mean(errors_stress ** 2))), 2) if len(errors_stress) > 0 else None,
        "n_reconstructed": int(mask.sum().sum()),
        "n_stress_reconstructed": int(stress_mask.sum().sum()),
    }

    # Per-tenor MAE
    for col in TENOR_COLUMNS:
        if col in df_clean.columns and col in mask.columns:
            col_mask = mask[col].values
            if col_mask.sum() > 0:
                col_actual = df_clean[col].values[col_mask]
                col_predicted = df_filled[col].values[col_mask]
                metrics[f"mae_{col}_bp"] = round(
                    float(np.mean(np.abs(col_actual - col_predicted)) * 100), 2
                )

    logger.info(
        f"{label}: MAE={metrics['mae_bp']}bp, RMSE={metrics['rmse_bp']}bp, "
        f"Max={metrics['max_error_bp']}bp | "
        f"Stress MAE={metrics['mae_stress_bp']}bp"
    )

    return metrics


def check_thresholds(metrics: dict) -> dict:
    """
    Check whether results meet the success thresholds.
    """
    results = {
        "mae_full_pass": metrics["mae_bp"] < MAE_FULL_THRESHOLD,
        "mae_stress_pass": metrics["mae_stress_bp"] < MAE_STRESS_THRESHOLD if metrics["mae_stress_bp"] else None,
        "max_error_pass": metrics["max_error_bp"] < MAX_ERROR_THRESHOLD,
    }

    logger.info(
        f"Threshold checks — "
        f"MAE<{MAE_FULL_THRESHOLD}bp: {'✓' if results['mae_full_pass'] else '✗'} | "
        f"Stress MAE<{MAE_STRESS_THRESHOLD}bp: {'✓' if results['mae_stress_pass'] else '✗'} | "
        f"Max<{MAX_ERROR_THRESHOLD}bp: {'✓' if results['max_error_pass'] else '✗'}"
    )
    return results


def plot_error_by_tenor(
    metrics_list: list[dict],
    output_dir: str | Path,
) -> Path:
    """
    Bar chart comparing MAE by tenor across all methods.
    """
    output_dir = Path(output_dir)
    methods = [m["method"] for m in metrics_list]
    colors = ["#AEB6BF", "#F39C12", "#1A5276"]

    tenor_cols = [c for c in TENOR_COLUMNS if f"mae_{c}_bp" in metrics_list[0]]
    tenor_labels = [TENOR_LABELS[c] for c in tenor_cols]

    x = np.arange(len(tenor_cols))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (metrics, color) in enumerate(zip(metrics_list, colors)):
        values = [metrics.get(f"mae_{c}_bp", 0) for c in tenor_cols]
        ax.bar(x + i * width, values, width, label=metrics["method"], color=color, alpha=0.85)

    ax.axhline(y=MAE_FULL_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
               label=f"Success threshold ({MAE_FULL_THRESHOLD}bp)")
    ax.set_xlabel("Tenor", fontsize=11)
    ax.set_ylabel("MAE (basis points)", fontsize=11)
    ax.set_title("Reconstruction Error by Tenor — KNN vs Baselines", fontsize=13, pad=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(tenor_labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "error_by_tenor.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Error by tenor plot saved to {path}")
    return path


def plot_reconstruction_sample(
    df_clean: pd.DataFrame,
    df_knn: pd.DataFrame,
    df_linear: pd.DataFrame,
    mask: pd.DataFrame,
    tenor: str,
    output_dir: str | Path,
) -> Path:
    """
    Time series plot comparing KNN vs linear reconstruction for a single tenor
    during the stress period.
    """
    output_dir = Path(output_dir)

    stress_idx = df_clean.index[
        (df_clean.index >= STRESS_START) & (df_clean.index <= STRESS_END)
    ]

    actual = df_clean.loc[stress_idx, tenor]
    knn = df_knn.loc[stress_idx, tenor]
    linear = df_linear.loc[stress_idx, tenor]
    masked_dates = mask.loc[stress_idx, tenor]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(stress_idx, actual, color="#1A5276", linewidth=2, label="Actual", zorder=3)
    ax.plot(stress_idx, linear, color="#F39C12", linewidth=1.5,
            linestyle="--", label="Linear interpolation", zorder=2)
    ax.plot(stress_idx, knn, color="#E74C3C", linewidth=1.5,
            linestyle="--", label="KNN reconstruction", zorder=2)

    # Highlight masked points
    masked_actual = actual[masked_dates]
    ax.scatter(masked_actual.index, masked_actual.values, color="#1A5276",
               s=40, zorder=4, label="Masked observations")

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Yield (%)", fontsize=11)
    label = TENOR_LABELS.get(tenor, tenor)
    ax.set_title(
        f"{label} Treasury Yield — KNN vs Linear Reconstruction (Sep—Dec 2008)",
        fontsize=13, pad=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    path = output_dir / f"reconstruction_{tenor}_stress.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Reconstruction plot saved to {path}")
    return path
