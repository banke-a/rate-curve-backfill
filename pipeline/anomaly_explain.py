"""
Stage 6 — LLM Anomaly Explanation
For dates where reconstruction error exceeds the threshold,
use Claude to generate a plain-English market narrative explaining
what drove the anomalous curve behaviour.

This transforms a data quality flag into actionable context —
the kind of explanation a model validator or risk manager would
include in a review of the reconstruction methodology.
"""

import json
import logging
import os

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024

TENOR_LABELS = {
    "DGS1MO": "1 Month", "DGS3MO": "3 Month", "DGS6MO": "6 Month",
    "DGS1": "1 Year", "DGS2": "2 Year", "DGS3": "3 Year",
    "DGS5": "5 Year", "DGS7": "7 Year", "DGS10": "10 Year",
    "DGS20": "20 Year", "DGS30": "30 Year"
}

SYSTEM_PROMPT = """You are a senior fixed income analyst with deep expertise in US Treasury markets 
and the Global Financial Crisis of 2007-2009.

Your task is to provide a concise, factual market narrative explaining why US Treasury yields 
behaved anomalously on a specific date during the GFC. You will be given:
- The date
- Which tenor(s) showed the largest reconstruction errors
- The actual yield levels and daily moves on that date
- The broader curve context

Your explanation should:
1. Identify the specific market event or policy action that drove the anomalous behaviour
2. Explain why this caused the particular curve shape or move observed
3. Note why this makes algorithmic reconstruction difficult (no historical precedent, 
   structural break, or regime change)

Write 2-3 paragraphs. Be precise and factual. Do not speculate beyond what is historically documented.
Return ONLY a JSON object with these keys:
- date: the date as a string
- event_summary: one sentence describing the primary market event
- market_narrative: 2-3 paragraph explanation
- reconstruction_challenge: one sentence explaining why this date is hard to reconstruct algorithmically
"""


def _build_context(
    date: pd.Timestamp,
    df_clean: pd.DataFrame,
    df_filled: pd.DataFrame,
    mask: pd.DataFrame,
) -> dict:
    """
    Build the context payload for a single anomalous date.
    """
    date_str = date.strftime("%Y-%m-%d")

    # Actual rates on this date
    actual_rates = df_clean.loc[date].to_dict()

    # Daily move (vs previous business day)
    prev_dates = df_clean.index[df_clean.index < date]
    if len(prev_dates) > 0:
        prev_date = prev_dates[-1]
        daily_moves = (df_clean.loc[date] - df_clean.loc[prev_date]) * 100  # basis points
        daily_moves_dict = {k: round(v, 1) for k, v in daily_moves.to_dict().items() if not np.isnan(v)}
    else:
        daily_moves_dict = {}

    # Reconstruction errors on this date
    errors = {}
    if date in mask.index:
        masked_tenors = mask.loc[date][mask.loc[date]].index.tolist()
        for tenor in masked_tenors:
            if tenor in df_clean.columns and tenor in df_filled.columns:
                actual = df_clean.loc[date, tenor]
                reconstructed = df_filled.loc[date, tenor]
                errors[TENOR_LABELS.get(tenor, tenor)] = round(
                    abs(actual - reconstructed) * 100, 1
                )

    # Curve shape context
    curve_shape = {
        TENOR_LABELS.get(k, k): round(v, 2)
        for k, v in actual_rates.items()
        if not np.isnan(v)
    }

    return {
        "date": date_str,
        "curve_yields_pct": curve_shape,
        "daily_moves_bp": daily_moves_dict,
        "reconstruction_errors_bp": errors,
    }


def explain_anomaly(
    date: pd.Timestamp,
    df_clean: pd.DataFrame,
    df_filled: pd.DataFrame,
    mask: pd.DataFrame,
) -> dict:
    """
    Generate a plain-English market narrative for a single anomalous date.

    Args:
        date: The anomalous date to explain.
        df_clean: Original clean DataFrame.
        df_filled: KNN-imputed DataFrame.
        mask: Boolean mask of simulated gaps.

    Returns:
        Dictionary with date, event summary, narrative, and reconstruction challenge.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    context = _build_context(date, df_clean, df_filled, mask)
    date_str = date.strftime("%d %B %Y")

    user_message = json.dumps({
        "instruction": f"Explain the anomalous US Treasury curve behaviour on {date_str}.",
        "data": context
    }, indent=2)

    logger.info(f"Generating anomaly explanation for {date_str}")

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    raw_text = response.content[0].text.strip()

    # Parse JSON response
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(lines[1:-1])

    result = json.loads(raw_text)
    result["context"] = context
    result["tokens_used"] = response.usage.output_tokens

    logger.info(f"Explanation generated for {date_str} — {response.usage.output_tokens} tokens")
    return result


def explain_high_error_dates(
    df_clean: pd.DataFrame,
    df_filled: pd.DataFrame,
    mask: pd.DataFrame,
    error_threshold_bp: float = 50.0,
    max_dates: int = 10,
) -> list[dict]:
    """
    Identify dates with highest reconstruction errors and generate explanations.

    Args:
        df_clean: Original clean DataFrame.
        df_filled: KNN-imputed DataFrame.
        mask: Boolean mask of simulated gaps.
        error_threshold_bp: Only explain dates exceeding this error (basis points).
        max_dates: Maximum number of dates to explain (cost control).

    Returns:
        List of explanation dictionaries, one per anomalous date.
    """
    # Compute per-date maximum reconstruction error
    errors = (df_clean - df_filled).abs() * 100  # basis points
    errors_masked = errors.where(mask)
    max_error_per_date = errors_masked.max(axis=1).dropna()

    # Filter to high-error dates
    high_error_dates = max_error_per_date[
        max_error_per_date > error_threshold_bp
    ].sort_values(ascending=False)

    if len(high_error_dates) == 0:
        logger.info(f"No dates exceed {error_threshold_bp}bp threshold")
        return []

    logger.info(
        f"Found {len(high_error_dates)} dates exceeding {error_threshold_bp}bp threshold"
    )
    for date, error in high_error_dates.items():
        logger.info(f"  {date.strftime('%Y-%m-%d')}: {error:.1f}bp max error")

    # Explain up to max_dates
    dates_to_explain = high_error_dates.head(max_dates).index
    explanations = []

    for date in dates_to_explain:
        try:
            explanation = explain_anomaly(date, df_clean, df_filled, mask)
            explanations.append(explanation)
        except Exception as e:
            logger.error(f"Failed to generate explanation for {date}: {e}")
            explanations.append({
                "date": date.strftime("%Y-%m-%d"),
                "error": str(e)
            })

    return explanations
