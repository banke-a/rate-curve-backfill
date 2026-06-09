"""
run.py — Pipeline entry point

Usage:
    Full pipeline:
        python run.py

    Ingest only:
        python run.py --ingest-only

    Through gap analysis only:
        python run.py --analyse-only

    Use cached FRED data:
        python run.py --use-cache
"""

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from pipeline import ingest, validate, analyse_gaps, impute_baseline, impute_knn, evaluate

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")

DATA_DIR = os.getenv("DATA_DIR", "data/raw")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
CACHE_PATH = f"{DATA_DIR}/treasury_gfc_raw.csv"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"


def main():
    parser = argparse.ArgumentParser(description="Rate Curve Backfilling Pipeline")
    parser.add_argument("--ingest-only", action="store_true")
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--k", type=int, default=5, help="KNN k value (default 5)")
    args = parser.parse_args()

    for d in [DATA_DIR, OUTPUT_DIR, PLOTS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Stage 1 — Ingest
    logger.info("Stage 1 — Ingest")
    cache = CACHE_PATH if args.use_cache or Path(CACHE_PATH).exists() else None
    df_raw = ingest.load(cache_path=cache)
    if not Path(CACHE_PATH).exists():
        df_raw.to_csv(CACHE_PATH)

    if args.ingest_only:
        logger.info(f"Shape: {df_raw.shape} | Date range: {df_raw.index.min().date()} to {df_raw.index.max().date()}")
        return

    # Stage 2 — Validate
    logger.info("Stage 2 — Validate")
    result = validate.validate(df_raw)
    df_clean = result["df"]
    logger.info(f"Clean business days: {len(df_clean)}")

    # Stage 3 — Gap simulation
    logger.info("Stage 3 — Gap simulation")
    df_gaps, mask = analyse_gaps.simulate_gaps(df_clean)
    analyse_gaps.plot_missing_heatmap(df_gaps, PLOTS_DIR)
    analyse_gaps.plot_curve_shapes(df_clean, PLOTS_DIR)

    if args.analyse_only:
        logger.info(f"Plots saved to {PLOTS_DIR}")
        return

    # Stage 4a — Baseline imputation
    logger.info("Stage 4a — Baseline imputation")
    df_linear = impute_baseline.linear_interpolation(df_gaps)
    df_ffill = impute_baseline.forward_fill(df_gaps)

    # Stage 4b — KNN imputation
    logger.info(f"Stage 4b — KNN imputation (k={args.k})")
    df_knn = impute_knn.impute(df_gaps, k=args.k)

    # Evaluate K values
    logger.info("Evaluating K values")
    k_results = impute_knn.evaluate_k_values(df_gaps, df_clean, mask)
    logger.info(f"\nK value comparison:\n{k_results.to_string(index=False)}")

    # Stage 5 — Evaluate
    logger.info("Stage 5 — Evaluation")
    metrics_linear = evaluate.compute_metrics(df_clean, df_linear, mask, label="Linear")
    metrics_ffill = evaluate.compute_metrics(df_clean, df_ffill, mask, label="Forward Fill")
    metrics_knn = evaluate.compute_metrics(df_clean, df_knn, mask, label=f"KNN (k={args.k})")

    # Threshold checks
    logger.info("Checking success thresholds")
    thresholds = evaluate.check_thresholds(metrics_knn)

    # Visualisations
    evaluate.plot_error_by_tenor(
        [metrics_linear, metrics_ffill, metrics_knn], PLOTS_DIR
    )
    evaluate.plot_reconstruction_sample(
        df_clean, df_knn, df_linear, mask, "DGS10", PLOTS_DIR
    )
    evaluate.plot_reconstruction_sample(
        df_clean, df_knn, df_linear, mask, "DGS7", PLOTS_DIR
    )

    # Save results
    results = {
        "k_evaluation": k_results.to_dict(orient="records"),
        "metrics": {
            "linear": metrics_linear,
            "forward_fill": metrics_ffill,
            f"knn_k{args.k}": metrics_knn,
        },
        "threshold_checks": thresholds,
    }
    results_path = Path(OUTPUT_DIR) / "backfill_metadata.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    for method, m in [("Linear", metrics_linear), ("Forward Fill", metrics_ffill), (f"KNN k={args.k}", metrics_knn)]:
        logger.info(f"{method:20s} MAE={m['mae_bp']:5.2f}bp | Stress MAE={m['mae_stress_bp']:5.2f}bp | Max={m['max_error_bp']:6.2f}bp")
    logger.info("="*60)
    logger.info(f"KNN threshold checks: {thresholds}")


if __name__ == "__main__":
    main()
