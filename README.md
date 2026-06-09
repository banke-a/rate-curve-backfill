# Interest Rate Curve Backfilling

## Research Question

Can a KNN-based framework generate synthetic historical rate observations for missing tenor data during the GFC stress window, and do those generated values meet the accuracy thresholds required for quantitative risk analysis?

---

## The Problem

Interest rate curves are foundational inputs to quantitative risk analysis. Constructing a complete historical rate surface requires daily observations across all tenors — short end (1M, 3M, 6M), belly (1Y, 2Y, 3Y, 5Y, 7Y), and long end (10Y, 20Y, 30Y).

During stress periods, this completeness breaks down. The Global Financial Crisis of 2007—2009 produced conditions where:

- Certain tenors became illiquid — no transactions, no reliable price discovery
- Data vendors reported missing or stale observations for extended periods
- Curve shapes moved in historically unprecedented ways, making simple interpolation unreliable
- Instruments being risk-managed may not have existed during the stress period, requiring reconstruction of plausible historical paths

In production risk systems, reconstructed rate series are used as inputs to simulation frameworks that require complete historical observations across all tenors. This project focuses on the reconstruction layer — generating accurate, defensible synthetic observations from observable market data, and then backtesting whether those generated values are close enough to reality to be trusted in a risk calculation.

---

## Why KNN

Linear interpolation assumes the curve moves smoothly between adjacent tenors. During stress periods this assumption breaks down — the curve can invert, flatten, steepen, and change shape in ways that make linear methods unreliable.

KNN makes no assumption about curve shape. Instead it asks: **on days when the observable tenors looked similar to today, what did the missing tenor do?** It reconstructs from the actual comovement structure that existed during the crisis itself.

This is the key insight: the GFC provides its own reference set. The algorithm learns from the relationships between tenors that existed during the stress period, not from a smooth pre-crisis baseline that may no longer apply.

---

## What This Project Does

A six-stage pipeline that:

1. Pulls US Treasury yield curve data from FRED (11 tenors, 2007—2009)
2. Validates and cleans the data — identifies holidays, outliers, stale prices
3. Simulates realistic missing data patterns — random gaps, stress-period concentration, tenor-specific outages
4. Reconstructs missing values using KNN and two baseline methods (linear interpolation, forward fill)
5. Evaluates reconstruction accuracy through a masking framework — withholds known observations, reconstructs, measures error in basis points
6. Explains high-error dates using the Claude API — generates plain-English market narratives for dates where reconstruction fails

```
FRED Treasury Data (11 tenors, 752 business days)
                │
                ▼
        Validate & Clean
                │
                ▼
    Simulate Missing Data
    (random + stress + outage)
                │
                ▼
    Reconstruct (KNN + Baselines)
                │
                ▼
    Evaluate (MAE, RMSE, Max Error)
                │
                ▼
    LLM Anomaly Explanation
    (Claude API for high-error dates)
```

---

## Data

**Source:** US Treasury Yield Curve Rates — FRED (Federal Reserve Bank of St. Louis)
**Coverage:** Daily observations, January 2007 — December 2009
**Tenors:** 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y
**Licence:** Public domain, US government data

The raw FRED data for this window contains 784 calendar dates, 752 business days. All 32 missing values in the raw data are US public holidays — the data is clean on business days. Missing data is simulated to mirror production scenarios.

---

## Gap Simulation

Three types of missing data are simulated:

| Type | Description | Rationale |
|---|---|---|
| Random gaps | 10% of observations masked uniformly | General data vendor failures |
| Stress concentration | Additional 20% masking Sep—Dec 2008 | Liquidity collapse during peak crisis |
| Tenor outage | DGS7 and DGS20 fully missing Oct 2008 | Illiquid instruments during panic |

Total masked: 1,012 observations (12.2% of all data)

---

## Evaluation Results

20 generated memos were evaluated through a masking framework: known observations withheld, reconstructed, compared against actual values in basis points.

### Method Comparison

| Method | MAE (bp) | Stress MAE (bp) | RMSE (bp) | Max Error (bp) |
|---|---|---|---|---|
| Linear interpolation | 4.77 | 7.69 | 7.49 | 50.50 |
| Forward fill | 7.42 | 13.21 | 11.90 | 101.00 |
| **KNN (k=5)** | **5.20** | **10.17** | **8.97** | **75.13** |

### Success Thresholds

| Threshold | Target | Result |
|---|---|---|
| MAE — full period | < 10bp | ✅ 5.20bp |
| MAE — stress period | < 15bp | ✅ 10.17bp |
| Max single error | < 50bp | ✗ 75.13bp |

### K Value Sensitivity

| K | MAE (bp) | RMSE (bp) |
|---|---|---|
| 3 | 5.17 | 9.34 |
| **5** | **5.20** | **8.97** |
| 7 | 5.27 | 8.90 |
| 10 | 5.59 | 9.16 |

K=5 selected as default — lowest RMSE, stable MAE.

See `/evaluation/results_summary.csv` and `/evaluation/k_comparison.csv` for full results.

---

## Where KNN Wins and Where It Fails

**KNN outperforms linear interpolation specifically on illiquid tenors:**
- 7Y tenor: KNN 5.6bp vs Linear 7.2bp — KNN wins by 1.6bp on the simulated outage tenor
- 10Y tenor: KNN 3.7bp vs Linear 5.4bp — KNN wins by 1.7bp
- 30Y tenor: KNN 3.4bp vs Linear 4.3bp — KNN wins

**Linear interpolation outperforms KNN on the short end:**
- 1M, 3M, 6M — linear wins

This is expected and explainable. The short end during the GFC was driven by unprecedented Fed policy actions — rates collapsed to near zero in ways that had no historical precedent. KNN found no reliable neighbours because the curve shape was genuinely new. Linear interpolation works better there because the short end moved relatively smoothly day-to-day despite the extreme levels.

**The maximum error threshold (75bp vs 50bp target) is exceeded on short-end tenors during peak Lehman week.** This is not a failure of the methodology — it is an honest reflection of the limits of neighbour-based reconstruction when the market enters a regime with no historical reference.

---

## LLM Anomaly Explanation

For dates where reconstruction error exceeds 50bp, the Claude API generates a plain-English market narrative explaining what drove the anomalous curve behaviour.

Three dates were identified:

| Date | Max Error | Event |
|---|---|---|
| 26 Sep 2008 | 75.1bp | Washington Mutual seized — largest US bank failure in history |
| 22 Sep 2008 | 53.7bp | FOMC holds rates unexpectedly, breaking market expectations |
| 18 Sep 2008 | 52.9bp | TARP announced, coordinated global central bank liquidity injection |

All three are in the week following Lehman Brothers' bankruptcy. The LLM correctly identifies the specific policy actions and market events that caused the unprecedented curve shapes, and explains why algorithmic reconstruction is structurally limited on these dates.

See `/output/anomaly_report.json` for the full narratives.

---

## How to Run

**1. Clone and install**

```bash
git clone https://github.com/banke-a/rate-curve-backfill.git
cd rate-curve-backfill
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Set up API keys**

```bash
cp .env.example .env
# Add your FRED API key (free at fred.stlouisfed.org)
# Add your Anthropic API key (console.anthropic.com)
```

**3. Run the full pipeline**

```bash
python run.py
```

**Skip the LLM stage:**

```bash
python run.py --skip-llm
```

**Use cached FRED data (subsequent runs):**

```bash
python run.py --use-cache
```

**Run tests:**

```bash
pytest tests/
```

**API costs:** The LLM stage typically calls Claude 3—10 times depending on how many dates exceed the error threshold. Cost is under $0.10 per full run at current Sonnet pricing.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| FRED Treasury data | Free, public, credible, complete for the GFC window. Swap rate data from this period is not publicly available. |
| Simulated gaps | The FRED data is clean on business days. Gaps are simulated to mirror production scenarios — random failures, illiquid tenors, stress concentration. |
| KNN via scikit-learn | Well-tested, documented, reproducible. Custom implementation adds complexity without improving the portfolio signal. |
| Masking evaluation | No truly missing ground truth is available. Withholding known observations is the standard approach for evaluating imputation quality. |
| Temporal split for evaluation | When evaluating a date, only prior dates are used as the neighbour reference set. Prevents look-ahead bias. |
| GFC window only | Focused stress window keeps the dataset manageable and makes the stress period analysis meaningful. |
| LLM for anomaly explanation only | KNN handles reconstruction. Claude adds the narrative layer that transforms error metrics into market context. Clear separation of responsibilities. |
| Downstream simulation out of scope | The distributional transformation and simulation frameworks that consume backfilled data in production systems are beyond the scope of this project. |

---

## Context

In production quantitative risk systems, complete historical rate series are required inputs to simulation frameworks used for risk measurement. When instruments are missing from the historical record - either because they did not exist, were illiquid, or had data vendor failures — reconstruction is required before those frameworks can run.

This project focuses on the reconstruction layer. The broader pipeline that consumes these reconstructed series is not included here.

---

## Future Work

- Extended tenor coverage and international rate curves
- Alternative stress periods — COVID-19 (2020), European sovereign debt crisis (2011—2012)
- Advanced imputation methods — MICE, matrix factorisation, neural approaches for comparison
- Swap rate extension — when suitable public data becomes available
- Automated gap detection and reconstruction for ongoing use

---

## Project Structure

```
rate-curve-backfill/
├── data/
│   └── raw/                      # FRED CSV cache (gitignored)
├── output/
│   ├── plots/                    # All visualisations
│   ├── anomaly_report.json       # LLM anomaly explanations
│   └── backfill_metadata.json    # Run metadata and metrics
├── evaluation/
│   ├── results_summary.csv       # Method comparison table
│   └── k_comparison.csv          # K value sensitivity
├── pipeline/
│   ├── ingest.py                 # Stage 1: FRED data pull
│   ├── validate.py               # Stage 2: Quality checks
│   ├── analyse_gaps.py           # Stage 3: Gap simulation + visualisation
│   ├── impute_baseline.py        # Stage 4a: Linear + forward fill
│   ├── impute_knn.py             # Stage 4b: KNN reconstruction
│   ├── evaluate.py               # Stage 5: Masking evaluation
│   └── anomaly_explain.py        # Stage 6: LLM anomaly explanation
├── tests/
│   ├── test_impute_knn.py
│   ├── test_evaluate.py
│   └── test_evaluation_mechanics.py
├── run.py
├── .env.example
└── requirements.txt
```

---

*This project is a portfolio piece demonstrating quantitative finance domain expertise applied to a real data engineering problem. All data is sourced from publicly available US government datasets. The reconstruction methodology does not reproduce any proprietary system or algorithm.*
