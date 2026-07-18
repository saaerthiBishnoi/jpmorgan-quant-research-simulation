# JPMorgan Chase & Co. — Quantitative Research Job Simulation

Prototype models built as part of JPMorgan Chase's Quantitative Research virtual job simulation (via Forage). This simulation covers commodity price forecasting, derivative contract pricing, and credit risk modeling — core responsibilities of a quantitative researcher supporting trading desks and risk management teams.

## Overview

| Task | Description | Techniques |
|------|-------------|------------|
| [Task 1](./task1_price_estimation) | Forecast natural gas prices for any date, past or future | Linear regression, seasonal decomposition |
| [Task 2](./task2_contract_pricing) | Price a commodity storage contract given injection/withdrawal dates and costs | Cash flow modeling |
| [Task 3](./task3_default_prediction) | Predict probability of default (PD) on personal loans and compute expected loss | Logistic regression, credit risk (PD × EAD × LGD) |
| [Task 4](./task4_fico_bucketing) | Convert continuous FICO scores into optimal discrete rating buckets | Dynamic programming, log-likelihood optimization (quantization) |

---

## Task 1: Natural Gas Price Estimation

**Business context:** A trading desk wants to price natural gas storage contracts but needs price estimates for dates beyond the available monthly market data.

**Approach:** Modeled the price as a linear trend (long-term drift) plus a seasonal adjustment component (monthly average deviation from trend), capturing the storage-driven pattern where prices rise in winter (withdrawal season) and fall in summer (injection season).

**Deliverable:** `price_estimate(date)` — returns an estimated price for any historical date and extrapolates up to one year into the future.

---

## Task 2: Commodity Storage Contract Pricing

**Business context:** A client wants to buy natural gas now, store it, and sell it later to profit from seasonal price differentials. The desk needs to know the fair value of this contract before quoting a price.

**Approach:** Built a cash flow model that accounts for multiple injection/withdrawal dates, purchase and sale prices (via the Task 1 model), storage capacity constraints, and storage rental costs.

**Deliverable:** `price_contract(...)` — returns the net value of a storage contract given injection dates, withdrawal dates, volumes, storage capacity, and storage costs.

---

## Task 3: Probability of Default & Expected Loss

**Business context:** The retail bank's risk team is seeing higher-than-expected default rates on personal loans and needs a model to estimate default probability and expected loss for capital reserve planning.

**Approach:** Trained a logistic regression classifier on borrower features (credit lines outstanding, loan amount, total debt, income, years employed, FICO score) to predict probability of default. Combined this with the standard credit risk formula to compute expected loss.

**Deliverable:** `expected_loss(...)` — returns the expected monetary loss on a loan, using:

Achieved AUC ≈ 1.0 and accuracy ≈ 99.9% on held-out test data (synthetic dataset with strong signal).

---

## Task 4: FICO Score Bucketing (Quantization)

**Business context:** A downstream machine learning model requires categorical inputs, but FICO scores are continuous (300–850). The team needs a principled way to convert scores into discrete rating buckets, rather than arbitrary equal-width bins.

**Approach:** Implemented optimal quantization using dynamic programming to maximize a log-likelihood objective — choosing bucket boundaries that create the most distinct default rates between buckets, rather than just grouping similar score values together.

**Deliverable:** `fico_to_rating(fico_score)` — maps any FICO score to a rating from 1 (best credit) to N (worst credit).

Validated results show a clear, monotonic default rate separation across buckets — e.g., ~66% default rate in the lowest-scoring bucket down to ~4.6% in the highest-scoring bucket.

---

## Tech Stack

- Python (pandas, numpy, scikit-learn, matplotlib)
- Logistic regression
- Dynamic programming
- Time series trend/seasonality decomposition

## Data

All datasets provided as part of the Forage JPMorgan Quantitative Research simulation (synthetic data for educational purposes).

## Disclaimer

This repository contains work completed as part of Forage's JPMorgan Chase & Co. Quantitative Research virtual job simulation program. All models are prototypes built for educational purposes and are not used in any production trading or risk system.
