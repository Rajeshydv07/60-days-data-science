# Day 40: Evaluating Business Decisions with A/B Testing

## Project Overview

This project is part of my **60 Days of Data Science** portfolio. Day 40 covers A/B Testing — the scientific framework used by tech companies to evaluate product changes before rolling them out to all users.

The scenario simulates an e-commerce company that redesigned its homepage Call-to-Action (CTA) button. The question: does the new design improve conversion, revenue, and engagement compared to the old design? We answer that with a full, end-to-end A/B test.

---

## Dataset Description

The dataset `ab_test_data.csv` is a simulated experiment dataset with **10,000 users** split across two groups:

| Column | Description |
|---|---|
| `user_id` | Unique identifier for each user |
| `group` | Either `control` (old CTA) or `experiment` (new CTA) |
| `converted` | 1 if the user purchased, 0 if not |
| `revenue` | Revenue generated (0 for non-converters) |
| `session_time_min` | Duration of the browsing session in minutes |
| `pages_viewed` | Number of pages the user visited |
| `device` | Device type: desktop, mobile, or tablet |
| `country` | Country of origin: US, UK, CA, AU, IN |

- **Control group**: 4,800 users, baseline conversion rate ~11.3%
- **Experiment group**: 5,200 users, improved conversion rate ~14.3%

---

## Analysis Steps

### 1. Sample Ratio Mismatch (SRM) Check
Before any statistical testing, we verify that users were correctly split between groups. A failed SRM check means the experiment was biased and results cannot be trusted. Our chi-square test confirmed no SRM.

### 2. Exploratory Data Analysis
We compared group distributions across all four metrics: conversion rate, revenue distribution (for converters only), session time, and pages viewed. The experiment group showed better performance across all metrics.

### 3. Primary Metric: Conversion Rate (Two-Proportion Z-Test)

**Hypotheses:**
- H0: Conversion rate of experiment = Conversion rate of control
- H1: Conversion rate of experiment != Conversion rate of control
- Significance level: alpha = 0.05

**Results:**

| Metric | Control | Experiment |
|---|---|---|
| Users | 4,800 | 5,200 |
| Conversions | ~542 | ~745 |
| Conversion Rate | 11.27% | 14.33% |
| Relative Lift | — | +27.1% |
| Z-Statistic | — | 6.34 |
| P-Value | — | < 0.00001 |
| Significant | — | YES |

The p-value is far below the 0.05 threshold, so we reject the null hypothesis with very high confidence. The new design genuinely improves conversion.

### 4. Secondary Metrics (Two-Sample T-Test)

| Metric | Control | Experiment | Change | Significant |
|---|---|---|---|---|
| Revenue per user | $6.00 | $9.26 | +54.3% | YES |
| Session time | 8.90 min | 10.58 min | +18.9% | YES |
| Pages viewed | 4.09 | 4.62 | +12.9% | YES |

All secondary metrics improved significantly, confirming that the new design drives both more and higher-quality engagement.

### 5. Power Analysis
The experiment was checked for adequate statistical power (at least 80%). With the observed effect size and sample sizes used, the test exceeded 99% power, meaning the experiment would reliably detect this level of improvement.

### 6. Segment Analysis
We broke down conversion rates by device type (desktop, mobile, tablet) and country (US, UK, CA, AU, IN). The experiment group showed consistent lift across all segments, confirming no harmful subgroup effects.

### 7. Cumulative Timeline (Anti-Peeking Check)
We plotted cumulative conversion rates and p-values day-by-day over the 14-day experiment. The p-value converged below 0.05 only after sufficient data was collected, showing the result is not an artifact of early stopping.

---

## Visualizations

| File | Description |
|---|---|
| `eda_distributions.png` | Conversion rate bar chart, revenue histogram, session time histogram, pages viewed box plot |
| `segment_analysis.png` | Conversion rate by device and by country for both groups |
| `cumulative_timeline.png` | Cumulative conversion rate and p-value over the 14-day experiment |
| `results_dashboard.png` | Full summary: conversion rate, revenue, session time bar charts and relative lift chart |

---

## Business Recommendation

**Ship the new CTA design to 100% of users.**

Reasoning:

1. The conversion rate improvement of +27.1% is statistically significant with a p-value far below 0.05.
2. Revenue per user also increased by +54.3%, so the new design is not just generating low-value conversions.
3. Engagement metrics (session time and pages viewed) improved, showing the design improves the overall experience.
4. The effect is consistent across all device types and countries, meaning there are no subgroups being harmed.
5. The experiment was sufficiently powered and ran for its full planned duration.

**Estimated Annual Revenue Impact:**
- Monthly additional conversions: ~1,645
- Average order value (experiment converters): ~$64.61
- Estimated monthly revenue gain: ~$106,300
- Estimated annual revenue gain: ~$1,275,600

---

## Technologies Used

- **Pandas**: Data loading, groupby aggregation, segment pivot tables
- **NumPy**: Array operations, proportion and power calculations
- **SciPy (stats)**: Two-proportion z-test, two-sample t-test, chi-square test
- **Matplotlib**: All chart layouts, bar plots, histograms, box plots, timeline plots
- **Seaborn**: Color palette integration and visual styling




