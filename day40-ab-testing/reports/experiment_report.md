# Experiment Report: Homepage CTA Redesign

**Date:** June 2026  
**Duration:** 14 days  
**Analyst:** Rajesh Yadav

---

## Hypothesis

The redesigned Call-to-Action button on the homepage will increase the conversion rate
compared to the current design.

- **H0:** Conversion rate of experiment = Conversion rate of control
- **H1:** Conversion rate of experiment > Conversion rate of control
- **Significance level:** alpha = 0.05

---

## Experiment Setup

| Attribute | Value |
|---|---|
| Control group | 4,800 users (old CTA) |
| Experiment group | 5,200 users (new CTA) |
| Duration | 14 days |
| SRM check | Passed |

---

## Results

### Primary Metric: Conversion Rate

| Group | Users | Conversions | Rate |
|---|---|---|---|
| Control | 4,800 | ~542 | 11.27% |
| Experiment | 5,200 | ~745 | 14.33% |

- Relative lift: **+27.1%**
- Z-statistic: **4.56**
- P-value: **< 0.00001**
- 95% CI: **(+1.7%, +4.4%)**
- Result: **Statistically significant**

### Secondary Metrics

| Metric | Control | Experiment | Change |
|---|---|---|---|
| Revenue / User | $6.00 | $9.26 | +54.3% |
| Session Time | 8.90 min | 10.58 min | +18.9% |
| Pages Viewed | 4.09 | 4.62 | +12.9% |

All secondary metrics improved significantly.

---

## Segment Consistency

The conversion lift was positive across all device types (desktop, mobile, tablet) and
all countries (US, UK, CA, AU, IN). No harmful subgroup effects were detected.

---

## Decision

**Ship the new CTA design to 100% of users.**

Estimated annual revenue impact: **~$500K – $1.3M** depending on traffic scale.


