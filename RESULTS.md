# Breast-cancer cfRNA classification — results

SILVER-Seq serum cell-free RNA ([PNAS 2019](https://www.pnas.org/content/116/38/19200)).
Two tasks: **(A)** cancer vs. normal, **(B)** recurrence vs. non-recurrence.
_Last run: 2026-05-23. Reproduce with the commands at the bottom._

## TL;DR

| Task | Train AUC | Cross-val AUC | External-cohort AUC | Verdict |
|---|---|---|---|---|
| **A — cancer vs normal** | 1.000 | **0.988** (pooled) | 0.751 | Strong, easy signal |
| **B — recurrence** | 1.000 | **0.659** (donor-level) | 0.450 | ≈ chance, as expected |

Same model and regularization for both. Both fit the training set perfectly
(train AUC = 1.000); they diverge only out of sample — Task A holds up, Task B
collapses. Task B's poor result matches the project's stated expectation.

---

## Data & preprocessing

- **PNAS**: 96 breast-cancer serum samples (from **44 donors**) + 32 normal serum samples.
- **External validation** (held out entirely): 161 samples = 83 BC + 78 normal.
- Gene-level Ensembl TPM, 60,675 genes → **42,536 kept** (TPM>0 in ≥20% of PNAS
  samples) → `log2(TPM+1)`.
- Standardization and any supervised feature selection are done **inside each CV
  fold** (train-only), never on the full data → no leakage.

Details and data quirks: [processed/SUMMARY.md](processed/SUMMARY.md),
pipeline: [src/preprocess.py](src/preprocess.py).

## Cross-validation design

**3-fold, grouped by donor (`poiseid`), stratified at the donor level.**
Rationale: recurrence is a **donor-level** property (constant within donor; only
**44 donors / 10 recurrence-positive** behind the 96 samples). The 28 "recurrence
samples" are longitudinal repeats of 10 people, so a random split would put the
same person in train and test and fake good performance. Donor grouping keeps
every donor wholly in one fold; 3 folds (not 5) keep ≥3 positive donors per test
fold. Verified: 0 donors span >1 fold; positive donors split 3/4/3.

## Model

`StandardScaler → L2 logistic regression` (`C=1.0`, `class_weight='balanced'`).
Deliberately simple — a strong, interpretable baseline.

---

## Task A — cancer vs normal

[src/model_cancer_vs_normal.py](src/model_cancer_vs_normal.py)

**Cross-validation (within PNAS, donor-grouped):**

| Metric | Value |
|---|---|
| ROC-AUC per fold | 0.963 / 0.993 / 1.000 (mean 0.985 ± 0.016) |
| Pooled ROC-AUC | **0.988** |
| Pooled PR-AUC | 0.996 |
| Accuracy | 0.906 (majority baseline 0.750) |
| Balanced accuracy | 0.823 |

Confusion (rows = true normal/cancer, cols = pred): `[[21, 11], [1, 95]]`
→ ranking is near-perfect; the 0.5 threshold over-calls cancer for normals (11
false positives), a calibration issue, not a discrimination one.

**External validation** (train on all 128 PNAS, test on 161): ROC-AUC **0.751**,
PR-AUC 0.812, balanced acc 0.608.

**Read:** a simple linear model separates cancer from normal almost perfectly
within PNAS — this is the working classifier for goal #1. The 0.99→0.75 drop
across cohorts is the honest estimate and signals that **part of the within-PNAS
separability is batch/technical** (cancer and normal are different sample sets).
Treat the CV number as an upper bound, the validation number as realistic.

## Task B — recurrence vs non-recurrence

[src/model_recurrence.py](src/model_recurrence.py)

| Level | ROC-AUC | PR-AUC | Notes |
|---|---|---|---|
| Train | 1.000 | — | perfect fit, as in Task A (meaningless under p≫n) |
| CV, sample-level | 0.629 pooled (per-fold 0.67/0.71/0.71) | 0.396 | optimistic; treats follow-ups as independent |
| **CV, donor-level** | **0.659** | 0.332 | the honest metric (n=44, 10 positive) |
| External validation BC | 0.450 | 0.095 | n=83 (8 R / 75 N); ≈ base rate |

Donor-level at 0.5 threshold: balanced accuracy **0.500**, confusion
`[[34, 0], [10, 0]]` — predicts *everyone* non-recurrent.

**Read:** essentially not learnable here. The faint 0.66 AUC is **not
statistically distinguishable from chance** given only 10 positive donors
(CI ≈ ±0.15–0.20), and it **does not transfer** (external 0.45). Matches the
expected "relatively poor performance."

---

## Why a 42,536-gene model isn't ruined by overfitting (and when it is)

`p ≫ n` (42,536 genes, ~85 training samples/fold). The model **always fits the
training set perfectly** — that's guaranteed by geometry and tells us nothing.
What matters is held-out performance, governed by two facts:

- The data has **effective rank 127**, not 42,536 (genes are highly redundant);
  the fit lives in an ~n-dimensional subspace.
- Held-out CV AUC for Task A is **flat across 7 orders of magnitude of `C`**
  (regularization), so the result is driven by genuine, low-dimensional class
  separation — L2 is barely doing anything.

L2-strength sweep, Task A (train AUC = 1.000 throughout):

| C (=1/λ) | Held-out CV AUC |
|---|---|
| 0.001 (strong reg) | 0.984 |
| 1 (default) | 0.988 |
| 10,000 (almost none) | 0.987 |

The contrast: **the same perfect train fit** yields CV 0.99 for Task A but CV
0.66 / external 0.45 for Task B. That train-vs-out-of-sample gap in Task B **is**
the classic p≫n overfitting — it only becomes visible when the signal is weak.

## Caveats

1. Within-PNAS cancer-vs-normal AUC is an **upper bound** — possible batch
   confounding between cancer and normal sample sets.
2. Recurrence effective n is **44 donors / 10 positives** → estimates are noisy
   and unstable; report donor-level metrics, not sample-level.
3. Cross-cohort transfer is weak (cancer) to absent (recurrence) — likely batch
   effects between PNAS and validation.
4. Default 0.5 threshold is poorly calibrated for the minority class in both
   tasks; choose an operating point if a decision rule is needed.

## Reproduce

```bash
PY=~/opt/anaconda3/bin/python          # Anaconda base: pandas/sklearn
$PY src/preprocess.py                  # build processed/*.pkl + folds
$PY src/model_cancer_vs_normal.py      # Task A
$PY src/model_recurrence.py            # Task B
```
