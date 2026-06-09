# Breast-Cancer Classification from Serum Cell-Free RNA (SILVER-seq)

Predicting breast cancer and its recurrence from **cell-free RNA (cfRNA)** in a
single droplet of serum, using the **SILVER-seq** data from
[Zhou et al., *PNAS* 2019](https://www.pnas.org/content/116/38/19200).

We address two supervised classification tasks and, beyond raw performance,
investigate **generalization to an external cohort**, **batch correction**,
**feature-selection strategies**, and the **biological interpretability** of the
learned models.

- **Task A — Cancer vs. Normal** *(detection):* given an unseen serum cfRNA
  profile, is the donor a breast-cancer patient or a healthy control?
- **Task B — Recurrence vs. Non-recurrence** *(prognosis):* given a breast-cancer
  patient, will the disease recur? Defined at the **donor** level.

---

## Data

cfRNA whole-transcriptome expression (gene-level Ensembl TPM, 60,675 genes)
measured by **SILVER-seq** (Small-Input Liquid-Volume Extracellular RNA
sequencing), which profiles RNA from ~5–7 µL of serum.

| Cohort | n | Composition | Role |
|---|---|---|---|
| **PNAS** (primary) | 128 | 96 cancer (from **44 donors**, 10 recurrence-positive) + 32 normal | Train + cross-validation |
| **External validation** | 161 | 83 cancer + 78 normal | Held out entirely (out-of-cohort test) |

**Key structural fact:** the 96 cancer samples come from only **44 donors** —
many are longitudinal repeat draws of the same person. Recurrence is a
*donor-level* property, so the effective sample size for Task B is **44 donors /
10 positives**, not 96/28. This drives the entire evaluation design.

Provenance, file formats, and download links: [data/README.md](data/README.md),
[data/NOTE.md](data/NOTE.md). Large TPM/count matrices are not committed — see
[data/NOTE.md](data/NOTE.md) for Google-Drive links.

---

## Preprocessing — `src/preprocess.py`

Guiding rule: **apply label-blind cleaning up front; defer leakage-prone steps
(standardization, supervised feature selection) into the CV loop.**

1. **Load & repair raw files** — handle old-Mac `\r` line endings; the cancer
   matrices have no header (columns map *positionally* to patients S01–S96,
   cross-checked against sample IDs); other matrices have an unlabeled gene-ID
   column.
2. **Harmonize genes** across the cancer, normal, and validation files to a
   common gene set.
3. **Presence filter (label-blind):** keep genes with TPM > 0 in ≥ 20% of PNAS
   samples → **60,675 → 42,536 genes**.
4. **Transform:** `log2(TPM + 1)`.

Outputs go to [processed/](processed/) as `cancer_vs_normal.pkl`,
`recurrence.pkl`, `validation.pkl` (+ human-readable fold/gene CSVs). Each pickle
holds `X` (samples × genes), `y`, donor `group`, `fold`, and `genes`. Details:
[processed/SUMMARY.md](processed/SUMMARY.md).

---

## Evaluation design

**3-fold cross-validation, grouped by donor (`poiseid`) and stratified by label
at the donor level.**

- **Why grouping:** a random split could place a patient's repeat draws in both
  train and test, letting the model recognize the *person* and inflating scores.
  Donor grouping keeps every donor wholly within one fold.
- For Task A the 32 normals (no repeat structure) are each their own group → 76
  groups; for Task B → 44 donor groups.
- **3 folds** (not 5/10) so each test fold keeps ≥ 3 recurrence-positive donors;
  the 10 positives split **3 / 4 / 3**. Verified: 0 donors span > 1 fold.
- **Leakage control:** `StandardScaler` and any feature selection are fit on the
  training portion of each fold only.
- **Metrics:** ROC-AUC (primary), PR-AUC, balanced accuracy, confusion matrices.
  For Task B, **donor-level** AUC (average each donor's scores, then score
  donors) is the honest metric; sample-level is optimistic.
- **External validation:** train on the full PNAS cohort, test on the held-out
  161-sample cohort.

---

## Analyses

### 1. Multi-model comparison & robustness — `Group2_Project_Code.ipynb`

The primary analysis. **Seven classifiers** — L2 / L1 / elastic-net logistic
regression, RBF-SVM, KNN, random forest, HistGradientBoosting (each in a
`StandardScaler` pipeline) — are benchmarked on the same donor-grouped CV and on
the held-out cohort, then stress-tested with batch correction and feature
selection.

**Donor-grouped cross-validation (PNAS).** Task A separates near-perfectly:
several models reach pooled OOF AUC ≈ 1.000 and L2 logistic regression 0.993.
Task B is at chance even for the best model (elastic-net **donor-level AUC 0.60**,
L2 0.58), with balanced accuracy ≈ 0.50 — exactly the project's stated
expectation for recurrence.

**External-cohort validation (161 samples).** Performance drops out of cohort, as
expected:

| Task | Best model (external) | External ROC-AUC |
|---|---|---|
| A — cancer vs normal | L2 logistic regression | **0.74** |
| B — recurrence | RBF-SVM | **0.58** (≈ chance) |

**Batch correction.** ComBat-seq (`inmoose.pycombat.pycombat_seq`) realigns the
validation read counts to the PNAS reference. The effect is model-dependent:
RBF-SVM on Task A *improves* (0.71 → **0.73**) and becomes the best corrected
model, while L2 logistic regression drops (0.74 → 0.65) — evidence that part of
the raw-cohort signal was technical batch effect, not biology.

**Feature selection.** Six gene sets are compared on external AUC:

| Strategy | Genes | Task A best | Task B best |
|---|---|---|---|
| All | 42,536 | 0.74 (L2) | 0.58 (SVM) |
| Bio (PAM50, Oncotype DX, MammaPrint, BreastOncPx, H/I — Table S5) | 515 | 0.62 | 0.58 |
| DGE (Welch t-test, BH-FDR < 0.05) | A: 1,889 / B: 6 | **0.75 (SVM)** | 0.69 (KNN) |
| Var500 / Var1000 (top training-variance) | 500 / 1,000 | 0.64 | 0.67 (RF) |

Differentially-expressed genes give the strongest Task-A model
(**SVM on 1,889 DGE genes, AUC 0.75**); for Task B every strategy stays near
chance with only 6 genes passing FDR.

**Outputs** → [results/](results/): `compare_task{A,B}.csv`,
`feature_selection_task{A,B}.csv`, `oof_task{A,B}.csv`, fitted-model pickles, and
comparison figures (`feature_selection_comparison.png`).

### 2. Baseline reference models — `src/`

Single-model reference implementations behind the comparison, with full
write-ups in [RESULTS.md](RESULTS.md):

| Script | What it does |
|---|---|
| [src/model_cancer_vs_normal.py](src/model_cancer_vs_normal.py) | Task A: `StandardScaler → L2 logistic regression` (C=1.0, balanced) |
| [src/model_recurrence.py](src/model_recurrence.py) | Task B: same model, donor-level metrics |
| [src/model_elasticnet.py](src/model_elasticnet.py) | Elastic-net benchmark under **nested** donor-grouped CV (does sparsity fix the p≫n overfit?) |

All models fit the training set perfectly (AUC 1.000) — expected under p ≫ n
(42,536 genes vs. ~85 samples/fold) — so only held-out and external numbers are
meaningful.

### 3. Biological interpretation — `biological_inference.ipynb`

- **Feature importance** per model: coefficients for the three logistic-regression
  variants; **SHAP** (`TreeExplainer`) for random forest and HistGBM.
- **Top-20 feature overlap** across the five models, visualized with **Venn** and
  **UpSet** plots — the logistic models largely agree (~10/20 shared genes) while
  the tree models pick distinct features.
- **Candidate-gene validation:** expression of top hits (e.g. **RNY1** /
  `ENSG00000201098`, **MT-CYB** / `ENSG00000198727`) compared between cancer and
  normal in **both** the training and external cohorts via violin plots, checking
  whether discriminative genes reproduce out of cohort.

---

## Repository layout

```
data/                      raw inputs + provenance (large matrices via Drive links)
src/
  preprocess.py            build processed/*.pkl + donor folds
  model_cancer_vs_normal.py  Task A baseline
  model_recurrence.py        Task B baseline
  model_elasticnet.py        nested-CV elastic-net benchmark
processed/                 analysis-ready pickles, folds, gene list, SUMMARY.md
Group2_Project_Code.ipynb  7-model comparison, batch correction, feature selection
biological_inference.ipynb model interpretation (coef/SHAP), overlap, gene plots
results/                   CSVs, fitted models, figures (generated)
RESULTS.md                 detailed baseline results & discussion
```

---

## Reproduce

```bash
PY=~/opt/anaconda3/bin/python        # env with pandas / scikit-learn

# 1. Build processed datasets + donor folds
$PY src/preprocess.py

# 2. Baseline models
$PY src/model_cancer_vs_normal.py
$PY src/model_recurrence.py
$PY src/model_elasticnet.py

# 3. Notebooks (run top-to-bottom)
#    Group2_Project_Code.ipynb   — model comparison, batch correction, feature selection
#    biological_inference.ipynb  — interpretation & candidate-gene plots
```

Notebook extras require `inmoose` (ComBat-seq), `shap`, `mygene`, `venn`, and
`upsetplot` in addition to the core stack.

---

## Caveats

1. Within-PNAS cancer-vs-normal AUC is an **upper bound** — cancer and normal are
   different sample sets, so some separability may be batch/technical (hence the
   ComBat-seq experiment).
2. Recurrence has an effective n of **44 donors / 10 positives** → estimates are
   noisy; the best donor-level CV AUC (~0.60) is not statistically distinguishable
   from chance and does not transfer (best external ~0.58). Report donor-level,
   not sample-level, metrics.
3. Cross-cohort transfer is weak (cancer) to absent (recurrence); likely batch
   effects between PNAS and validation.
4. The default 0.5 threshold is poorly calibrated for the minority class in both
   tasks.
```