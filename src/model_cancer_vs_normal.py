#!/usr/bin/env python
"""
Simple baseline for cancer vs. normal (Task A).

Model: StandardScaler -> L2 logistic regression (class_weight='balanced').
Everything is fit INSIDE each fold on training data only (no leakage). We reuse
the donor-grouped 3-fold split from preprocess.py and report out-of-fold metrics,
plus a quick external-validation pass (train on all PNAS, test on validation).
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    accuracy_score, confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed"


def make_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            max_iter=2000, solver="lbfgs")),
    ])


def main() -> None:
    d = pickle.load(open(PROC / "cancer_vs_normal.pkl", "rb"))
    X, y, fold = d["X"].values, d["y"].values, d["fold"].values
    n_splits = d["n_splits"]

    # ---- out-of-fold cross-validation ----
    oof = np.full(len(y), np.nan)
    per_fold = []
    for k in range(n_splits):
        tr, te = fold != k, fold == k
        model = make_model().fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        oof[te] = p
        per_fold.append(roc_auc_score(y[te], p))

    pred = (oof >= 0.5).astype(int)
    print("=" * 64)
    print("Task A: cancer vs normal  |  StandardScaler + L2 logistic")
    print(f"  n={len(y)}  (cancer={int(y.sum())}, normal={int((1-y).sum())})  "
          f"features={X.shape[1]:,}")
    print("-" * 64)
    print(f"  per-fold ROC-AUC : {[f'{a:.3f}' for a in per_fold]}  "
          f"(mean {np.mean(per_fold):.3f} ± {np.std(per_fold):.3f})")
    print(f"  pooled  ROC-AUC : {roc_auc_score(y, oof):.3f}")
    print(f"  pooled  PR-AUC  : {average_precision_score(y, oof):.3f}")
    print(f"  accuracy        : {accuracy_score(y, pred):.3f}  "
          f"(majority-class baseline {max(y.mean(), 1-y.mean()):.3f})")
    print(f"  balanced acc    : {balanced_accuracy_score(y, pred):.3f}")
    print(f"  confusion (rows=true 0/1, cols=pred 0/1):\n{confusion_matrix(y, pred)}")

    # ---- external validation (train on ALL PNAS, test on validation cohort) ----
    val = pickle.load(open(PROC / "validation.pkl", "rb"))
    Xv, yv = val["X"].values, val["y_cancer"].values
    model = make_model().fit(X, y)
    pv = model.predict_proba(Xv)[:, 1]
    print("-" * 64)
    print("  External validation (trained on all 128 PNAS samples):")
    print(f"    n={len(yv)} (cancer={int(yv.sum())}, normal={int((1-yv).sum())})  "
          f"ROC-AUC {roc_auc_score(yv, pv):.3f}  PR-AUC {average_precision_score(yv, pv):.3f}")
    print(f"    accuracy {accuracy_score(yv, (pv>=0.5).astype(int)):.3f}  "
          f"balanced acc {balanced_accuracy_score(yv, (pv>=0.5).astype(int)):.3f}")


if __name__ == "__main__":
    main()
