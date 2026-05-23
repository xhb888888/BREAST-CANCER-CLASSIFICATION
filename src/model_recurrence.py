#!/usr/bin/env python
"""
Simple baseline for recurrence vs. non-recurrence (Task B). Same model as Task A
(StandardScaler + L2 logistic, class_weight balanced), fit inside each donor-grouped
fold. Reports BOTH sample-level and donor-level metrics -- the donor level is the
honest one (recurrence is a donor property; effective n = 44 donors / 10 positive).
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
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
        ("clf", LogisticRegression(penalty="l2", C=1.0, class_weight="balanced",
                                   max_iter=2000, solver="lbfgs")),
    ])


def main() -> None:
    d = pickle.load(open(PROC / "recurrence.pkl", "rb"))
    X, y = d["X"].values, d["y"].values
    fold, group = d["fold"].values, d["group"].values
    ns = d["n_splits"]

    oof = np.full(len(y), np.nan)
    per_fold, tr_aucs = [], []
    for k in range(ns):
        tr, te = fold != k, fold == k
        m = make_model().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        per_fold.append(roc_auc_score(y[te], oof[te]))
        tr_aucs.append(roc_auc_score(y[tr], m.predict_proba(X[tr])[:, 1]))

    print("=" * 64)
    print("Task B: recurrence vs non-recurrence | StandardScaler + L2 logistic")
    print(f"  samples={len(y)} (R={int(y.sum())}, N={int((1-y).sum())}), "
          f"donors={len(set(group))}, features={X.shape[1]:,}")
    print("-" * 64)
    print("  SAMPLE level (n=96, treats follow-ups as independent -> optimistic):")
    print(f"    train AUC (mean) : {np.mean(tr_aucs):.3f}   <- perfect-ish fit, as in Task A")
    print(f"    per-fold CV AUC  : {[f'{a:.3f}' for a in per_fold]}")
    print(f"    pooled  CV AUC   : {roc_auc_score(y, oof):.3f}   PR-AUC {average_precision_score(y, oof):.3f}")

    # ---- donor-level (the honest metric) ----
    dl = pd.DataFrame({"g": group, "y": y, "p": oof}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    auc_d = roc_auc_score(dl["y"], dl["p"])
    pred_d = (dl["p"] >= 0.5).astype(int)
    print("  DONOR level (n=44, 10 positive -> the real estimate):")
    print(f"    ROC-AUC {auc_d:.3f}   PR-AUC {average_precision_score(dl['y'], dl['p']):.3f}   "
          f"(chance AUC = 0.5, positive rate = {dl['y'].mean():.2f})")
    print(f"    balanced acc {balanced_accuracy_score(dl['y'], pred_d):.3f}   "
          f"confusion {confusion_matrix(dl['y'], pred_d).tolist()}")

    # ---- external validation (train on all PNAS BC, test on validation BC) ----
    val = pickle.load(open(PROC / "validation.pkl", "rb"))
    mask = val["y_recur"].notna().values
    Xv, yv = val["X"].values[mask], val["y_recur"].values[mask].astype(int)
    m = make_model().fit(X, y)
    pv = m.predict_proba(Xv)[:, 1]
    print("-" * 64)
    print(f"  External validation BC (n={len(yv)}, R={int(yv.sum())}, N={int((1-yv).sum())}):")
    print(f"    ROC-AUC {roc_auc_score(yv, pv):.3f}   PR-AUC {average_precision_score(yv, pv):.3f}")


if __name__ == "__main__":
    main()
