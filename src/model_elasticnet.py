#!/usr/bin/env python
"""
Elastic-net benchmark vs. the L2 baseline, on the SAME donor-grouped CV.

To answer "does elastic net help the overfit?" honestly we use NESTED CV:
  * outer = the donor-grouped 3-fold split from preprocess.py (what we report on)
  * inner = donor-grouped CV over a (C, l1_ratio) grid to PICK hyperparameters
Tuning happens only inside the inner loop, so the outer score is unbiased.

We also print the OPTIMISTIC number (best grid point chosen directly on the outer
folds) to make the selection-overfitting gap explicit, plus the median number of
genes elastic net actually keeps.
"""
from __future__ import annotations
import pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed"

C_GRID = [0.03, 0.1, 0.3, 1.0]
L1_GRID = [0.3, 0.6, 0.9]          # elastic-net mixing (0=ridge, 1=lasso)
TOPK = 2000                         # label-blind variance prefilter (fit per fold)


def _variance_score(X, y=None):
    """Label-blind feature score = per-gene variance (ignores y -> no leakage)."""
    return np.asarray(X).var(axis=0)


def enet(C, l1):
    return Pipeline([("var", SelectKBest(_variance_score, k=TOPK)),
                     ("s", StandardScaler()),
                     ("c", LogisticRegression(penalty="elasticnet", solver="saga",
                                              C=C, l1_ratio=l1, class_weight="balanced",
                                              max_iter=5000, tol=1e-3))])


def l2(C=1.0):
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(C=C, class_weight="balanced", max_iter=2000))])


def donor_auc(group, y, p):
    dl = pd.DataFrame({"g": group, "y": y, "p": p}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    return roc_auc_score(dl["y"], dl["p"])


def inner_select(Xtr, ytr, gtr):
    """Pick (C,l1) by inner donor-grouped CV sample-level AUC."""
    best, best_auc = (C_GRID[0], L1_GRID[0]), -1
    igkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
    for C in C_GRID:
        for l1 in L1_GRID:
            oof = np.full(len(ytr), np.nan)
            for a, b in igkf.split(Xtr, ytr, gtr):
                oof[b] = enet(C, l1).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
            auc = roc_auc_score(ytr, oof)
            if auc > best_auc:
                best_auc, best = auc, (C, l1)
    return best


def benchmark(task_file, label, donor_level):
    print(f"[{label}] loading + running ...", flush=True)
    D = pickle.load(open(PROC / task_file, "rb"))
    X, y, fold, group = D["X"].values, D["y"].values, D["fold"].values, D["group"].values
    score = (lambda yy, pp, gg: donor_auc(gg, yy, pp)) if donor_level else (lambda yy, pp, gg: roc_auc_score(yy, pp))

    # ---- L2 baseline (outer CV) ----
    oof_l2 = np.full(len(y), np.nan)
    for k in np.unique(fold):
        a, b = fold != k, fold == k
        oof_l2[b] = l2().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    auc_l2 = score(y, oof_l2, group)

    # ---- elastic net, NESTED (honest) ----
    oof_en = np.full(len(y), np.nan)
    picks, n_genes = [], []
    for k in np.unique(fold):
        a, b = fold != k, fold == k
        C, l1 = inner_select(X[a], y[a], group[a])
        m = enet(C, l1).fit(X[a], y[a])
        oof_en[b] = m.predict_proba(X[b])[:, 1]
        picks.append((C, l1))
        n_genes.append(int((m.named_steps["c"].coef_ != 0).sum()))
        print(f"    outer fold {k}: picked C={C}, l1={l1}", flush=True)
    auc_en_nested = score(y, oof_en, group)

    # ---- elastic net, OPTIMISTIC (best grid point on the OUTER folds) ----
    best_opt = -1
    for C in C_GRID:
        for l1 in L1_GRID:
            oof = np.full(len(y), np.nan)
            for k in np.unique(fold):
                a, b = fold != k, fold == k
                oof[b] = enet(C, l1).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
            best_opt = max(best_opt, score(y, oof, group))

    print(f"\n##### {label}  ({'donor-level' if donor_level else 'sample-level'} AUC) #####", flush=True)
    print(f"  L2 baseline (C=1, all {X.shape[1]:,} genes) : {auc_l2:.3f}")
    print(f"  Elastic net, NESTED (honest)         : {auc_en_nested:.3f}")
    print(f"  Elastic net, OPTIMISTIC              : {best_opt:.3f}   <- inflated by selecting on test folds")
    print(f"  selection-overfit gap                : +{best_opt - auc_en_nested:.3f}")
    print(f"  nonzero genes per fold (of {TOPK})     : {n_genes}")
    print(f"  hyperparams picked per fold          : {picks}", flush=True)


def main():
    benchmark("recurrence.pkl", "Task B: recurrence", donor_level=True)
    benchmark("cancer_vs_normal.pkl", "Task A: cancer vs normal", donor_level=False)


if __name__ == "__main__":
    main()
