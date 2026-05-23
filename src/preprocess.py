#!/usr/bin/env python
"""
Preprocessing pipeline for the SILVER-Seq cell-free RNA breast-cancer study.

Produces analysis-ready datasets for two classification tasks:
  A) cancer vs. normal      (PNAS: 96 BC samples + 32 normal serum samples)
  B) recurrence vs. non-recurrence (PNAS: 96 BC samples, donor-level label)

Plus the optional external validation cohort (161 samples), held out entirely.

Key data quirks handled here (discovered during exploration):
  * pnas_patient_info.csv uses old-Mac carriage-return (\\r) line endings.
  * pnas_*_96_nodup.txt expression matrices have NO header row; their 96
    columns map *positionally* to patient_info rows S01..S96.
  * normal / validation matrices DO have a header but the gene-id column is
    unlabeled (header has N fields, data rows have N+1) -> read with index_col=0.
  * Files are labeled "exon level" but IDs are gene-level Ensembl (ENSG); 60,675
    genes, no duplicates.

CRITICAL modeling fact baked into the CV design:
  Recurrence is a DONOR-level property (constant within poiseid). There are only
  44 donors and 10 recurrence-positive donors behind the 96 samples. We therefore
  split by donor (StratifiedGroupKFold grouped on poiseid) so a person's repeat
  follow-up draws never straddle train/test.

What is and isn't baked into the saved matrices (to avoid target leakage):
  * APPLIED here (label-blind, safe): gene presence filter + log2(TPM+1).
  * DEFERRED to the modeling CV loop (must be fit on train folds only):
    z-score standardization and any supervised / variance feature selection.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "processed"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
N_SPLITS = 3                 # 3-fold: keeps ~3 recurrence-positive donors / test fold
RANDOM_STATE = 42
MIN_DETECT_FRAC = 0.20       # keep genes with TPM>0 in >= 20% of PNAS samples


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_patient_info() -> pd.DataFrame:
    """96 BC samples, indexed S01..S96 in file order (matches expression columns)."""
    df = pd.read_csv(DATA / "pnas_patient_info.csv", lineterminator="\r")
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)
    assert len(df) == 96, f"expected 96 BC samples, got {len(df)}"

    # Position i (1-based) <-> sample S{i}. Cross-check against sample_id prefix.
    df["sample"] = [f"S{i:02d}" for i in range(1, len(df) + 1)]
    prefix = df["sample_id"].astype(str).str.split("_").str[0]
    assert (prefix == df["sample"]).all(), "sample_id prefix != positional S## order"

    df["donor"] = "BC_" + df["poiseid"].astype(str)          # group key
    df["recur"] = df["recurStatus"].astype(str).str[0]       # 'R' / 'N'
    df["y_recur"] = (df["recur"] == "R").astype(int)
    return df.set_index("sample")


def _read_expr_no_header(path: Path, sample_names: list[str]) -> pd.DataFrame:
    """BC matrices: no header; col0=gene, cols map positionally to sample_names.
    Returns samples x genes."""
    m = pd.read_csv(path, sep="\t", header=None, index_col=0)
    assert m.shape[1] == len(sample_names), (
        f"{path.name}: {m.shape[1]} value cols vs {len(sample_names)} samples"
    )
    m.columns = sample_names
    m.index.name = "gene"
    return m.T  # samples x genes


def _read_expr_with_header(path: Path) -> pd.DataFrame:
    """normal / validation matrices: header present, gene-id column unlabeled.
    Returns samples x genes."""
    m = pd.read_csv(path, sep="\t", index_col=0)
    m.index.name = "gene"
    return m.T  # samples x genes


def load_pnas_expression(sample_order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bc = _read_expr_no_header(DATA / "pnas_tpm_96_nodup.txt", sample_order)
    normal = _read_expr_with_header(DATA / "pnas_normal_tpm.txt")
    return bc, normal


def load_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (expression samples x genes, sample metadata) for the 161-sample cohort."""
    expr = _read_expr_with_header(DATA / "validation_exon_tpm")

    bc_meta = pd.read_excel(DATA / "validation_bc_meta.xlsx")
    bc_meta = bc_meta.rename(columns={bc_meta.columns[0]: "sample"})
    rec_col = [c for c in bc_meta.columns if "Recurrence" in c][0]
    bc_meta["y_cancer"] = 1
    bc_meta["y_recur"] = (
        bc_meta[rec_col].astype(str).str.strip().str.lower().eq("recurrent").astype(int)
    )

    nm_meta = pd.read_excel(DATA / "validation_normal_meta.xlsx")
    nm_meta = nm_meta.rename(columns={nm_meta.columns[0]: "sample"})
    nm_meta["y_cancer"] = 0
    nm_meta["y_recur"] = np.nan  # normals have no recurrence label

    meta = pd.concat(
        [bc_meta[["sample", "y_cancer", "y_recur"]],
         nm_meta[["sample", "y_cancer", "y_recur"]]],
        ignore_index=True,
    ).dropna(subset=["sample"]).set_index("sample")

    # keep only samples present in both expression and metadata, in expr order
    common = [s for s in expr.index if s in meta.index]
    return expr.loc[common], meta.loc[common]


# ----------------------------------------------------------------------------
# Transform helpers
# ----------------------------------------------------------------------------
def select_genes(pnas_samples_x_genes: pd.DataFrame) -> list[str]:
    """Label-blind presence filter on the PNAS training universe."""
    detect_frac = (pnas_samples_x_genes > 0).mean(axis=0)
    genes = detect_frac[detect_frac >= MIN_DETECT_FRAC].index.tolist()
    return genes


def log2_tpm(df_samples_x_genes: pd.DataFrame) -> pd.DataFrame:
    return np.log2(df_samples_x_genes + 1.0)


def assign_folds(y: pd.Series, groups: pd.Series) -> pd.Series:
    """Donor-grouped, stratified 3-fold -> per-sample test-fold index (0..N_SPLITS-1).

    The label is constant within group (donor), so we split at the *group* level
    and broadcast to samples. This keeps every donor wholly in one fold (no
    leakage) AND distributes the few positive donors evenly across folds --
    which sample-level StratifiedGroupKFold fails to do here.
    """
    glabel = pd.DataFrame({"y": y.values, "g": groups.values}).groupby("g")["y"].agg(
        ["first", "nunique"])
    assert (glabel["nunique"] == 1).all(), "label is not constant within group"

    uniq = glabel.index.to_numpy()
    ulab = glabel["first"].to_numpy()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    g2fold = {}
    for k, (_, test_idx) in enumerate(skf.split(np.zeros(len(uniq)), ulab)):
        for i in test_idx:
            g2fold[uniq[i]] = k

    fold = groups.map(g2fold).astype(int)
    fold.index = y.index
    assert (fold >= 0).all(), "every sample must land in exactly one test fold"
    return fold


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    info = load_patient_info()
    sample_order = info.index.tolist()                      # S01..S96
    bc_tpm, normal_tpm = load_pnas_expression(sample_order)

    # ---- gene alignment across PNAS cohorts ----
    genes_common = bc_tpm.columns.intersection(normal_tpm.columns)
    bc_tpm, normal_tpm = bc_tpm[genes_common], normal_tpm[genes_common]

    # presence filter on PNAS BC+normal pooled (label-blind)
    pnas_pool = pd.concat([bc_tpm, normal_tpm], axis=0)
    genes = select_genes(pnas_pool)
    bc_tpm, normal_tpm = bc_tpm[genes], normal_tpm[genes]

    # ---- Task A: cancer vs normal ----
    Xa = log2_tpm(pd.concat([bc_tpm, normal_tpm], axis=0))
    ya = pd.Series([1] * len(bc_tpm) + [0] * len(normal_tpm), index=Xa.index, name="y")
    groups_a = pd.Series(
        info["donor"].tolist() + [f"NORM_{s}" for s in normal_tpm.index],
        index=Xa.index, name="group",
    )
    folds_a = assign_folds(ya, groups_a)

    # ---- Task B: recurrence vs non-recurrence (BC only) ----
    Xb = log2_tpm(bc_tpm)
    yb = info["y_recur"].rename("y")
    groups_b = info["donor"].rename("group")
    folds_b = assign_folds(yb, groups_b)

    # ---- External validation cohort ----
    val_expr, val_meta = load_validation()
    val_expr = val_expr.reindex(columns=genes, fill_value=0.0)  # align to PNAS gene set
    Xv = log2_tpm(val_expr)

    # ---- Save ----
    cancer_vs_normal = {
        "X": Xa, "y": ya, "group": groups_a, "fold": folds_a,
        "genes": genes, "n_splits": N_SPLITS,
    }
    recurrence = {
        "X": Xb, "y": yb, "group": groups_b, "fold": folds_b,
        "genes": genes, "n_splits": N_SPLITS,
    }
    validation = {
        "X": Xv, "y_cancer": val_meta["y_cancer"], "y_recur": val_meta["y_recur"],
        "genes": genes,
    }

    with open(OUT / "cancer_vs_normal.pkl", "wb") as fh:
        pickle.dump(cancer_vs_normal, fh)
    with open(OUT / "recurrence.pkl", "wb") as fh:
        pickle.dump(recurrence, fh)
    with open(OUT / "validation.pkl", "wb") as fh:
        pickle.dump(validation, fh)

    # transparent CSVs (labels + folds; matrices stay in pickle for size)
    pd.DataFrame({"y": ya, "group": groups_a, "fold": folds_a}).to_csv(
        OUT / "folds_cancer_vs_normal.csv")
    pd.DataFrame({"y": yb, "group": groups_b, "fold": folds_b}).to_csv(
        OUT / "folds_recurrence.csv")
    pd.Series(genes, name="gene").to_csv(OUT / "feature_genes.txt", index=False)

    # ---- summary ----
    def fold_table(y, fold):
        t = pd.crosstab(fold, y)
        t.index.name = "fold"
        return t

    summary = {
        "n_genes_after_filter": len(genes),
        "min_detect_frac": MIN_DETECT_FRAC,
        "n_splits": N_SPLITS,
        "cancer_vs_normal": {
            "n_samples": int(len(ya)),
            "n_groups": int(groups_a.nunique()),
            "class_counts": ya.value_counts().to_dict(),
        },
        "recurrence": {
            "n_samples": int(len(yb)),
            "n_donors": int(groups_b.nunique()),
            "n_pos_donors": int(info.groupby("donor")["y_recur"].first().sum()),
            "class_counts_samples": yb.value_counts().to_dict(),
        },
        "validation": {
            "n_samples": int(len(Xv)),
            "y_cancer_counts": val_meta["y_cancer"].value_counts().to_dict(),
            "y_recur_counts": val_meta["y_recur"].dropna().astype(int).value_counts().to_dict(),
        },
    }
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=int)

    print("=" * 70)
    print(f"Genes after presence filter (>= {MIN_DETECT_FRAC:.0%} of PNAS samples): "
          f"{len(genes):,} / {len(genes_common):,}")
    print("\n[Task A] cancer vs normal  ->  X", Xa.shape,
          "| classes", ya.value_counts().to_dict(),
          "| groups", groups_a.nunique())
    print(fold_table(ya, folds_a))
    print("\n[Task B] recurrence        ->  X", Xb.shape,
          "| sample classes", yb.value_counts().to_dict(),
          "| donors", groups_b.nunique(),
          "| pos donors", summary["recurrence"]["n_pos_donors"])
    print(fold_table(yb, folds_b))
    print("\n[External] validation      ->  X", Xv.shape,
          "| cancer", val_meta["y_cancer"].value_counts().to_dict(),
          "| recur(BC)", val_meta["y_recur"].dropna().astype(int).value_counts().to_dict())
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
