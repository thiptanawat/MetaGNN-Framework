#!/usr/bin/env python3
"""
MetaGNN vs Expression-Based Baselines (LUAD)
=============================================

Fair comparison: baselines use raw INPUT expression features (X_R[:, 0]),
while MetaGNN uses its GNN OUTPUT scores. This isolates what the GNN
architecture adds beyond simple expression thresholding.

Baselines (all operate on raw GPR-mapped expression, NOT MetaGNN output):
  1. GIMME-score: max(0, expression - threshold) normalized to [0,1]
  2. iMAT-score: three-tier (high/medium/low) expression classification
  3. Expression threshold: binary cutoff at 50th percentile

Usage:
  cd ~/MetaGNN-LUAD
  python 10_fast_gimme_imat_luad.py [--n_patients 50]
"""

import ast
import json
import logging
import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GPR parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_gene_sets(gene_sets_str):
    if pd.isna(gene_sets_str) or not str(gene_sets_str).strip():
        return []
    s = str(gene_sets_str).strip()
    if s in ("nan", "[['[]']]", "[[[]]]", "[]", "[[]]"):
        return []
    try:
        parsed = ast.literal_eval(s)
        if not isinstance(parsed, list):
            return []
        result = []
        for group in parsed:
            if isinstance(group, list):
                genes = [str(g) for g in group if str(g) not in ('[]', '', 'nan')]
                if genes:
                    result.append(genes)
            elif isinstance(group, str) and group not in ('[]', '', 'nan'):
                result.append([group])
        return result
    except (ValueError, SyntaxError):
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data(base_dir: Path, n_patients: int = None):
    """
    Load MetaGNN output scores AND raw input features for the same patients.
    """
    data_dir = base_dir / "data_luad" / "processed"
    results_dir = base_dir / "results_luad" / "full_cohort"

    # Load MetaGNN scores
    npz = np.load(results_dir / "test_scores.npz")
    metagnn_scores = npz['scores']  # (n_patients, n_reactions)
    npz_pids = npz.get('patient_ids', None)
    if npz_pids is not None:
        patient_ids = [str(p) for p in npz_pids]
    else:
        patient_ids = None

    # Load pseudo-labels
    labels_path = data_dir / "activity_pseudolabels.npy"
    if not labels_path.exists():
        # Try .pt format
        import torch
        labels = torch.load(data_dir / "activity_pseudolabels.pt",
                            map_location='cpu').numpy().astype(np.float32)
    else:
        labels = np.load(labels_path)

    # Load GPR mask
    gpr_path = data_dir / "gpr_table.tsv"
    if not gpr_path.exists():
        # GPR table might be in CRC processed dir; check parent
        import os
        crc_dir = os.environ.get('CRC_DATA_DIR', '')
        if crc_dir:
            gpr_path = Path(crc_dir) / "gpr_table.tsv"
    gpr_df = pd.read_csv(gpr_path, sep='\t')
    col = 'gene_sets_str' if 'gene_sets_str' in gpr_df.columns else 'gpr_rule'
    gpr_mask = np.array([len(parse_gene_sets(gpr_df.iloc[i].get(col, ''))) > 0
                         for i in range(len(gpr_df))])

    # Load per-patient input features (X_R)
    feat_dir = data_dir / "reaction_features"
    available_h5 = sorted(feat_dir.glob("*.h5"))

    if patient_ids is not None:
        pid_to_file = {}
        for h5 in available_h5:
            pid = h5.stem
            pid_to_file[pid] = h5

        matched_indices = []
        matched_pids = []
        matched_files = []
        for idx, pid in enumerate(patient_ids):
            if pid in pid_to_file:
                matched_indices.append(idx)
                matched_pids.append(pid)
                matched_files.append(pid_to_file[pid])

        logger.info(f"Matched {len(matched_pids)}/{len(patient_ids)} patients to feature files")
    else:
        matched_files = available_h5
        matched_pids = [f.stem for f in matched_files]
        matched_indices = list(range(min(len(matched_files), metagnn_scores.shape[0])))

    # Subsample
    if n_patients is not None and n_patients < len(matched_indices):
        np.random.seed(42)
        sel = np.random.choice(len(matched_indices), size=n_patients, replace=False)
        matched_indices = [matched_indices[i] for i in sel]
        matched_pids = [matched_pids[i] for i in sel]
        matched_files = [matched_files[i] for i in sel]

    # Load MetaGNN scores for matched patients
    metagnn_scores = metagnn_scores[matched_indices]

    # Load input expression features
    n_rxn = metagnn_scores.shape[1]
    input_expr = np.zeros((len(matched_files), n_rxn), dtype=np.float32)
    for j, h5_path in enumerate(matched_files):
        with h5py.File(h5_path, 'r') as f:
            # Try X_R first (uppercase), then x_r
            for key in ['X_R', 'x_r', 'reaction_features']:
                if key in f:
                    X_R = f[key][:]
                    break
            else:
                raise KeyError(f"No reaction feature key found in {h5_path}")
            input_expr[j] = X_R[:, 0]  # column 0 = GPR-mapped expression

    logger.info(f"Loaded {len(matched_pids)} patients")
    logger.info(f"  MetaGNN scores: {metagnn_scores.shape}")
    logger.info(f"  Input expression: {input_expr.shape}")
    logger.info(f"  GPR-annotated: {gpr_mask.sum()}/{len(gpr_mask)}")
    logger.info(f"  Active labels: {labels.sum():.0f}/{len(labels)} ({labels.mean()*100:.1f}%)")

    return metagnn_scores, input_expr, labels, gpr_mask, matched_pids


# ─────────────────────────────────────────────────────────────────────────────
# Baseline methods (operate on raw INPUT expression, not GNN output)
# ─────────────────────────────────────────────────────────────────────────────

def gimme_score(expr: np.ndarray, threshold_pctl: float = 25.0) -> np.ndarray:
    threshold = np.percentile(expr, threshold_pctl)
    raw = expr - threshold
    raw = np.maximum(raw, 0)
    rng = raw.max() - raw.min()
    if rng > 0:
        return raw / rng
    return np.full_like(raw, 0.5)


def imat_score(expr: np.ndarray, high_pctl: float = 75.0,
               low_pctl: float = 25.0) -> np.ndarray:
    high = np.percentile(expr, high_pctl)
    low = np.percentile(expr, low_pctl)
    result = np.full_like(expr, 0.5)
    result[expr >= high] = 1.0
    result[expr <= low] = 0.0
    return result


def threshold_score(expr: np.ndarray, threshold_pctl: float = 50.0) -> np.ndarray:
    threshold = np.percentile(expr, threshold_pctl)
    return (expr > threshold).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_method(scores_2d, labels, mask=None):
    if mask is not None:
        scores_2d = scores_2d[:, mask]
        labels_eval = labels[mask]
    else:
        labels_eval = labels

    aurocs, f1s = [], []
    for j in range(scores_2d.shape[0]):
        s = scores_2d[j]
        preds = (s > 0.5).astype(np.int32)

        try:
            auc = roc_auc_score(labels_eval, s)
            if not np.isnan(auc):
                aurocs.append(auc)
        except ValueError:
            pass
        f1s.append(f1_score(labels_eval, preds, zero_division=0))

    return {
        'auroc_mean': float(np.mean(aurocs)) if aurocs else 0.0,
        'auroc_std': float(np.std(aurocs)) if aurocs else 0.0,
        'f1_mean': float(np.mean(f1s)),
        'f1_std': float(np.std(f1s)),
        'n': len(f1s),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--n_patients", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_path = Path(args.output) if args.output else (
        base_dir / "results_luad" / "method_comparison.json"
    )

    # ── Load data ────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("MetaGNN vs Expression-Based Baselines (LUAD)")
    logger.info("=" * 70)

    metagnn_scores, input_expr, labels, gpr_mask, patient_ids = load_data(
        base_dir, n_patients=args.n_patients
    )
    n_patients, n_reactions = metagnn_scores.shape

    # ── Compute baselines from INPUT expression ──────────────────────────
    logger.info("\nComputing baselines from raw input expression features...")

    gimme_all = np.zeros_like(input_expr)
    imat_all = np.zeros_like(input_expr)
    thresh_all = np.zeros_like(input_expr)

    for j in range(n_patients):
        expr = input_expr[j]
        gimme_all[j] = gimme_score(expr, threshold_pctl=25)
        imat_all[j] = imat_score(expr, high_pctl=75, low_pctl=25)
        thresh_all[j] = threshold_score(expr, threshold_pctl=50)

    # ── Evaluate ─────────────────────────────────────────────────────────
    logger.info("\nEvaluating methods...\n")

    methods = {
        "MetaGNN (GATv2)": metagnn_scores,
        "GIMME-score": gimme_all,
        "iMAT-score": imat_all,
        "Expr threshold (50th)": thresh_all,
    }

    results = {}
    for eval_name, mask, mask_key in [
        ("All reactions", None, "all"),
        ("GPR-only reactions", gpr_mask, "gpr"),
    ]:
        n_eval = n_reactions if mask is None else mask.sum()
        logger.info(f"  {eval_name} (n={n_eval}):")
        logger.info(f"  {'Method':<28} {'AUROC':>16} {'F1':>16}")
        logger.info(f"  {'-'*60}")

        for method_name, method_scores in methods.items():
            m = evaluate_method(method_scores, labels, mask=mask)
            auroc_str = f"{m['auroc_mean']:.4f}±{m['auroc_std']:.4f}"
            f1_str = f"{m['f1_mean']:.4f}±{m['f1_std']:.4f}"
            logger.info(f"  {method_name:<28} {auroc_str:>16} {f1_str:>16}")
            results[f"{method_name}__{mask_key}"] = m

        logger.info("")

    # ── Summary ──────────────────────────────────────────────────────────
    gnn = results["MetaGNN (GATv2)__gpr"]
    gim = results["GIMME-score__gpr"]
    imt = results["iMAT-score__gpr"]
    thr = results["Expr threshold (50th)__gpr"]

    logger.info("=" * 70)
    logger.info("Summary (GPR-only reactions — fair comparison):")
    logger.info("=" * 70)
    logger.info(f"  MetaGNN AUROC:     {gnn['auroc_mean']:.4f} ± {gnn['auroc_std']:.4f}")
    logger.info(f"  GIMME AUROC:       {gim['auroc_mean']:.4f} ± {gim['auroc_std']:.4f}")
    logger.info(f"  iMAT AUROC:        {imt['auroc_mean']:.4f} ± {imt['auroc_std']:.4f}")
    logger.info(f"  Threshold AUROC:   {thr['auroc_mean']:.4f} ± {thr['auroc_std']:.4f}")

    d_gimme = gnn['auroc_mean'] - gim['auroc_mean']
    d_imat = gnn['auroc_mean'] - imt['auroc_mean']
    d_thresh = gnn['auroc_mean'] - thr['auroc_mean']

    logger.info(f"\n  MetaGNN vs GIMME:     {d_gimme:+.4f} AUROC")
    logger.info(f"  MetaGNN vs iMAT:      {d_imat:+.4f} AUROC")
    logger.info(f"  MetaGNN vs Threshold: {d_thresh:+.4f} AUROC")
    logger.info("=" * 70)

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "cancer_type": "luad",
        "n_patients": n_patients,
        "n_reactions": n_reactions,
        "n_gpr_reactions": int(gpr_mask.sum()),
        "note": "Baselines use raw INPUT expression features; MetaGNN uses GNN output scores",
        "methods": results,
        "summary_gpr": {
            "metagnn_auroc": gnn['auroc_mean'],
            "gimme_auroc": gim['auroc_mean'],
            "imat_auroc": imt['auroc_mean'],
            "threshold_auroc": thr['auroc_mean'],
            "delta_gimme": float(d_gimme),
            "delta_imat": float(d_imat),
            "delta_threshold": float(d_thresh),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
