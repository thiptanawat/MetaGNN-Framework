"""Step 6: turn the per-arm prediction files into the audit's numbers.

Everything is computed on the (held-out line x panel gene) grid.

Orientation convention: a *score* is always "higher = more dependent". DeepMeta's preds_raw is
already oriented that way; a baseline that predicts a gene-effect value is negated before
scoring, so that all scores are comparable. Gene effect itself is kept in its native DepMap
orientation (more negative = more dependent), and concordance is defined against NEGATIVE gene
effect.

Three nested domains for every statistic, reported side by side and labelled:
  primary_template_domain  : a gene-line pair counts only when BOTH lines' lineage tissue
                             templates contain the gene's node at all. A node in the template
                             but absent from the sample's SEN (unexpressed) keeps prediction 0;
                             a line whose template lacks the node is not scored on that gene.
                             This is the fair domain: the model is only asked about genes its
                             tissue network actually carries.
  secondary_explicit_only  : only pairs where every arm emitted an explicit prediction.
  tertiary_zero_fill_all   : the whole grid, absent nodes zero-filled (widest, most diluted).

Statistics
  (a) macro within-gene concordance C = mean over panel genes of the share of held-out line
      pairs (different gene effect) whose score ordering matches the ordering of -gene effect,
      score ties counted one half.
  (b) per-gene Spearman(score, -gene effect); count of genes with constant scores.
  (c) per-line AUROC of the score against gene effect < -0.5 across the panel.
  (d) donor effect D_arm = mean_g (C_own,g - C_arm,g) on identical gene-line pairs.
  (e) baselines fitted on the development lines (per-gene mean, lineage-conditioned per-gene
      mean, expression ridge, multi-output random forest).
  (f) 95% percentile bootstrap intervals: 2000 replicates over held-out lines grouped by
      PatientID (all pairs rebuilt inside each replicate, via the quadratic form below), and
      2000 replicates over gene families.
  (g) each donor effect labelled against the smallest effect of interest, 0.01 concordance.

Bootstrap identity used throughout: with W[g,i,j] the concordance contribution of the ordered
pair (i,j) for gene g (1 concordant, 1/2 score tie, 0 discordant) and V[g,i,j] its validity
(1 when both lines are observed and their gene effects differ), a resample that draws line i
n_i times has C_g = (n' W_g n) / (n' V_g n). V does not depend on the arm, which is what makes
every arm's concordance rest on exactly the same pairs.

  python3 metrics.py --seed 11
  python3 metrics.py --seed 11 --arms own within --boot 2000 --rf_trees 200 --out smoke.json
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import config as C
import jsonutil
from manifest import load_expression, load_gene_effect
import deepmeta_io as dio


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- loading
def load_preds(seed, arms, panel, cells_allowed, preds_dir=None):
    """arm -> (score matrix cells x genes with 0 for nodes absent from the SEN, explicit mask)."""
    d = preds_dir or os.path.join(C.PREDS, f"seed{seed}")
    frames, cells = {}, None
    for arm in arms:
        p = os.path.join(d, f"preds_{arm}.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        df = df[df.gene_name.isin(set(panel.ensembl))]
        dup = df.duplicated(["cell", "gene_name"]).sum()
        if dup:
            df = df.drop_duplicates(["cell", "gene_name"], keep="last")
        frames[arm] = (df, int(dup))
        cs = set(df.cell) & set(cells_allowed)
        cells = cs if cells is None else (cells & cs)
    cells = sorted(cells)
    genes = list(panel.ensembl)
    gi = {g: i for i, g in enumerate(genes)}
    ci = {c: i for i, c in enumerate(cells)}
    P, M, dups = {}, {}, {}
    for arm, (df, dup) in frames.items():
        x = np.zeros((len(cells), len(genes)), dtype=np.float64)
        m = np.zeros((len(cells), len(genes)), dtype=bool)
        df = df[df.cell.isin(ci)]
        r = df.cell.map(ci).to_numpy()
        c = df.gene_name.map(gi).to_numpy()
        x[r, c] = df.preds_raw.to_numpy()
        m[r, c] = True
        P[arm], M[arm], dups[arm] = x, m, dup
    return cells, genes, P, M, dups


# ---------------------------------------------------------------- concordance machinery
def pair_tensors(score, ge, valid_extra=None, chunk=64):
    """W (concordance contributions) and V (valid pairs) as (genes, cells, cells) float32."""
    n_c, n_g = ge.shape
    W = np.zeros((n_g, n_c, n_c), dtype=np.float32)
    V = np.zeros((n_g, n_c, n_c), dtype=np.float32)
    obs = ~np.isnan(ge)
    for s in range(0, n_g, chunk):
        e = min(s + chunk, n_g)
        g = ge[:, s:e].T                      # (chunk, cells)
        p = score[:, s:e].T
        o = obs[:, s:e].T
        dg = g[:, :, None] - g[:, None, :]    # gene effect difference
        dp = p[:, :, None] - p[:, None, :]
        ok = o[:, :, None] & o[:, None, :]
        ok &= dg != 0                          # "different gene effect"
        if valid_extra is not None:
            m = valid_extra[:, s:e].T
            ok &= m[:, :, None] & m[:, None, :]
        idx = np.arange(n_c)
        ok[:, idx, idx] = False
        # more negative gene effect must receive the higher score
        conc = ((dp > 0) & (dg < 0)) | ((dp < 0) & (dg > 0))
        tie = dp == 0
        w = np.where(conc, 1.0, 0.0) + np.where(tie, 0.5, 0.0)
        W[s:e] = (w * ok).astype(np.float32)
        V[s:e] = ok.astype(np.float32)
    return W, V


def concordance(W, V, n=None):
    """Per-gene concordance for a resample with per-line multiplicities n (None = each once)."""
    if n is None:
        num = W.sum(axis=(1, 2))
        den = V.sum(axis=(1, 2))
    else:
        num = ((W @ n) * n).sum(axis=1)
        den = ((V @ n) * n).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return c, den


def macro(c):
    if not np.isfinite(c).any():
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(c))


# ---------------------------------------------------------------- baselines
def fit_baselines(panel, samples, cells, genes, rf_trees, rf_jobs, seed, ridge_alphas):
    """Scores (higher = more dependent) on the held-out grid, from development-line data only."""
    dev = [s["ModelID"] for s in samples if s["development"] in (True, "True")]
    lineage = {s["ModelID"]: s["OncotreeLineage"] for s in samples}
    ge_cols = list(panel.ge_col)
    G_dev = load_gene_effect(dev, ge_cols).to_numpy()
    mu = np.nanmean(G_dev, axis=0)
    Y = np.where(np.isnan(G_dev), mu, G_dev)             # mean-imputed targets for the fits
    info = {"development_lines": len(dev),
            "target_nan_fraction": float(np.isnan(G_dev).mean())}

    out = {}
    # 1) training-only per-gene mean  (constant across lines -> all score ties -> C = 1/2)
    out["baseline_gene_mean"] = np.tile(-mu, (len(cells), 1))

    # 2) lineage-conditioned per-gene mean
    lin_mean, n_lin = {}, {}
    for l in sorted({lineage[m] for m in dev}):
        rows = [i for i, m in enumerate(dev) if lineage[m] == l]
        with np.errstate(invalid="ignore"):
            with warnings.catch_warnings():        # a gene unscreened in a whole lineage
                warnings.simplefilter("ignore", RuntimeWarning)
                lin_mean[l] = np.nanmean(G_dev[rows], axis=0)
        n_lin[l] = len(rows)
    lc = np.vstack([lin_mean.get(lineage[c], mu) for c in cells])
    lc = np.where(np.isnan(lc), mu, lc)
    out["baseline_lineage_mean"] = -lc
    info["lineage_mean_fallback_cells"] = [c for c in cells if lineage[c] not in lin_mean]
    info["development_lines_per_lineage"] = n_lin

    # expression design: the 7993 model genes, log2(TPM+1)
    order = dio.model_gene_order()
    expr = load_expression(sorted(set(dev) | set(cells)))[order]
    Xd = expr.loc[dev].to_numpy()
    Xh = expr.loc[cells].to_numpy()

    # 3) expression-only ridge, single alpha chosen by 5-fold CV on the development lines
    xm, ym = Xd.mean(0), Y.mean(0)
    cv_mse = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in kf.split(Xd):
        A = Xd[tr] - Xd[tr].mean(0)
        B = Y[tr] - Y[tr].mean(0)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        UtB = U.T @ B
        for a in ridge_alphas:
            coef = Vt.T @ ((s / (s ** 2 + a))[:, None] * UtB)
            pred = (Xd[te] - Xd[tr].mean(0)) @ coef + Y[tr].mean(0)
            cv_mse[a] = cv_mse.get(a, 0.0) + float(((pred - Y[te]) ** 2).mean()) / 5
    alpha = min(cv_mse, key=cv_mse.get)
    A = Xd - xm
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    coef = Vt.T @ ((s / (s ** 2 + alpha))[:, None] * (U.T @ (Y - ym)))
    out["baseline_ridge_expression"] = -((Xh - xm) @ coef + ym)
    info["ridge_alpha"] = float(alpha)
    info["ridge_cv_mse"] = {str(k): v for k, v in cv_mse.items()}
    info["ridge_n_features"] = int(Xd.shape[1])

    # 4) multi-output random forest on the 2000 highest-variance model genes
    var = Xd.var(axis=0)
    top = np.argsort(-var)[:2000]
    rf = RandomForestRegressor(n_estimators=rf_trees, max_features="sqrt",
                               random_state=seed, n_jobs=rf_jobs)
    t0 = time.time()
    rf.fit(Xd[:, top], Y)
    out["baseline_random_forest"] = -rf.predict(Xh[:, top])
    info["rf_trees"] = rf_trees
    info["rf_features"] = int(len(top))
    info["rf_fit_seconds"] = round(time.time() - t0, 1)
    return out, info


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=C.PRIMARY_SEED)
    ap.add_argument("--arms", nargs="*", default=None, help="default: every arm with a preds file")
    ap.add_argument("--boot", type=int, default=C.N_BOOT)
    ap.add_argument("--rf_trees", type=int, default=200)
    ap.add_argument("--rf_jobs", type=int, default=2)
    ap.add_argument("--no_baselines", action="store_true")
    ap.add_argument("--ridge_alphas", nargs="*", type=float,
                    default=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ap.add_argument("--preds_dir", default=None)
    ap.add_argument("--manifest", default=C.MANIFEST_JSON)
    ap.add_argument("--out", default=C.RESULTS_JSON)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    panel = pd.DataFrame(man["panel"])
    samples = man["samples"]
    heldout = [s["ModelID"] for s in samples if s["heldout"] in (True, "True")]
    patient = {s["ModelID"]: s["PatientID"] for s in samples}
    lineage = {s["ModelID"]: s["OncotreeLineage"] for s in samples}

    preds_dir = args.preds_dir or os.path.join(C.PREDS, f"seed{args.seed}")
    arms = args.arms or [a for a in C.ARM_NAMES
                         if os.path.exists(os.path.join(preds_dir, f"preds_{a}.csv"))]
    assert "own" in arms, "the own arm is the reference and must be present"
    cells, genes, P, M, dups = load_preds(args.seed, arms, panel, heldout, preds_dir)
    log(f"arms {arms}: {len(cells)} held-out lines x {len(genes)} panel genes")

    ge = load_gene_effect(cells, list(panel.ge_col)).to_numpy()
    res = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed, "arms": arms, "preds_dir": preds_dir,
        "counts": {
            "heldout_lines_in_manifest": len(heldout),
            "heldout_lines_scored": len(cells),
            "panel_genes": len(genes),
            "cells": cells,
            "duplicate_prediction_rows_dropped": dups,
            "explicit_prediction_fraction": {a: float(M[a].mean()) for a in arms},
            "gene_effect_observed_fraction": float((~np.isnan(ge)).mean()),
            "dependency_threshold": C.DEP_THRESHOLD,
            "dependency_positive_fraction": float(np.nanmean(ge < C.DEP_THRESHOLD)),
            "patients_among_scored_lines": len({patient[c] for c in cells}),
            "lineages_among_scored_lines": len({lineage[c] for c in cells}),
        },
        "analyses": {},
    }

    scores = {a: P[a] for a in arms}
    if not args.no_baselines:
        log("fitting baselines on the development lines")
        base, binfo = fit_baselines(panel, samples, cells, genes, args.rf_trees,
                                    args.rf_jobs, args.seed, args.ridge_alphas)
        scores.update(base)
        res["baselines_info"] = binfo
        # held-out squared error of every gene-effect-predicting baseline, and of the per-gene
        # mean it is measured against (DeepMeta emits a probability, so it has no MSE here)
        obs = ~np.isnan(ge)
        mse = {}
        for name, sc in base.items():
            err = (-sc - np.where(obs, ge, 0.0)) ** 2
            mse[name] = float(err[obs].mean())
        res["baselines_info"]["heldout_mse"] = mse
        res["baselines_info"]["heldout_mse_ratio_to_gene_mean"] = {
            k: (v / mse["baseline_gene_mean"] if mse["baseline_gene_mean"] > 0 else float("nan"))
            for k, v in mse.items()}
        log("baseline held-out MSE: " + json.dumps(mse))

    all_explicit = np.ones_like(M[arms[0]])
    for a in arms:
        all_explicit &= M[a]
    res["counts"]["all_arms_explicit_fraction"] = float(all_explicit.mean())

    # template domain: for each held-out line, which panel-gene nodes its lineage tissue
    # template (the assigned GSM enzyme graph) contains at all. A node in the template but
    # absent from this sample's SEN (because it is unexpressed) keeps its zero-filled
    # prediction and is still scored; a node the template lacks is not scored for that line.
    net_of = {s["ModelID"]: s["net_template"] for s in samples}
    template_nodes = {nt: (lambda n: set(n["from"]) | set(n["to"]))(dio.tissue_net(nt))
                      for nt in sorted({net_of[c] for c in cells})}
    domain = np.zeros((len(cells), len(genes)), dtype=bool)
    for i, c in enumerate(cells):
        nodes = template_nodes[net_of[c]]
        domain[i] = np.fromiter((g in nodes for g in genes), dtype=bool, count=len(genes))
    domain_size = domain.sum(axis=0)                     # scored lines whose template has the gene
    res["counts"]["template_domain_fraction"] = float(domain.mean())
    res["counts"]["panel_genes_in_no_heldout_template"] = int((domain_size == 0).sum())
    res["counts"]["template_domain_size_per_gene"] = {
        "min": int(domain_size.min()), "median": int(np.median(domain_size)),
        "max": int(domain_size.max()), "mean": float(domain_size.mean())}

    for analysis, extra in (("primary_template_domain", domain),
                            ("secondary_explicit_only", all_explicit),
                            ("tertiary_zero_fill_all", None)):
        log(f"analysis: {analysis}")
        A = {}
        W, V, cg, den = {}, None, {}, None
        for name, sc in scores.items():
            w, v = pair_tensors(sc, ge, extra)
            V = v            # V is identical for every score (it depends only on gene effect)
            cg[name], den = concordance(w, v)
            if name in arms:
                W[name] = w  # kept only for the arms, which are the ones bootstrapped over lines
            else:
                del w
        n_pairs = (den / 2).astype(int)
        genes_with_pairs = int((n_pairs > 0).sum())

        A["concordance"] = {name: macro(cg[name]) for name in cg}
        A["concordance_per_gene"] = {name: {genes[i]: (None if np.isnan(cg[name][i])
                                                       else float(cg[name][i]))
                                            for i in range(len(genes))} for name in cg}
        A["genes_with_at_least_one_pair"] = genes_with_pairs
        A["genes_without_any_pair"] = int(len(genes) - genes_with_pairs)
        if analysis == "primary_template_domain":
            A["template_domain_size_per_gene"] = {genes[i]: int(domain_size[i])
                                                  for i in range(len(genes))}
        nz = n_pairs[n_pairs > 0]
        A["pairs_per_gene"] = {  # summary over the genes that contribute at least one pair
            "total": int(n_pairs.sum()),
            "median": int(np.median(nz)) if nz.size else 0,
            "min": int(nz.min()) if nz.size else 0,
            "max": int(nz.max()) if nz.size else 0}

        # (b) Spearman against negative gene effect
        sp, const = {}, {}
        for name, sc in scores.items():
            rho = []
            n_const = 0
            for j in range(len(genes)):
                ok = ~np.isnan(ge[:, j])
                if extra is not None:
                    ok &= extra[:, j]
                x = sc[ok, j]
                y = -ge[ok, j]
                if len(x) == 0:
                    rho.append(np.nan)
                    continue
                flat = bool(np.all(x == x[0]))
                n_const += int(flat)
                if len(x) < 3 or flat:
                    rho.append(np.nan)
                    continue
                rho.append(stats.spearmanr(x, y).statistic)
            rho = np.array(rho, dtype=float)
            fin = np.isfinite(rho)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sp[name] = {"mean": float(np.nanmean(rho)), "median": float(np.nanmedian(rho)),
                            "q25": float(np.nanquantile(rho, .25)),
                            "q75": float(np.nanquantile(rho, .75)),
                            "n_genes_evaluated": int(fin.sum()),
                            "frac_positive": (float(np.mean(rho[fin] > 0))
                                              if fin.any() else float("nan"))}
            const[name] = n_const
            A.setdefault("spearman_per_gene", {})[name] = {
                genes[i]: (None if not np.isfinite(rho[i]) else float(rho[i]))
                for i in range(len(genes))}
        A["spearman_vs_negative_gene_effect"] = sp
        A["genes_with_constant_scores"] = const

        # (c) per-line AUROC against binary dependency
        lab = ge < C.DEP_THRESHOLD
        auc = {}
        for name, sc in scores.items():
            vals, undef = [], 0
            for i in range(len(cells)):
                ok = ~np.isnan(ge[i])
                if extra is not None:
                    ok &= extra[i]
                y = lab[i][ok]
                if y.sum() == 0 or y.sum() == y.size:
                    undef += 1
                    continue
                vals.append(roc_auc_score(y, sc[i][ok]))
            auc[name] = {"mean": float(np.mean(vals)) if vals else float("nan"),
                         "median": float(np.median(vals)) if vals else float("nan"),
                         "q25": float(np.quantile(vals, .25)) if vals else float("nan"),
                         "q75": float(np.quantile(vals, .75)) if vals else float("nan"),
                         "n_lines_evaluated": len(vals), "n_lines_undefined": undef}
        A["per_line_auroc_dependency_lt_%.1f" % C.DEP_THRESHOLD] = auc

        # (d) donor effects on identical gene-line pairs
        A["donor_effect"] = {a: macro(cg["own"] - cg[a]) for a in arms if a != "own"}
        A["donor_effect_vs_baselines"] = {b: macro(cg["own"] - cg[b])
                                          for b in cg if b.startswith("baseline_")}

        # (f) bootstrap over held-out lines grouped by PatientID
        if args.boot:
            groups = {}
            for i, c in enumerate(cells):
                groups.setdefault(patient[c], []).append(i)
            gkeys = sorted(groups)
            rng = np.random.default_rng(args.seed)
            boot = {name: np.empty(args.boot) for name in arms}
            bdiff = {a: np.empty(args.boot) for a in arms if a != "own"}
            t0 = time.time()
            for b in range(args.boot):
                pick = rng.integers(len(gkeys), size=len(gkeys))
                n = np.zeros(len(cells), dtype=np.float32)
                for k in pick:
                    for i in groups[gkeys[k]]:
                        n[i] += 1
                cb = {}
                for name in arms:
                    c_, _ = concordance(W[name], V, n)
                    cb[name] = macro(c_)
                    boot[name][b] = cb[name]
                for a in bdiff:
                    bdiff[a][b] = cb["own"] - cb[a]
            log(f"  line bootstrap {args.boot} reps in {time.time() - t0:.0f}s")
            A["bootstrap_lines"] = {
                "replicates": args.boot, "n_patient_groups": len(gkeys),
                "concordance_ci": {k: [float(np.nanpercentile(v, 2.5)),
                                       float(np.nanpercentile(v, 97.5))] for k, v in boot.items()},
                "donor_effect_ci": {k: [float(np.nanpercentile(v, 2.5)),
                                        float(np.nanpercentile(v, 97.5))] for k, v in bdiff.items()},
            }

            # bootstrap over gene families
            fam = pd.Series(list(panel.family), index=range(len(genes)))
            idx_by_fam = fam.groupby(fam).indices
            fam_index = [np.asarray(idx_by_fam[k]) for k in sorted(idx_by_fam)]
            rng = np.random.default_rng(args.seed + 1)
            fboot = {name: np.empty(args.boot) for name in cg}
            fdiff = {a: np.empty(args.boot) for a in arms if a != "own"}
            for b in range(args.boot):
                pick = rng.integers(len(fam_index), size=len(fam_index))
                idx = np.concatenate([fam_index[k] for k in pick])
                for name in cg:
                    fboot[name][b] = macro(cg[name][idx])
                for a in fdiff:
                    fdiff[a][b] = macro((cg["own"] - cg[a])[idx])
            A["bootstrap_families"] = {
                "replicates": args.boot, "n_families": len(fam_index),
                "concordance_ci": {k: [float(np.nanpercentile(v, 2.5)),
                                       float(np.nanpercentile(v, 97.5))] for k, v in fboot.items()},
                "donor_effect_ci": {k: [float(np.nanpercentile(v, 2.5)),
                                        float(np.nanpercentile(v, 97.5))] for k, v in fdiff.items()},
            }

            # (g) verdict against the smallest effect of interest
            verdict = {}
            for a in bdiff:
                lo, hi = A["bootstrap_lines"]["donor_effect_ci"][a]
                flo, fhi = A["bootstrap_families"]["donor_effect_ci"][a]
                verdict[a] = {
                    "donor_effect": A["donor_effect"][a],
                    "ci_lines": [lo, hi], "ci_families": [flo, fhi],
                    "label_lines": label_effect(lo, hi),
                    "label_families": label_effect(flo, fhi),
                }
            A["verdicts_vs_sesi_%.2f" % C.SESI] = verdict
        res["analyses"][analysis] = A
        del W, V

    jsonutil.dump(res, args.out)
    log("wrote " + args.out)
    prim = res["analyses"]["primary_template_domain"]
    print(json.dumps({"analysis": "primary_template_domain",
                      "concordance": prim["concordance"],
                      "donor_effect": prim["donor_effect"],
                      "verdicts": prim.get("verdicts_vs_sesi_0.01", {})}, indent=1))


def label_effect(lo, hi, sesi=C.SESI):
    """Where the 95% interval of (C_own - C_arm) falls relative to 0 and the SESI."""
    if hi < 0:
        return "reversed"
    if hi < sesi:
        return f"no sample-specific advantage larger than {sesi}"
    if lo > sesi:
        return "advantage"
    return "inconclusive"


if __name__ == "__main__":
    main()
