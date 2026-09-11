#!/usr/bin/env python3
"""Gene-to-reaction projection through gene-protein-reaction rules, and its verification.

Reaction-level expression is the usual aggregation: the minimum across the genes of one enzyme
complex (an AND group) and the maximum across alternative complexes or isozymes (OR groups). The
gene sets are the project's own gene table (paper1/code/build_aligned_recon3d.py): a list of AND
groups per reaction, each a list of Entrez gene identifiers. A group whose genes are not all
measured is unresolvable and is skipped; a reaction with no resolvable group receives no value.

Run as a script on the GPU host to verify the projection against the released reaction-level
matrix: the Entrez-to-Ensembl mapping of the cohort build is not shipped, so the script recovers it
from the single-gene reactions (a reaction whose only group is one gene equals that gene's column
across all patients) and then checks every multi-gene reaction whose genes were all recovered.
"""
import numpy as np

def project(gene_sets, values, missing=None):
    """values: dict gene_id -> float (one sample). Returns the reaction value or `missing`."""
    best = None
    for grp in gene_sets:
        vs = [values.get(g) for g in grp]
        if any(v is None for v in vs): continue
        m = min(vs)
        best = m if best is None else max(best, m)
    return missing if best is None else best

def project_matrix(gene_sets_per_rxn, G, gene_index, fill=0.0):
    """G: samples x genes array; gene_index: gene_id -> column. Returns samples x reactions and a
    coverage mask (a reaction is covered when at least one group is fully measured)."""
    n = G.shape[0]; R = len(gene_sets_per_rxn)
    X = np.full((n, R), fill, dtype=np.float32); cov = np.zeros(R, dtype=bool)
    for j, gs in enumerate(gene_sets_per_rxn):
        best = None
        for grp in gs:
            cols = [gene_index.get(g) for g in grp]
            if any(c is None for c in cols): continue
            m = G[:, cols].min(axis=1)
            best = m if best is None else np.maximum(best, m)
        if best is not None: X[:, j] = best; cov[j] = True
    return X, cov

if __name__ == "__main__":
    import os, json, gzip, argparse, h5py
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/metagnn/work/crc_624"))
    ap.add_argument("--aligned", default=os.path.expanduser("~/metagnn/out/recon3d_aligned.json"))
    ap.add_argument("--out", default=os.path.expanduser("~/metagnn/out/gpr_verify.json"))
    A = ap.parse_args()
    AL = (json.load(gzip.open(A.aligned)) if A.aligned.endswith(".gz") else json.load(open(A.aligned)))["reactions"]
    with h5py.File(os.path.join(A.data, "tcga_crc_rnaseq.h5")) as h:
        E = h["vst_expression"][:].astype(np.float64); gids = [g.decode().split(".")[0] for g in h["gene_ids"][:]]; pids = [p.decode() for p in h["patient_ids"][:]]
    XR = np.stack([h5py.File(os.path.join(A.data, "reaction_features", p + ".h5"))["X_R"][:, 0] for p in pids]).astype(np.float64)
    # recover the Entrez -> Ensembl column mapping from single-gene reactions
    ens_by_col = {}
    from collections import defaultdict
    hashes = defaultdict(list)
    for k in range(E.shape[0]): hashes[np.round(E[k], 4).tobytes()].append(k)
    entrez2col = {}; conflicts = 0
    for j, r in enumerate(AL):
        gs = r["gene_sets"]
        if len(gs) == 1 and len(gs[0]) == 1 and XR[:, j].max() > 0:
            cands = hashes.get(np.round(XR[:, j], 4).tobytes(), [])
            if len(cands) == 1:
                g = gs[0][0]
                if g in entrez2col and entrez2col[g] != cands[0]: conflicts += 1
                entrez2col[g] = cands[0]
    gene_index = {g: c for g, c in entrez2col.items()}
    Gm = E.T  # samples x genes
    X2, cov = project_matrix([r["gene_sets"] for r in AL], Gm, gene_index)
    multi = [j for j, r in enumerate(AL) if r["gene_sets"] and (len(r["gene_sets"]) > 1 or len(r["gene_sets"][0]) > 1)
             and all(g in gene_index for grp in r["gene_sets"] for g in grp)]
    err = np.abs(X2[:, multi] - XR[:, multi]).max(axis=0) if multi else np.array([])
    res = dict(n_genes_mapped=len(entrez2col), conflicts=conflicts, n_multi_gene_reactions_checked=len(multi),
               max_abs_error=float(err.max()) if len(err) else None, n_reactions_exact=int((err < 1e-4).sum()) if len(err) else None,
               n_reactions_off=int((err >= 1e-4).sum()) if len(err) else None,
               zero_share_in_matrix=float((E == 0).mean()), matrix_min=float(E.min()), matrix_max=float(E.max()), matrix_median=float(np.median(E)))
    json.dump(res, open(A.out, "w"), indent=1); print(json.dumps(res, indent=1))
