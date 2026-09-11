#!/usr/bin/env python3
"""Reaction folds that keep label-blind reaction families together.

The main grid draws its reaction folds at random, so near-duplicate reactions (compartment
variants, gene-rule twins) can sit on both sides of a fold boundary. build_families.py groups
the 10,600 reactions into families from stoichiometry and gene rules alone; this module turns a
family map into (a) three outer reaction folds in which every family lies entirely in one fold and
(b) an inner-validation partition of a cell's training reactions drawn family by family, so that
early stopping never reads a relative of a fitted reaction either.

Both draws balance the folds on the reaction count, the shared-label prevalence and expression
availability, the same three quantities the random folds are stratified on; the family map itself
never saw a label. Families with more than one member are placed greedily, largest first, into the
least loaded fold, load measured in units of each stratum's target; singletons are then dealt round-robin
within each (label, availability) stratum in a seeded random order, which is what balances the
strata exactly. Every quantity here is a deterministic function of the seed.
"""
import json
import numpy as np

def load_family_map(path):
    d = json.load(open(path))
    return np.asarray(d["component_id"], dtype=np.int64), d.get("scheme", "?")

def _strata(y, has):
    return 2 * np.asarray(y).astype(int) + np.asarray(has).astype(int)

def family_rfolds(comp, y, has, n_folds=3, seed=2024):
    """List of (train_idx, test_idx) over reactions; each family is inside one test fold."""
    comp = np.asarray(comp); N = len(comp); st = _strata(y, has)
    rng = np.random.default_rng(seed)
    fam_members = {}
    for i, c in enumerate(comp): fam_members.setdefault(int(c), []).append(i)
    fold_of = np.full(N, -1, dtype=int)
    n_strata = int(st.max()) + 1
    target = np.array([np.sum(st == s) / n_folds for s in range(n_strata)])
    totals = np.zeros((n_folds, n_strata))
    multi = sorted([m for m in fam_members.values() if len(m) > 1], key=lambda m: (-len(m), m[0]))
    for m in multi:
        counts = np.bincount(st[m], minlength=n_strata)
        # largest family first into the least loaded fold, load measured in units of each stratum's target
        scores = [np.sum((totals[f] + counts) / np.maximum(target, 1.0)) for f in range(n_folds)]
        f = int(np.argmin(scores)); totals[f] += counts; fold_of[m] = f
    singles = np.array([m[0] for m in fam_members.values() if len(m) == 1], dtype=int)
    for s in range(n_strata):
        idx = singles[st[singles] == s]; idx = idx[rng.permutation(len(idx))]
        for i in idx:
            f = int(np.argmin(totals[:, s] - target[s])); totals[f, s] += 1; fold_of[i] = f
    assert (fold_of >= 0).all()
    folds = []
    for f in range(n_folds):
        te = np.flatnonzero(fold_of == f); tr = np.flatnonzero(fold_of != f)
        folds.append((tr, te))
    return folds

def family_inner_val(comp, rtr, y, has, frac=0.2, seed=2024):
    """Boolean mask over all reactions: the inner-validation reactions of one cell, whole families
    of the training reactions rtr, close to frac of each (label, availability) stratum of rtr."""
    comp = np.asarray(comp); st = _strata(y, has); N = len(comp)
    rng = np.random.default_rng(seed)
    tr_set = np.zeros(N, dtype=bool); tr_set[rtr] = True
    n_strata = int(st.max()) + 1
    target = np.array([frac * np.sum(st[rtr] == s) for s in range(n_strata)])
    fam_members = {}
    for i in rtr: fam_members.setdefault(int(comp[i]), []).append(i)
    fams = list(fam_members.values()); order = rng.permutation(len(fams))
    got = np.zeros(n_strata); mask = np.zeros(N, dtype=bool)
    for k in order:
        m = fams[k]; counts = np.bincount(st[m], minlength=n_strata)
        if np.all(got + counts <= target * 1.05 + 1):
            mask[m] = True; got += counts
        if np.all(got >= target * 0.97): break
    return mask

def fold_report(comp, folds, y, has):
    """Per-fold sizes, prevalence and availability, and the family exposure of each held-out set:
    the share of held-out reactions with any family relative among the training reactions (zero by
    construction for the map the folds were built from)."""
    comp = np.asarray(comp); out = []
    for f, (tr, te) in enumerate(folds):
        tr_fams = set(comp[tr].tolist())
        exposed = int(np.sum([comp[i] in tr_fams for i in te]))
        out.append(dict(fold=f, n_train=int(len(tr)), n_test=int(len(te)), prevalence_test=float(np.mean(np.asarray(y)[te])),
                        exprbearing_test=float(np.mean(np.asarray(has)[te])), n_families_test=int(len(set(comp[te].tolist()))),
                        heldout_with_family_relative_in_train=exposed))
    return out

if __name__ == "__main__":
    import argparse, os, torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--family_map", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=2024); ap.add_argument("--out", default=None)
    A = ap.parse_args()
    comp, scheme = load_family_map(A.family_map)
    y = np.asarray(torch.load(os.path.join(A.data, "activity_pseudolabels.pt"), map_location="cpu", weights_only=False)).astype(int)
    # expression availability from the same probe the training scripts use: the first 20 training
    # patients of patient fold 0, read from the feature files directly
    import pandas as pd, h5py
    from sklearn.model_selection import StratifiedKFold, train_test_split
    meta = pd.read_csv(os.path.join(A.data, "clinical_metadata.tsv"), sep="\t"); ids = meta["tcga_barcode"].tolist(); proj = meta["project"].fillna("U").tolist()
    pk = StratifiedKFold(5, shuffle=True, random_state=A.seed); tr, te = next(pk.split(ids, proj))
    trn, val = train_test_split([ids[i] for i in tr], test_size=0.15, random_state=A.seed)
    cols = []
    for p in trn[:20]:
        with h5py.File(os.path.join(A.data, "reaction_features", p + ".h5")) as h: cols.append(np.asarray(h["X_R"][:, 0]))
    has = np.stack(cols).max(0) > 0
    folds = family_rfolds(comp, y, has, seed=A.seed)
    rep = fold_report(comp, folds, y, has)
    for r in rep: print(r)
    inner = [dict(fold=f, n_inner_val=int(family_inner_val(comp, tr, y, has, seed=A.seed + 500 + 0 * 10 + f).sum()), n_train=int(len(tr))) for f, (tr, te) in enumerate(folds)]
    for r in inner: print(r)
    if A.out:
        json.dump(dict(scheme=scheme, seed=A.seed, folds=rep, inner_val_p0=inner,
                       test_index=[te.tolist() for _, te in folds]), open(A.out, "w"))
