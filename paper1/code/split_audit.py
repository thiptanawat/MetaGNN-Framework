#!/usr/bin/env python3
"""Audit of the reaction split: how many held-out reactions have a near-duplicate in training.

The reaction folds are drawn at random, stratified on the label crossed with expression
availability, and never grouped. A reference network carries many reactions that differ
only in compartment, or that share a stoichiometry or a gene rule with another reaction,
so a held-out reaction can have a training reaction that is, for the purposes of a
per-reaction feature model, the same reaction. This script counts them, fold by fold, on
the byte-identical fold construction of double_holdout.py and naive_baselines_allfolds.py.

For every reaction fold and every held-out reaction of that fold, four criteria are
tested against the training reactions of the same fold:

  stoich        an identical stoichiometric signature: the same metabolites, in the same
                compartments, with the same coefficients
  stoich_comp   the same stoichiometry once the compartment suffix of every metabolite is
                removed, so that the same conversion carried out in two compartments, or
                the transport of one metabolite between two different pairs of
                compartments, counts as a duplicate
  gpr           an identical, nonempty gene rule, taken from the project's gene table (the
                table the expression mapping uses) in a canonical, order-free form
  any           at least one of the three

Each criterion reports the count and the share of held-out reactions matched, the share of
matched held-out reactions that are active under the label, the same share among the
unmatched ones, and the label agreement between a matched held-out reaction and its
training partners. Three further counts are recorded for reference: matches on the raw
BiGG gene rule string of the aligned reactions, matches on the exact stoichiometry with the
direction reversed, and matches on the metabolite set alone (the same metabolites in the
same compartments, whatever the coefficients).

Inputs: the aligned reference network (P1_ALIGNED, or data/recon3d_aligned.json.gz in the
repository), and the label and expression-availability vectors that define the folds. When
the deposited cohort is present (P1_DATA), both come from it exactly as in
naive_baselines_allfolds.py; otherwise they are read from paper2/rxn_context.npz, which
carries the same label vector and the same availability mask, and the file records which
source was used together with the checks that tie the two together.

Output: results/split_audit.json (or P1_OUT), which build_results_p1.py folds into
data/results_p1.json.
"""
import os, re, sys, json, gzip, argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
N = 10600; SEED = 2024

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.environ.get("P1_OUT", os.path.join(ROOT, "results", "split_audit.json")))
ap.add_argument("--matched_out", default=None,
                help="also write the held-out reaction indices matched under 'any', per reaction fold, "
                     "so a rescoring can drop them")
ap.add_argument("--family_map", default=None,
                help="a family map from build_families.py; audit the family-disjoint folds instead of the random ones")
A = ap.parse_args()

# ---- the aligned reference network, canonical order ---------------------------------------
def _first(*cands):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None

aligned = _first(os.path.expanduser(os.environ.get("P1_ALIGNED", "")),
                 os.path.join(ROOT, "data", "recon3d_aligned.json.gz"),
                 os.path.join(ROOT, "data", "recon3d_aligned.json"),
                 os.path.expanduser("~/metagnn/out/recon3d_aligned.json"))
if aligned is None:
    raise SystemExit("recon3d_aligned.json(.gz) not found; set P1_ALIGNED")
fh = gzip.open(aligned, "rt") if aligned.endswith(".gz") else open(aligned)
AL = json.load(fh)["reactions"]; assert len(AL) == N

# ---- the label and the availability mask that define the folds --------------------------
D = os.path.expanduser(os.environ.get("P1_DATA", "~/metagnn/work/crc_624"))
source = None
if os.path.exists(os.path.join(D, "activity_pseudolabels.pt")) and os.path.isdir(os.path.join(D, "reaction_features")):
    # identical to naive_baselines_allfolds.py: the label vector, and the availability mask read from
    # the first twenty training patients of patient fold 0
    import torch, h5py, pandas as pd
    from sklearn.model_selection import train_test_split
    y = np.asarray(torch.load(D + "/activity_pseudolabels.pt", map_location="cpu", weights_only=False)).astype(int)
    meta = pd.read_csv(D + "/clinical_metadata.tsv", sep="\t")
    ids = meta["tcga_barcode"].tolist(); proj = meta["project"].fillna("U").tolist()
    pk = StratifiedKFold(5, shuffle=True, random_state=SEED); pfolds = []
    for tr, te in pk.split(ids, proj):
        tr_ids = [ids[i] for i in tr]
        trn, val = train_test_split(tr_ids, test_size=0.15, random_state=SEED)
        pfolds.append(dict(train=trn, val=val, test=[ids[i] for i in te]))
    def load(ps):
        return np.stack([h5py.File(f"{D}/reaction_features/{p}.h5")["X_R"][:, 0] for p in ps])
    has = load(pfolds[0]["train"][:20]).max(0) > 0
    source = dict(labels=D + "/activity_pseudolabels.pt", availability="first 20 training patients of patient fold 0")
else:
    ctx = _first(os.path.join(os.path.dirname(ROOT), "paper2", "rxn_context.npz"))
    if ctx is None:
        raise SystemExit("neither the deposited cohort (P1_DATA) nor paper2/rxn_context.npz is present")
    z = np.load(ctx, allow_pickle=True)
    y = z["labels"].astype(int); has = z["has"].astype(bool)
    _ctx = os.path.relpath(ctx, os.path.dirname(ROOT)) if os.path.abspath(ctx).startswith(os.path.dirname(ROOT)) else ctx
    source = dict(labels=_ctx, availability=_ctx + " [has]")
assert len(y) == N and len(has) == N

# ---- reaction folds --------------------------------------------------------------------------
# Without --family_map these are the random folds of double_holdout.py, byte for byte. With it they
# are the family-disjoint folds the training runs used, so the identical audit prices the identical
# split and the two runs are comparable line for line.
if A.family_map:
    sys.path.insert(0, HERE)
    from family_folds import load_family_map, family_rfolds
    _comp, _scheme = load_family_map(A.family_map)
    rfolds = [(np.asarray(tr), np.asarray(te)) for tr, te in family_rfolds(_comp, y, has, n_folds=3, seed=SEED)]
else:
    _scheme = None
    rk = StratifiedKFold(3, shuffle=True, random_state=SEED)
    rfolds = list(rk.split(np.zeros(N), 2 * y + has.astype(int)))

# ---- the signatures -------------------------------------------------------------------------
def strip_comp(m):
    return re.sub(r"_[a-z]$", "", m)

def sig_exact(r):
    return frozenset((m, float(c)) for m, c in r["metabolites"].items())

def sig_reversed(r):
    return frozenset((m, -float(c)) for m, c in r["metabolites"].items())

def sig_nocomp(r):
    # a multiset, so that a transport reaction (the same metabolite on both sides once the
    # compartment is dropped) keeps both entries
    return tuple(sorted((strip_comp(m), float(c)) for m, c in r["metabolites"].items()))

def sig_set(r):
    return frozenset(r["metabolites"])

def sig_gpr(r):
    gs = r.get("gene_sets") or []
    if not gs:
        return None
    return frozenset(frozenset(str(g) for g in s) for s in gs if s) or None

def sig_gpr_bigg(r):
    s = r.get("gene_reaction_rule")
    return s.strip() if isinstance(s, str) and s.strip() else None

SIGS = {
    "stoich": [sig_exact(r) for r in AL],
    "stoich_reversed": [sig_reversed(r) for r in AL],
    "stoich_comp": [sig_nocomp(r) for r in AL],
    "stoich_set": [sig_set(r) for r in AL],
    "gpr": [sig_gpr(r) for r in AL],
    "gpr_bigg": [sig_gpr_bigg(r) for r in AL],
}
MAIN = ("stoich", "stoich_comp", "gpr")
EXTRA = ("stoich_reversed", "stoich_set", "gpr_bigg")

def partners(crit, rtr, rte):
    """For every held-out reaction, the training reactions of the same fold sharing its signature."""
    key = SIGS["stoich"] if crit == "stoich_reversed" else SIGS[crit]
    idx = {}
    for j in rtr:
        k = key[j]
        if k is not None:
            idx.setdefault(k, []).append(int(j))
    out = {}
    for i in rte:
        k = SIGS[crit][i]
        if k is None:
            continue
        hits = idx.get(k)
        if hits:
            out[int(i)] = hits
    return out

def summarize(matched, rte, label):
    """Counts, shares and label statistics for one criterion on one fold."""
    n_held = len(rte); m = sorted(matched)
    active_matched = [int(label[i]) for i in m]
    unmatched = [int(i) for i in rte if int(i) not in matched]
    agree = [float(np.mean([label[j] == label[i] for j in matched[i]])) for i in m]
    rec = dict(n_matched=len(m), share_matched=(len(m) / n_held if n_held else None),
               n_matched_active=int(sum(active_matched)),
               active_share_matched=(float(np.mean(active_matched)) if m else None),
               active_share_unmatched=(float(np.mean([label[i] for i in unmatched])) if unmatched else None),
               label_agreement=(float(np.mean(agree)) if agree else None),
               label_agreement_majority_share=(float(np.mean([a > 0.5 for a in agree])) if agree else None),
               partners_per_matched=(float(np.mean([len(matched[i]) for i in m])) if m else None))
    return rec

R = dict(_generated_by="code/split_audit.py", seed=SEED, n_reactions=N, rfolds=len(rfolds),
         split=("family_disjoint" if A.family_map else "random"), family_scheme=_scheme,
         family_map=(os.path.relpath(A.family_map, ROOT) if A.family_map and A.family_map.startswith(ROOT) else A.family_map),
         source=source, aligned=os.path.relpath(aligned, ROOT) if aligned.startswith(ROOT) else aligned,
         checks=dict(n_active=int(y.sum()), n_expression_bearing=int(has.sum()),
                     heldout_sizes=[int(len(te)) for _, te in rfolds]),
         criteria=dict(stoich="identical metabolites, compartments and coefficients",
                       stoich_comp="identical metabolites and coefficients with the compartment suffix removed",
                       gpr="identical nonempty gene rule of the project's gene table, canonical form",
                       any="at least one of stoich, stoich_comp, gpr",
                       stoich_reversed="reference only: the exact stoichiometry with every coefficient negated",
                       stoich_set="reference only: the same metabolites in the same compartments, coefficients ignored",
                       gpr_bigg="reference only: identical nonempty raw gene rule string of the aligned BiGG record"),
         folds={}, all={})

tot = {c: {} for c in MAIN + ("any",) + EXTRA}
MATCHED = {}
prev_all = []
for rf, (rtr, rte) in enumerate(rfolds):
    fold = dict(n_train=int(len(rtr)), n_heldout=int(len(rte)),
                prevalence_heldout=float(y[rte].mean()), prevalence_train=float(y[rtr].mean()))
    per = {c: partners(c, rtr, rte) for c in MAIN + EXTRA}
    anym = {}
    for c in MAIN:
        for i, hits in per[c].items():
            anym.setdefault(i, set()).update(hits)
    per["any"] = {i: sorted(h) for i, h in anym.items()}
    for c in MAIN + ("any",) + EXTRA:
        fold[c] = summarize(per[c], rte, y)
        tot[c].update({i: per[c][i] for i in per[c]})
    # what the three criteria share and what each adds
    sets = {c: set(per[c]) for c in MAIN}
    fold["overlap"] = dict(stoich_and_stoich_comp=len(sets["stoich"] & sets["stoich_comp"]),
                           stoich_comp_only=len(sets["stoich_comp"] - sets["stoich"] - sets["gpr"]),
                           gpr_only=len(sets["gpr"] - sets["stoich"] - sets["stoich_comp"]),
                           stoich_only=len(sets["stoich"] - sets["gpr"]),
                           all_three=len(sets["stoich"] & sets["stoich_comp"] & sets["gpr"]))
    R["folds"][f"r{rf}"] = fold
    MATCHED[f"r{rf}"] = dict(heldout=[int(i) for i in rte], matched_any=sorted(int(i) for i in per["any"]))
    print(f"r{rf}: held out {len(rte)}, train {len(rtr)} | "
          + " | ".join(f"{c} {fold[c]['n_matched']} ({100*fold[c]['share_matched']:.1f}%)" for c in MAIN + ("any",)),
          flush=True)

# every reaction is held out in exactly one fold, so the union over folds covers the network once
all_te = np.concatenate([te for _, te in rfolds])
assert len(set(all_te.tolist())) == N
for c in MAIN + ("any",) + EXTRA:
    rec = summarize(tot[c], all_te, y)
    rec["share_by_fold"] = [R["folds"][f"r{rf}"][c]["share_matched"] for rf in range(len(rfolds))]
    rec["share_min"] = min(rec["share_by_fold"]); rec["share_max"] = max(rec["share_by_fold"])
    R["all"][c] = rec
R["all"]["n_heldout"] = int(len(all_te))
R["all"]["prevalence_heldout"] = float(y[all_te].mean())
R["all"]["heldout_per_fold"] = [int(len(te)) for _, te in rfolds]

os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
json.dump(R, open(A.out, "w"), indent=1)
a = R["all"]
print("\nall %d held-out reactions over %d folds (prevalence %.4f):" % (a["n_heldout"], len(rfolds), a["prevalence_heldout"]))
for c in MAIN + ("any",) + EXTRA:
    v = a[c]
    print("  %-16s matched %5d (%5.1f%%; folds %.1f-%.1f%%)  active among matched %.4f  among unmatched %s  label agreement %s"
          % (c, v["n_matched"], 100 * v["share_matched"], 100 * v["share_min"], 100 * v["share_max"],
             v["active_share_matched"] if v["active_share_matched"] is not None else float("nan"),
             ("%.4f" % v["active_share_unmatched"]) if v["active_share_unmatched"] is not None else "n/a",
             ("%.4f" % v["label_agreement"]) if v["label_agreement"] is not None else "n/a"))
print("wrote", os.path.relpath(A.out, ROOT) if os.path.abspath(A.out).startswith(ROOT) else A.out)
if A.matched_out:
    json.dump(dict(_generated_by="code/split_audit.py --matched_out", seed=SEED,
                   split=("family_disjoint" if A.family_map else "random"),
                   family_map=(os.path.relpath(A.family_map, ROOT) if A.family_map and A.family_map.startswith(ROOT) else A.family_map),
                   criteria="any of stoich, stoich_comp, gpr", folds=MATCHED),
              open(A.matched_out, "w"))
    print("wrote", A.matched_out)
