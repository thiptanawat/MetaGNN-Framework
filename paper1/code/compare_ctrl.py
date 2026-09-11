#!/usr/bin/env python3
"""Match each control cell to its real-feature counterpart in the main grid.

Only gnn_B lambda=0.0 at patient fold 0 is comparable: the controls were run on
that model and fold, so mlp and mlp_emb cells must be excluded explicitly.
"""
import json, glob, os, statistics as st
def load(d, mode_from_name):
    out = {}
    for f in glob.glob(os.path.expanduser(d) + "/*.json"):
        r = json.load(open(f))
        if r["model"] != "gnn_B" or r["lambda_mb"] != 0.0 or r["pfold"] != 0: continue
        stem = os.path.basename(f)[:-5]
        mode = stem.rsplit("_", 1)[-1] if stem.rsplit("_", 1)[-1] in ("zero", "mean") else "real"
        out[(mode, r["rfold"])] = r
    return out
A = {**load("~/metagnn/out/dh", True), **load("~/metagnn/out/ctrl", True)}
print("%-6s %-7s %10s %10s %8s %9s %9s" % ("mode", "cell", "TRAINrxn", "HELDOUT", "rho", "interpat", "minutes"))
print("-" * 66)
rows = {}
for mode in ("real", "zero", "mean"):
    v = []
    for (m, rf), r in sorted(A.items()):
        if m != mode: continue
        print("%-6s p0r%-4d %10.4f %10.4f %8.3f %9.4f %9.1f" % (
            mode, rf, r["trainrxn_AUROC_perpatient"], r["heldout_AUROC_perpatient"],
            r["rho"], r["interpatient_r_median"], r["minutes"]))
        v.append(r["heldout_AUROC_perpatient"])
    if v:
        rows[mode] = v
        print("%-6s %-7s %10s %10.4f   mean of %d cells\n" % ("", "MEAN", "", st.mean(v), len(v)))
    else:
        print("%-6s (no cells yet)\n" % mode)
print("=" * 66)
print("held-out AUROC, gnn_B lambda=0, patient fold 0, matched reaction folds")
print("-" * 66)
for k, v in (("information-free indicator", 0.6085), ("raw expression, per patient", 0.6342),
             ("structure-only logistic regression", 0.7374)):
    print("  %-38s %.4f" % (k, v))
for mode, label in (("zero", "GNN, expression zeroed"), ("mean", "GNN, cohort-mean expression"),
                    ("real", "GNN, real per-patient expression")):
    if mode in rows: print("  %-38s %.4f" % (label, st.mean(rows[mode])))
print("-" * 66)
if "real" in rows and "zero" in rows:
    print("  expression contributes      %+.4f   (real - zeroed)" % (st.mean(rows["real"]) - st.mean(rows["zero"])))
    print("  GNN topology over logistic  %+.4f   (zeroed - 0.7374)" % (st.mean(rows["zero"]) - 0.7374))
if "real" in rows and "mean" in rows:
    print("  the PATIENT contributes     %+.4f   (real - cohort-mean)" % (st.mean(rows["real"]) - st.mean(rows["mean"])))
else:
    print("  the PATIENT contributes     pending (mean-mode cells still running)")
