"""Step 5: reproduce the authors' own benchmark before auditing anything.

Runs the own-input pipeline (24Q4 expression, recipient lineage templates) on the authors' 76
test lines, then

  * scores our predictions against the authors' labels in DeepMeta/data/test_dtV2.csv
    (AUROC on preds_raw, F1 at the published 0.5 cutoff),
  * scores the authors' own predictions (DeepMeta/data/test_preV2.csv) the same way, and
  * compares the two prediction sets on shared (cell, node) pairs by Pearson r and by
    agreement of the 0.5-thresholded calls.

A difference is expected and is itself a finding: the checkpoint was trained on an earlier
DepMap release, while every input here is rebuilt from 24Q4.

  python3 native_repro.py --batch_size 4
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import config as C
import jsonutil
import deepmeta_io as dio
from manifest import load_expression
from run_arms import run_cells


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=C.NATIVE_JSON)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()

    work = args.dir or os.path.join(C.OUT, "native")
    sen_dir = os.path.join(work, "sen")
    C.ensure_dirs(work, sen_dir)

    test_cells = list(pd.read_csv(C.DEEPMETA_FILES["test_cell_info.csv"]).cell)
    model = pd.read_csv(C.DEPMAP_FILES["Model.csv"], dtype=str).set_index("ModelID")
    res = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "authors_test_lines": len(test_cells), "excluded": {}}

    runnable = []
    for c in test_cells:
        if c not in model.index:
            res["excluded"][c] = "not in 24Q4 Model.csv"
            continue
        lin = model.loc[c, "OncotreeLineage"]
        if lin not in C.LINEAGE_MAP:
            res["excluded"][c] = f"lineage {lin!r} has no DeepMeta tissue template"
            continue
        runnable.append(c)
    if args.limit:
        runnable = runnable[:args.limit]
    expr = load_expression(runnable)
    res["runnable_lines"] = len(runnable)
    log(f"{len(runnable)} of {len(test_cells)} authors' test lines runnable on 24Q4; "
        f"excluded {res['excluded']}")

    todo = [c for c in runnable if not os.path.exists(os.path.join(sen_dir, f"{c}_feat.txt"))]
    for i, c in enumerate(todo):
        dio.write_sen(expr.loc[c], C.LINEAGE_MAP[model.loc[c, "OncotreeLineage"]][0], c, sen_dir)
        if (i + 1) % 10 == 0:
            log(f"  SEN {i + 1}/{len(todo)}")
    exp_file = os.path.join(work, "diff_exp_own.csv")
    have = set(pd.read_csv(exp_file).cell) if os.path.exists(exp_file) else set()
    miss = [c for c in runnable if c not in have]
    if miss:
        new = dio.diff_exp_rows(
            {c: expr.loc[c] for c in miss},
            {c: C.LINEAGE_MAP[model.loc[c, "OncotreeLineage"]][1] for c in miss})
        if have:
            new = pd.concat([pd.read_csv(exp_file), new], ignore_index=True)
        new.to_csv(exp_file, index=False)

    out_csv = os.path.join(work, "preds_native.csv")
    run_cells(runnable, sen_dir, exp_file, out_csv, args.batch_size, args.cores,
              args.device, C.CKPT, os.path.join(C.WORK, "native"), "native")

    ours = pd.read_csv(out_csv).rename(columns={"gene_name": "id"})
    lab = pd.read_csv(C.DEEPMETA_FILES["test_dtV2.csv"])          # id, is_dep, cell
    theirs = pd.read_csv(C.DEEPMETA_FILES["test_preV2.csv"]).rename(
        columns={"gene_name": "id", "preds_raw": "preds_raw_authors",
                 "preds": "preds_authors", "label": "label_authors"})

    m = ours.merge(lab, on=["cell", "id"], how="inner")
    res["ours_vs_authors_labels"] = {
        "n_pairs": len(m), "n_cells": int(m.cell.nunique()), "n_nodes": int(m.id.nunique()),
        "positive_rate": float(m.is_dep.mean()),
        "auroc": float(roc_auc_score(m.is_dep, m.preds_raw)),
        "f1_at_0.5": float(f1_score(m.is_dep, (m.preds_raw >= 0.5).astype(int))),
        "our_rows_without_label": int(len(ours) - len(m)),
        "label_rows_without_prediction": int(len(lab[lab.cell.isin(set(ours.cell))]) - len(m)),
    }
    t = theirs[theirs.cell.isin(set(ours.cell))]
    res["authors_vs_authors_labels"] = {
        "n_pairs": len(t), "positive_rate": float(t.label_authors.mean()),
        "auroc": float(roc_auc_score(t.label_authors, t.preds_raw_authors)),
        "f1_at_0.5": float(f1_score(t.label_authors,
                                    (t.preds_raw_authors >= 0.5).astype(int))),
    }
    j = ours.merge(theirs[["cell", "id", "preds_raw_authors", "preds_authors"]],
                   on=["cell", "id"], how="inner")
    ours_bin = (j.preds_raw >= 0.5).astype(int)
    res["ours_vs_authors_predictions"] = {
        "n_shared_pairs": len(j),
        "pearson_r": float(np.corrcoef(j.preds_raw, j.preds_raw_authors)[0, 1]),
        "spearman_rho": float(pd.Series(j.preds_raw).corr(
            pd.Series(j.preds_raw_authors), method="spearman")),
        "binary_agreement_at_0.5": float((ours_bin == j.preds_authors.astype(int)).mean()),
        "our_positive_rate": float(ours_bin.mean()),
        "authors_positive_rate": float(j.preds_authors.astype(int).mean()),
        "mean_abs_difference": float((j.preds_raw - j.preds_raw_authors).abs().mean()),
        "our_pairs_not_in_authors": int(len(ours) - len(j)),
        "authors_pairs_not_in_ours": int(len(t) - len(j)),
    }
    jsonutil.dump(res, args.out)
    log("wrote " + args.out)
    print(json.dumps({k: v for k, v in res.items() if k != "excluded"}, indent=1))


if __name__ == "__main__":
    main()
