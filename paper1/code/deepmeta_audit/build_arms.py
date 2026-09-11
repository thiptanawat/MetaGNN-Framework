"""Step 3: build DeepMeta inputs for every held-out line x arm.

Every sample-dependent input is recomputed from the substituted expression profile, while the
RECIPIENT's lineage templates (tissue GSM enzyme network and GTEx normal tissue) stay fixed.
An arm is therefore a pair (expression profile used for the SEN, expression profile used for
the differential-expression vector):

  arm            SEN profile        diff-exp profile
  own            own                own
  mean           lineage mean       lineage mean
  within         within donor       within donor
  cross          cross donor        cross donor
  within_expr    own                within donor
  within_graph   within donor       own

so only four SEN sets and four diff-exp files exist; the six arms are views on them, keyed by
the RECIPIENT ModelID (file names are always the recipient's, never the donor's).

Restartable: an existing <cell>_feat.txt in a SEN directory is skipped, and diff-exp files are
rewritten only when a cell is missing from them.

  python3 build_arms.py --seed 11
  python3 build_arms.py --seed 11 --arms own within --cells ACH-000350 ACH-000935
  python3 build_arms.py --seed 11 --batch_size 20
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import config as C
import deepmeta_io as dio
from manifest import load_expression

# arm -> (SEN profile source, diff-exp profile source)
ARM_SOURCES = {
    "own": ("own", "own"),
    "mean": ("mean", "mean"),
    "within": ("within", "within"),
    "cross": ("cross", "cross"),
    "within_expr": ("own", "within"),
    "within_graph": ("within", "own"),
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def seed_dir(seed):
    return os.path.join(C.ARMS, f"seed{seed}")


def profile_table(source, cells, expr, man, sch):
    """Expression profile (log2(TPM+1) by symbol) supplied to each recipient under `source`."""
    lineage = {s["ModelID"]: s["OncotreeLineage"] for s in man["samples"]}
    dev = [s["ModelID"] for s in man["samples"] if s["development"] in (True, "True")]
    out, prov = {}, {}
    if source == "own":
        for c in cells:
            out[c] = expr.loc[c]
            prov[c] = {"donor": c, "kind": "own"}
    elif source == "mean":
        dev_by_lin = {}
        for m in dev:
            dev_by_lin.setdefault(lineage[m], []).append(m)
        wide = expr.loc[dev].mean(axis=0)
        for c in cells:
            info = sch["mean"][c]
            if info["source"] == "lineage_mean":
                members = sorted(dev_by_lin[lineage[c]])
                out[c] = expr.loc[members].mean(axis=0)
                prov[c] = {"donor": None, "kind": "lineage_mean", "lineage": lineage[c],
                           "n_development": len(members)}
            else:
                out[c] = wide
                prov[c] = {"donor": None, "kind": "development_wide_mean",
                           "lineage": lineage[c], "n_development": len(dev),
                           "flag": info.get("flag")}
    else:                       # "within" / "cross": the donor cell's own profile
        for c in cells:
            d = sch[source][c]
            out[c] = expr.loc[d]
            prov[c] = {"donor": d, "kind": source, "donor_lineage": lineage[d]}
    return out, prov


def build_sen(seed, source, cells, profiles, man, batch_size):
    """One SEN per recipient from the substituted profile, on the recipient's tissue template."""
    net_of = {s["ModelID"]: s["net_template"] for s in man["samples"]}
    sen_dir = os.path.join(seed_dir(seed), "sen", source)
    os.makedirs(sen_dir, exist_ok=True)
    todo = [c for c in cells
            if not os.path.exists(os.path.join(sen_dir, f"{c}_feat.txt"))
            or not os.path.lexists(os.path.join(sen_dir, f"{c}.txt"))]
    log(f"SEN[{source}]: {len(cells)} cells, {len(todo)} to build")
    summary = {}
    for i in range(0, len(todo), batch_size):
        t0 = time.time()
        for c in todo[i:i + batch_size]:
            summary[c] = dio.write_sen(profiles[c], net_of[c], c, sen_dir)
        log(f"  SEN[{source}] {min(i + batch_size, len(todo))}/{len(todo)}"
            f" ({time.time() - t0:.0f}s/batch)")
    return sen_dir, summary


def build_diff(seed, source, cells, profiles, man):
    normal_of = {s["ModelID"]: s["gtex_normal"] for s in man["samples"]}
    d = os.path.join(seed_dir(seed), "exp")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"diff_exp_{source}.csv")
    have = set()
    old = None
    if os.path.exists(path):
        old = pd.read_csv(path)
        have = set(old.cell)
    todo = [c for c in cells if c not in have]
    if todo:
        new = dio.diff_exp_rows({c: profiles[c] for c in todo},
                                {c: normal_of[c] for c in todo})
        df = pd.concat([old, new], ignore_index=True) if old is not None else new
        df.to_csv(path, index=False)
        n_nan = int(np.isnan(df.iloc[:, 1:].to_numpy(dtype=float)).sum())
        n_inf = int(np.isinf(df.iloc[:, 1:].to_numpy(dtype=float)).sum())
        log(f"diff_exp[{source}]: +{len(todo)} rows -> {len(df)} (nan {n_nan}, inf {n_inf})")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=C.PRIMARY_SEED)
    ap.add_argument("--arms", nargs="*", default=C.ARM_NAMES)
    ap.add_argument("--cells", nargs="*", default=None, help="restrict to these ModelIDs")
    ap.add_argument("--limit", type=int, default=None, help="first N held-out lines (sorted)")
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--tag", default="", help="suffix for the arm spec / cell-list files, so a "
                                              "subset run does not overwrite the full one")
    ap.add_argument("--manifest", default=C.MANIFEST_JSON)
    ap.add_argument("--schedules", default=C.SCHEDULES_JSON)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    sch = json.load(open(args.schedules))["schedules"][str(args.seed)]
    heldout = sorted(s["ModelID"] for s in man["samples"] if s["heldout"] in (True, "True"))
    cells = args.cells if args.cells else heldout
    if args.limit:
        cells = cells[:args.limit]
    cells = [c for c in cells if c in set(heldout)]
    for a in args.arms:
        assert a in ARM_SOURCES, a
    sources = sorted({s for a in args.arms for s in ARM_SOURCES[a]})
    log(f"seed {args.seed}: {len(cells)} recipients, arms {args.arms}, sources {sources}")

    # donors must be in the expression cache too
    need = set(cells)
    for src in sources:
        if src in ("within", "cross"):
            need |= {sch[src][c] for c in cells}
    if "mean" in sources:
        need |= {s["ModelID"] for s in man["samples"] if s["development"] in (True, "True")}
    expr = load_expression(sorted(need))

    C.ensure_dirs(seed_dir(args.seed))
    provs, sen_dirs, exp_files = {}, {}, {}
    for src in sources:
        profiles, prov = profile_table(src, cells, expr, man, sch)
        provs[src] = prov
        sen_dirs[src], summary = build_sen(args.seed, src, cells, profiles, man, args.batch_size)
        exp_files[src] = build_diff(args.seed, src, cells, profiles, man)
        sp = os.path.join(seed_dir(args.seed), f"sen_summary_{src}.json")
        old = json.load(open(sp)) if os.path.exists(sp) else {}
        old.update(summary)
        json.dump(old, open(sp, "w"), indent=1)

    for arm in args.arms:
        sen_src, exp_src = ARM_SOURCES[arm]
        cl = pd.DataFrame({"cell": cells, "cell_index": range(len(cells))})
        cl_path = os.path.join(seed_dir(args.seed), f"cells_{arm}{args.tag}.csv")
        cl.to_csv(cl_path, index=False)
        json.dump({"arm": arm, "seed": args.seed,
                   "sen_source": sen_src, "exp_source": exp_src,
                   "sen_dir": sen_dirs[sen_src], "exp_file": exp_files[exp_src],
                   "cells_csv": cl_path, "n_cells": len(cells),
                   "donors": {c: provs[sen_src][c].get("donor") for c in cells},
                   "exp_donors": {c: provs[exp_src][c].get("donor") for c in cells},
                   "provenance_sen": provs[sen_src], "provenance_exp": provs[exp_src]},
                  open(os.path.join(seed_dir(args.seed), f"arm_{arm}{args.tag}.json"), "w"),
                  indent=1)
        log(f"arm {arm}: SEN {sen_src}, exp {exp_src}, {len(cells)} cells")


if __name__ == "__main__":
    main()
