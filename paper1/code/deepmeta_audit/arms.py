"""Step 2: donor schedules for the held-out lines.

Three schedules, numpy seeds 11 / 22 / 33 (11 is primary):

  within : uniform derangement of the held-out lines WITHIN each OncotreeLineage, drawn under the
           constraint that the donor's PatientID differs from the recipient's, so a line whose donor
           also contributed another held-out line is never given its own donor back.
           A lineage holding a single held-out line cannot be deranged inside itself; that
           line receives a donor drawn uniformly from the held-out lines of other lineages
           and is flagged (cross_lineage_fallback).
  cross  : uniform derangement of all held-out lines in which every donor's lineage differs
           from the recipient's. Sampled by rejection from uniform permutations, which is
           exactly uniform on the target set.
  mean   : the recipient is given the mean expression profile of the DEVELOPMENT lines of its
           own lineage; a lineage with fewer than MIN_DEV_FOR_LINEAGE_MEAN development lines
           falls back to the development-wide mean and is flagged.

Writes schedules.json.  No expression is read here -- only the assignment.
"""
import argparse
import json
import time

import numpy as np

import config as C
import jsonutil


def uniform_derangement(rng, items, max_tries=1_000_000, groups=None):
    """Uniform over derangements of `items` by rejection from uniform permutations.

    With `groups` (one label per item, here the PatientID) the rejected set widens from permutations
    with a fixed point to permutations in which any item draws a donor carrying its own label, so two
    cell lines from the same person are never each other's donors. Rejection from uniform
    permutations is uniform on whatever set survives, so the draw stays uniform over the
    donor-distinct derangements.
    """
    n = len(items)
    if n < 2:
        raise ValueError("a derangement needs at least 2 items")
    idx = np.arange(n)
    if groups is None:
        key = idx
    else:
        if len(set(groups)) < 2:
            raise ValueError("every item shares one group label; no donor-distinct derangement exists")
        lab = {g: i for i, g in enumerate(sorted(set(groups)))}
        key = np.array([lab[g] for g in groups])
    for t in range(max_tries):
        p = rng.permutation(n)
        if not (key[p] == key).any():
            return {items[i]: items[p[i]] for i in range(n)}, t + 1
    raise RuntimeError("no derangement found")


def uniform_cross_lineage_permutation(rng, cells, lineages, max_tries=20_000_000,
                                      report_every=1_000_000):
    """Uniform over permutations with donor lineage != recipient lineage (rejection)."""
    codes = np.array([lineages[c] for c in cells])
    uniq = {l: i for i, l in enumerate(sorted(set(codes)))}
    code = np.array([uniq[l] for l in codes])
    n = len(cells)
    for t in range(max_tries):
        p = rng.permutation(n)
        if not (code[p] == code).any():
            return {cells[i]: cells[p[i]] for i in range(n)}, t + 1
        if report_every and (t + 1) % report_every == 0:
            print(f"  ... {t + 1} rejection draws", flush=True)
    raise RuntimeError("no cross-lineage permutation found; relax max_tries")


def build_schedule(seed, heldout, lineage, patient, dev_by_lineage, n_dev_total):
    rng = np.random.default_rng(seed)
    lineages = sorted({lineage[c] for c in heldout})

    within, flags_within, tries_within = {}, {}, {}
    for l in lineages:
        cells = sorted(c for c in heldout if lineage[c] == l)
        if len(cells) >= 2:
            d, tries = uniform_derangement(rng, cells, groups=[patient[c] for c in cells])
            within.update(d)
            tries_within[l] = tries
        else:
            others = sorted(c for c in heldout if lineage[c] != l and patient[c] != patient[cells[0]])
            donor = others[int(rng.integers(len(others)))]
            within[cells[0]] = donor
            flags_within[cells[0]] = {
                "flag": "cross_lineage_fallback",
                "reason": "only one held-out line in this lineage",
                "lineage": l, "donor_lineage": lineage[donor]}

    cross, tries_cross = uniform_cross_lineage_permutation(rng, sorted(heldout), lineage)

    mean = {}
    for c in sorted(heldout):
        l = lineage[c]
        n_dev = dev_by_lineage.get(l, 0)
        if n_dev >= C.MIN_DEV_FOR_LINEAGE_MEAN:
            mean[c] = {"source": "lineage_mean", "lineage": l, "n_development": n_dev}
        else:
            mean[c] = {"source": "development_wide_mean", "lineage": l, "n_development": n_dev,
                       "flag": "too_few_development_lines",
                       "n_development_total": n_dev_total}

    assert set(within) == set(heldout) and set(cross) == set(heldout)
    assert all(within[c] != c for c in heldout)
    assert all(lineage[cross[c]] != lineage[c] for c in heldout)
    # every donor is a different person and not merely a different cell line, in both donor arms
    assert all(patient[within[c]] != patient[c] for c in heldout)
    assert all(patient[cross[c]] != patient[c] for c in heldout)
    same_lin = sum(lineage[within[c]] == lineage[c] for c in heldout)
    return {
        "seed": seed,
        "within": within,
        "within_flags": flags_within,
        "within_same_lineage_pairs": same_lin,
        "within_rejection_draws_by_lineage": tries_within,
        "cross": cross,
        "cross_rejection_draws": tries_cross,
        "mean": mean,
        "mean_fallback_cells": sorted(c for c in mean if "flag" in mean[c]),
        "donor_distinct_constraint": True,
        "heldout_lines_sharing_a_donor": sorted(
            c for c in heldout if sum(1 for o in heldout if patient[o] == patient[c]) > 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=C.MANIFEST_JSON)
    ap.add_argument("--out", default=C.SCHEDULES_JSON)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    samples = man["samples"]
    heldout = sorted(s["ModelID"] for s in samples if s["heldout"] in (True, "True"))
    lineage = {s["ModelID"]: s["OncotreeLineage"] for s in samples}
    patient = {s["ModelID"]: s["PatientID"] for s in samples}
    dev = [s for s in samples if s["development"] in (True, "True")]
    dev_by_lineage = {}
    for s in dev:
        dev_by_lineage[s["OncotreeLineage"]] = dev_by_lineage.get(s["OncotreeLineage"], 0) + 1

    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_heldout": len(heldout), "primary_seed": C.PRIMARY_SEED, "schedules": {}}
    for seed in C.SEEDS:
        t0 = time.time()
        sch = build_schedule(seed, heldout, lineage, patient, dev_by_lineage, len(dev))
        sch["seconds"] = round(time.time() - t0, 1)
        out["schedules"][str(seed)] = sch
        print(f"seed {seed}: within fallbacks {list(sch['within_flags'])}, "
              f"cross draws {sch['cross_rejection_draws']}, "
              f"mean fallbacks {sch['mean_fallback_cells']}, {sch['seconds']}s", flush=True)
    jsonutil.dump(out, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
