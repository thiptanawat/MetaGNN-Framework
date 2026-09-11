"""Brute-force checks of the maths the audit's conclusions rest on:

  1. the vectorised within-gene concordance (ties and missing gene effects included),
  2. the validity-mask path used by every domain (template_domain / explicit_only): a pair
     counts only when BOTH lines carry the mask bit for that gene, and
  3. the bootstrap identity C_g = (n' W_g n) / (n' V_g n), against an explicit resampled
     multiset of lines.

Each is checked against a naive double loop on small random data. Run: python3 selfcheck.py
"""
import itertools

import numpy as np

from metrics import concordance, pair_tensors


def brute(sc, ge, g, idx=None, mask=None):
    idx = list(range(len(ge)) if idx is None else idx)
    num = den = 0.0
    for a, b in itertools.combinations(range(len(idx)), 2):
        i, j = idx[a], idx[b]
        if np.isnan(ge[i, g]) or np.isnan(ge[j, g]) or ge[i, g] == ge[j, g]:
            continue
        if mask is not None and not (mask[i, g] and mask[j, g]):
            continue
        den += 1
        if sc[i, g] == sc[j, g]:
            num += 0.5
        elif (sc[i, g] > sc[j, g]) == (-ge[i, g] > -ge[j, g]):
            num += 1
    return (num / den if den else np.nan), den


def main(n_c=9, n_g=5, seed=7, trials=5):
    rng = np.random.default_rng(seed)
    ge = rng.normal(size=(n_c, n_g)).round(1)             # rounding creates gene-effect ties
    ge[rng.random((n_c, n_g)) < 0.1] = np.nan             # and missing values
    sc = rng.choice([0.0, 0.2, 0.5, 0.7], size=(n_c, n_g))  # and score ties
    mask = rng.random((n_c, n_g)) < 0.7                   # a template-domain-like validity mask

    # 1 + 2: unmasked and masked concordance
    for label, m in (("unmasked", None), ("masked (template-domain path)", mask)):
        W, V = pair_tensors(sc, ge, m)
        c, den = concordance(W, V)
        for g in range(n_g):
            cb, db = brute(sc, ge, g, mask=m)
            assert den[g] / 2 == db, (label, g, den[g] / 2, db)
            assert (np.isnan(c[g]) and np.isnan(cb)) or abs(c[g] - cb) < 1e-9, (label, g, c[g], cb)
        print(f"{label}: per-gene concordance matches brute force over "
              f"{int(den.sum() / 2)} pairs")

    # 3: bootstrap identity, checked on the masked tensors (the harder case)
    W, V = pair_tensors(sc, ge, mask)
    for t in range(trials):
        pick = rng.integers(n_c, size=n_c)
        n = np.bincount(pick, minlength=n_c).astype(np.float32)
        cq, dq = concordance(W, V, n)
        for g in range(n_g):
            cb, db = brute(sc, ge, g, idx=list(pick), mask=mask)
            assert abs(dq[g] / 2 - db) < 1e-6, (t, g, dq[g] / 2, db)
            assert (np.isnan(cq[g]) and np.isnan(cb)) or abs(cq[g] - cb) < 1e-6, (t, g, cq[g], cb)
        print(f"bootstrap replicate {t}: quadratic form matches the resampled multiset")
    print("selfcheck OK")


if __name__ == "__main__":
    main()
