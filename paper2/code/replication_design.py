#!/usr/bin/env python3
"""Sample of the prespecified replication of the own-versus-donor contrast (plan of 13 November 2026).

Draws, from one recorded seed, the reactions, the ordered list of patients and the donor schedules of
the replication, and writes them to results/replication/design_master.json. Reactions and patients
that any earlier reaction-activity collection of this study used are excluded, so the replication
sample is disjoint from every sample the earlier analyses saw. The file is committed together with
the analysis plan before the first call of the replication is issued; nothing in it depends on a
model answer.

    python3 code/replication_design.py            # writes the file; refuses to overwrite it
    python3 code/replication_design.py --check    # rebuilds the design and compares it with the file

The patients are kept in the order of the draw. The collection proceeds through that list in blocks,
and the analysis uses the blocks that were completed (REPLICATION_PLAN_2026-11-13.md).
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SEED = 20261113
N_REACTIONS = 300
N_PATIENTS_MAX = 120          # upper bound; the blocks actually completed define the analysis set
SCHEDULES = ("d1", "d2", "d3")  # three independent donor schedules; d1 is the primary donor arm
OUT = os.path.join(ROOT, "results", "replication", "design_master.json")


def earlier_designs():
    """Reactions and patients of every archived reaction-activity collection (design.json files
    that list both), with the names of the collections."""
    rx, pt, names = set(), set(), []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "*", "design.json"))):
        d = json.load(open(f))
        if "reactions" in d and "patients" in d:
            rx |= set(int(x) for x in d["reactions"]); pt |= set(int(x) for x in d["patients"])
            names.append(os.path.basename(os.path.dirname(f)))
    return rx, pt, names


def build():
    sys.path.insert(0, HERE)
    os.environ.setdefault("RECON3D_ALIGNED", os.path.join(ROOT, "recon3d_aligned.json"))
    if not os.path.exists(os.environ["RECON3D_ALIGNED"]):
        import gzip, shutil
        with gzip.open(os.environ["RECON3D_ALIGNED"] + ".gz", "rb") as fi, open(os.environ["RECON3D_ALIGNED"], "wb") as fo:
            shutil.copyfileobj(fi, fo)
    from aligned_rxn import ALIGNED
    D = np.load(os.path.join(ROOT, "rxn_context.npz"), allow_pickle=True)
    n_pat = int(D["X"].shape[0]); n_rxn = int(D["X"].shape[1])
    assert len(ALIGNED) == n_rxn
    used_rx, used_pt, names = earlier_designs()
    rng = np.random.default_rng(SEED)
    pool = np.array([i for i in np.where(ALIGNED)[0].tolist() if i not in used_rx])
    reactions = sorted(int(x) for x in rng.choice(pool, size=N_REACTIONS, replace=False))
    avail = np.array([k for k in range(n_pat) if k not in used_pt])
    patients = [int(x) for x in rng.permutation(avail)[:N_PATIENTS_MAX]]
    donors = {}
    for s, name in enumerate(SCHEDULES, 1):
        g = np.random.default_rng(SEED + s)
        # at every cell one donor, uniform over the other patients of the cohort (the donor is
        # redrawn at every reaction, as in the earlier collections)
        k = g.integers(0, n_pat - 1, size=(len(patients), len(reactions)))
        pj = np.array(patients)[:, None]
        donors[name] = (k + (k >= pj)).astype(int).tolist()
        assert all(donors[name][a][b] != patients[a] for a in range(len(patients)) for b in range(len(reactions)))
    return dict(
        note="sample of the prespecified replication; written before the first call",
        seed=SEED, n_reactions=N_REACTIONS, n_patients_max=N_PATIENTS_MAX, schedules=list(SCHEDULES),
        cohort_patients=n_pat, aligned_pool=int(ALIGNED.sum()),
        excluded_reactions=len(used_rx), excluded_patients=len(used_pt), excluded_from=names,
        pool_reactions=int(len(pool)), pool_patients=int(len(avail)),
        reactions=reactions, patients=patients, donors=donors)


def digest(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    d = build()
    if a.check:
        old = json.load(open(OUT))
        same = digest(old) == digest(d)
        print("design_master.json", "matches the rebuilt design" if same else "DIFFERS from the rebuilt design", digest(old))
        sys.exit(0 if same else 1)
    if os.path.exists(OUT):
        sys.exit("refusing to overwrite " + OUT)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(d, open(OUT, "w"), separators=(",", ":"))
    print("wrote", OUT)
    print("reactions", len(d["reactions"]), "of a pool of", d["pool_reactions"], "| patients", len(d["patients"]),
          "of a pool of", d["pool_patients"], "| canonical sha256", digest(d))
