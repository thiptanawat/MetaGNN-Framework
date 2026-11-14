#!/usr/bin/env python3
"""Sample of the second prespecified replication of the own-versus-donor contrast (plan of 14 November 2026).

Draws, from one recorded seed, the reactions, the ordered list of patients and the donor schedules of
the second replication, and writes them to results/replication2/design_master.json. Excluded are the
reactions and patients of every archived reaction-activity collection, as in replication_design.py,
and in addition the 300 reactions and the whole ordered list of 120 patients of the first replication
sample (results/replication/design_master.json), whether or not a patient of that list was called.
The second sample is therefore disjoint from every sample an earlier analysis saw. The file is
committed together with the analysis plan before the first call on this sample; nothing in it depends
on a model answer.

    python3 code/replication2_design.py            # writes the file; refuses to overwrite it
    python3 code/replication2_design.py --check    # rebuilds the design and compares it with the file

The patients are kept in the order of the draw. The collection proceeds through that list in blocks of
ten, and the analysis uses the blocks that were completed (REPLICATION2_PLAN_2026-11-14.md).
"""
import argparse, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replication_design as RD1          # the first sample's script: its list of earlier designs and its digest

SEED = 20261115
N_REACTIONS = 300
N_PATIENTS = 200                # 20 blocks of 10; the blocks actually completed define the analysis set
SCHEDULES = ("d1", "d2", "d3")  # three independent donor schedules, as in the first replication
FIRST = os.path.join(ROOT, "results", "replication", "design_master.json")
OUT = os.path.join(ROOT, "results", "replication2", "design_master.json")


def build():
    os.environ.setdefault("RECON3D_ALIGNED", os.path.join(ROOT, "recon3d_aligned.json"))
    if not os.path.exists(os.environ["RECON3D_ALIGNED"]):
        import gzip, shutil
        with gzip.open(os.environ["RECON3D_ALIGNED"] + ".gz", "rb") as fi, open(os.environ["RECON3D_ALIGNED"], "wb") as fo:
            shutil.copyfileobj(fi, fo)
    from aligned_rxn import ALIGNED
    D = np.load(os.path.join(ROOT, "rxn_context.npz"), allow_pickle=True)
    n_pat = int(D["X"].shape[0]); n_rxn = int(D["X"].shape[1])
    assert len(ALIGNED) == n_rxn
    used_rx, used_pt, names = RD1.earlier_designs()
    first = json.load(open(FIRST))
    n_archived_rx, n_archived_pt = len(used_rx), len(used_pt)
    assert not (set(first["reactions"]) & used_rx) and not (set(first["patients"]) & used_pt)
    used_rx |= set(int(x) for x in first["reactions"]); used_pt |= set(int(x) for x in first["patients"])
    rng = np.random.default_rng(SEED)
    pool = np.array([i for i in np.where(ALIGNED)[0].tolist() if i not in used_rx])
    reactions = sorted(int(x) for x in rng.choice(pool, size=N_REACTIONS, replace=False))
    avail = np.array([k for k in range(n_pat) if k not in used_pt])
    patients = [int(x) for x in rng.permutation(avail)[:N_PATIENTS]]
    donors = {}
    for s, name in enumerate(SCHEDULES, 1):
        g = np.random.default_rng(SEED + s)
        # at every cell one donor, uniform over the other patients of the cohort (the donor is
        # redrawn at every reaction, as in the earlier collections)
        k = g.integers(0, n_pat - 1, size=(len(patients), len(reactions)))
        pj = np.array(patients)[:, None]
        donors[name] = (k + (k >= pj)).astype(int).tolist()
        assert all(donors[name][a][b] != patients[a] for a in range(len(patients)) for b in range(len(reactions)))
    assert not (set(reactions) & used_rx) and not (set(patients) & used_pt)
    return dict(
        note="sample of the second prespecified replication; written before the first call on it",
        seed=SEED, n_reactions=N_REACTIONS, n_patients_max=N_PATIENTS, schedules=list(SCHEDULES),
        cohort_patients=n_pat, aligned_pool=int(ALIGNED.sum()),
        excluded_reactions=len(used_rx), excluded_patients=len(used_pt), excluded_from=names + ["replication (design_master.json)"],
        excluded_reactions_archived_designs=n_archived_rx, excluded_patients_archived_designs=n_archived_pt,
        excluded_reactions_first_replication=len(first["reactions"]), excluded_patients_first_replication=len(first["patients"]),
        first_replication_master_sha256=hashlib.sha256(open(FIRST, "rb").read()).hexdigest(),
        pool_reactions=int(len(pool)), pool_patients=int(len(avail)),
        reactions=reactions, patients=patients, donors=donors)


digest = RD1.digest

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    d = build()
    if a.check:
        old = json.load(open(OUT))
        same = digest(old) == digest(d)
        print("replication2/design_master.json", "matches the rebuilt design" if same else "DIFFERS from the rebuilt design", digest(old))
        sys.exit(0 if same else 1)
    if os.path.exists(OUT):
        sys.exit("refusing to overwrite " + OUT)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(d, open(OUT, "w"), separators=(",", ":"))
    print("wrote", OUT)
    print("reactions", len(d["reactions"]), "of a pool of", d["pool_reactions"], "| patients", len(d["patients"]),
          "of a pool of", d["pool_patients"], "| canonical sha256", digest(d))
