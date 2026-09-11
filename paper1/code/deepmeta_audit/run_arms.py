"""Step 4: run DeepMeta inference for each arm.

Each batch of cells gets a fresh PyG root directory; the processed graphs (60-180 MB per cell)
are deleted as soon as the batch's predictions are written, so peak disk stays at
batch_size x ~180 MB. Results are appended to preds/seed<seed>/preds_<arm>.csv
(cell, gene_name, preds_raw) after every batch, so an interrupted run resumes where it stopped.

  python3 run_arms.py --seed 11 --arms own within --batch_size 4
  python3 run_arms.py --seed 11 --limit 6           # smoke test
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd

import config as C


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def done_cells(path):
    if not os.path.exists(path):
        return set()
    try:
        return set(pd.read_csv(path, usecols=["cell"]).cell)
    except Exception:
        return set()


def run_cells(cells, sen_dir, exp_file, out_csv, batch_size, cores, device, model_file,
              work_root, tag="run"):
    """Predict for `cells`, one fresh PyG root per batch, appending after every batch."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(work_root, exist_ok=True)
    have = done_cells(out_csv)
    todo = [c for c in cells if c not in have]
    log(f"{tag}: {len(cells)} cells, {len(todo)} to run")
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    t_all = time.time()
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        work = os.path.join(work_root, f"{tag}_{i}")
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(os.path.join(work, "raw"), exist_ok=True)
        cl = os.path.join(work, "batch_cells.csv")
        pd.DataFrame({"cell": batch, "cell_index": range(len(batch))}).to_csv(cl, index=False)
        res = os.path.join(work, "res.csv")
        cmd = [sys.executable, os.path.join(C.AUDIT_DIR, "pred_enzyme_nograd.py"),
               "-e", exp_file, "-g", work, "-c", cl,
               "-n", sen_dir.rstrip(os.sep) + os.sep, "-t", str(cores),
               "-m", model_file, "-o", res, "-d", "val", "-b", "1", "-a", device]
        t0 = time.time()
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if p.returncode != 0 or not os.path.exists(res):
            print(p.stdout[-3000:], file=sys.stderr)
            print(p.stderr[-5000:], file=sys.stderr)
            shutil.rmtree(work, ignore_errors=True)
            raise RuntimeError(f"inference failed for {tag} batch starting at {i}")
        df = pd.read_csv(res)[["cell", "gene_name", "preds_raw"]]
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        shutil.rmtree(work, ignore_errors=True)     # keep peak disk at batch_size x ~180 MB
        log(f"  {tag} {min(i + batch_size, len(todo))}/{len(todo)} cells, "
            f"{len(df)} rows, {time.time() - t0:.0f}s/batch")
    log(f"{tag} finished in {time.time() - t_all:.0f}s -> {out_csv}")
    return out_csv


def run_arm(arm, seed, batch_size, cores, device, limit, out_dir, work_root, model_file,
            tag=""):
    spec = json.load(open(os.path.join(C.ARMS, f"seed{seed}", f"arm_{arm}{tag}.json")))
    cells = list(pd.read_csv(spec["cells_csv"]).cell)
    if limit:
        cells = cells[:limit]
    log(f"arm {arm}: SEN {spec['sen_source']}, exp {spec['exp_source']}")
    return run_cells(cells, spec["sen_dir"], spec["exp_file"],
                     os.path.join(out_dir, f"preds_{arm}.csv"),
                     batch_size, cores, device, model_file, work_root, tag=arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=C.PRIMARY_SEED)
    ap.add_argument("--arms", nargs="*", default=C.ARM_NAMES)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--cores", type=int, default=1, help="-t of pred_enzyme (graph building)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--limit", type=int, default=None, help="first N cells of each arm")
    ap.add_argument("--model", default=C.CKPT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="", help="read arm_<arm><tag>.json (see build_arms --tag)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(C.PREDS, f"seed{args.seed}")
    work_root = os.path.join(C.WORK, f"seed{args.seed}")
    C.ensure_dirs(out_dir, work_root)
    for arm in args.arms:
        run_arm(arm, args.seed, args.batch_size, args.cores, args.device, args.limit,
                out_dir, work_root, args.model, args.tag)
    shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    main()
