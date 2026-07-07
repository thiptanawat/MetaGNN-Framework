#!/usr/bin/env python3
"""
Multi-seed robustness experiment that wraps the real train_metagnn.py
in ~/Documents/GitHub/MetaGNN/src/.

For each seed it:
  1. Calls src/train_metagnn.py --seed S --results-dir results/seed_S ...
  2. Reads results/seed_S/results_summary.json (or test_results.csv)
  3. Appends the metrics to per_seed_metrics.json

Usage
-----
    python code/06_multiseed_experiment.py \
        --repo_root ~/Documents/GitHub/MetaGNN \
        --output_dir ~/Documents/GitHub/MetaGNN/results/multiseed_220 \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --device auto

Assumes train_metagnn.py has been patched to accept --seed.
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# Training hyperparameters that match run.sh. Override on the CLI if needed.
DEFAULT_TRAIN_ARGS = dict(
    hidden_dim=256,
    n_layers=3,
    n_heads=8,
    dropout=0.2,
    s1_epochs=50,
    s2_epochs=200,
    patience=20,
    mc_passes=100,
)


def load_metrics_from_run(run_dir: Path) -> Dict:
    """Collect metrics from whatever the training script produced.

    Priority order:
      1. results_summary.json (preferred; any dict -> copied as-is)
      2. test_results.csv     (CSV; compute AUROC/AUPRC/F1 columns if present)
    """
    summary_json = run_dir / "results_summary.json"
    if summary_json.exists():
        with open(summary_json, "r") as f:
            data = json.load(f)
        # Flatten one level in case the JSON has nested "test" dict
        flat: Dict[str, float] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}_{kk}"] = vv
            else:
                flat[k] = v
        return flat

    csv_path = run_dir / "test_results.csv"
    if csv_path.exists():
        import numpy as np
        metrics: Dict[str, float] = {}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        numeric_cols = set()
        for row in rows:
            for k, v in row.items():
                if v is None:
                    continue
                try:
                    float(v)
                    numeric_cols.add(k)
                except ValueError:
                    pass
        for col in numeric_cols:
            values = np.asarray([float(r[col]) for r in rows if r.get(col)])
            if values.size:
                metrics[col] = float(values.mean())
        return metrics

    raise FileNotFoundError(
        f"Neither results_summary.json nor test_results.csv found in {run_dir}"
    )


def run_single_seed(seed: int, repo_root: Path, run_dir: Path,
                    device: str, train_kwargs: Dict) -> Dict:
    train_script = repo_root / "src" / "train_metagnn.py"
    data_dir = repo_root / "data" / "processed"

    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(train_script),
        "--data-dir", str(data_dir),
        "--results-dir", str(run_dir),
        "--device", device,
        "--seed", str(seed),
        "--hidden-dim", str(train_kwargs["hidden_dim"]),
        "--n-layers", str(train_kwargs["n_layers"]),
        "--n-heads", str(train_kwargs["n_heads"]),
        "--dropout", str(train_kwargs["dropout"]),
        "--s1-epochs", str(train_kwargs["s1_epochs"]),
        "--s2-epochs", str(train_kwargs["s2_epochs"]),
        "--patience", str(train_kwargs["patience"]),
        "--mc-passes", str(train_kwargs["mc_passes"]),
    ]

    logger.info("Seed %d: launching training", seed)
    logger.info("  %s", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    train_sec = time.time() - t0
    logger.info("Seed %d: training finished in %.1fs", seed, train_sec)

    metrics = load_metrics_from_run(run_dir)
    metrics["seed"] = seed
    metrics["train_seconds"] = train_sec
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run MetaGNN across multiple seeds on the original code."
    )
    parser.add_argument("--repo_root", type=Path, required=True,
                        help="Path to ~/Documents/GitHub/MetaGNN (contains src/, data/)")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Where per-seed subdirectories and summary are written")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(10)))
    parser.add_argument("--device", type=str, default="auto",
                        help="Passed through to train_metagnn.py (auto/cpu/mps/cuda)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip a seed if its run_dir already has metrics")
    # Allow overriding model hyperparameters
    for k, v in DEFAULT_TRAIN_ARGS.items():
        parser.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = {k: getattr(args, k) for k in DEFAULT_TRAIN_ARGS}

    per_seed_metrics: List[Dict] = []
    for seed in args.seeds:
        run_dir = output_dir / f"seed_{seed:03d}"
        if args.skip_existing and (
            (run_dir / "results_summary.json").exists()
            or (run_dir / "test_results.csv").exists()
        ):
            logger.info("Seed %d already has metrics, loading from disk", seed)
            metrics = load_metrics_from_run(run_dir)
            metrics["seed"] = seed
            per_seed_metrics.append(metrics)
            continue

        metrics = run_single_seed(seed, repo_root, run_dir,
                                  args.device, train_kwargs)
        per_seed_metrics.append(metrics)

        # Checkpoint after every seed so a mid-run crash does not lose progress
        with open(output_dir / "per_seed_metrics.json", "w") as f:
            json.dump(per_seed_metrics, f, indent=2)

    logger.info("Done. Completed %d/%d seeds.",
                len(per_seed_metrics), len(args.seeds))
    logger.info("Summary written to %s", output_dir / "per_seed_metrics.json")


if __name__ == "__main__":
    main()
