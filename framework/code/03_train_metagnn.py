#!/usr/bin/env python3
"""
Training script for MetaGNN model.

Trains MetaGNN on heterogeneous graphs with support for single split
and k-fold cross-validation modes. Saves checkpoints and training curves.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class TrainingError(Exception):
    """Base exception for training failures."""
    pass


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")


def create_data_splits(
    num_samples: int,
    split_config: Dict,
    seed: int
) -> Dict[str, List[int]]:
    """
    Create train/val/test splits.

    Args:
        num_samples: Total number of samples
        split_config: Split configuration with mode and ratios
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', 'test' indices
    """
    set_seed(seed)

    mode = split_config.get("mode", "single")

    if mode == "single":
        # Single split with fixed train/val/test ratios
        train_ratio = split_config.get("train_ratio", 0.7)
        val_ratio = split_config.get("val_ratio", 0.15)
        test_ratio = split_config.get("test_ratio", 0.15)

        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        train_end = int(num_samples * train_ratio)
        val_end = train_end + int(num_samples * val_ratio)

        splits = {
            "train": indices[:train_end].tolist(),
            "val": indices[train_end:val_end].tolist(),
            "test": indices[val_end:].tolist(),
        }

        logger.info(
            f"Created single split: train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}"
        )

        return splits

    elif mode == "kfold":
        # K-fold cross-validation
        num_folds = split_config.get("num_folds", 5)
        stratify_column = split_config.get("stratify_by", None)

        logger.info(f"Will use {num_folds}-fold cross-validation")

        # Placeholder for stratified k-fold logic
        fold_splits = {}
        fold_size = num_samples // num_folds

        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for fold in range(num_folds):
            fold_start = fold * fold_size
            fold_end = fold_start + fold_size if fold < num_folds - 1 else num_samples

            test_indices = indices[fold_start:fold_end]
            train_indices = np.concatenate([indices[:fold_start], indices[fold_end:]])

            # Split train into train/val (80/20)
            train_val_split = int(0.8 * len(train_indices))
            perm = np.random.permutation(len(train_indices))
            train_perm = perm[:train_val_split]
            val_perm = perm[train_val_split:]

            fold_splits[fold] = {
                "train": train_indices[train_perm].tolist(),
                "val": train_indices[val_perm].tolist(),
                "test": test_indices.tolist(),
            }

            logger.info(
                f"Fold {fold}: train={len(fold_splits[fold]['train'])}, "
                f"val={len(fold_splits[fold]['val'])}, test={len(fold_splits[fold]['test'])}"
            )

        return fold_splits

    else:
        raise TrainingError(f"Unknown split mode: {mode}")


def train_fold(
    fold: int,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    config: Dict,
    output_dir: Path
) -> Dict:
    """
    Train model on a single fold.

    Args:
        fold: Fold number
        train_indices: Training sample indices
        val_indices: Validation sample indices
        test_indices: Test sample indices
        config: Model configuration
        output_dir: Directory to save outputs

    Returns:
        Dictionary with training metrics
    """
    logger.info(f"Training fold {fold}...")

    # Model hyperparameters from config
    hidden_dim = config.get("hidden_dim", 256)
    num_layers = config.get("layers", 3)
    num_heads = config.get("heads", 8)
    epochs = config.get("epochs", 200)
    patience = config.get("patience", 15)
    learning_rate = config.get("learning_rate", 1e-3)

    logger.info(
        f"Model config: hidden_dim={hidden_dim}, layers={num_layers}, "
        f"heads={num_heads}, epochs={epochs}, patience={patience}"
    )

    # Placeholder for actual training loop
    training_history = {
        "loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_auroc": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Placeholder training step
        train_loss = 0.0
        val_loss = 0.0
        val_f1 = 0.0
        val_auroc = 0.0

        # Logging
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Fold {fold} | Epoch {epoch + 1}/{epochs} | "
                f"loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_f1={val_f1:.4f} | val_auroc={val_auroc:.4f}"
            )

        training_history["loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["val_f1"].append(val_f1)
        training_history["val_auroc"].append(val_auroc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0

            # Save checkpoint
            checkpoint_dir = output_dir / f"fold_{fold}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "best_model.pt"

            logger.info(f"Saving checkpoint to {checkpoint_path}")

        else:
            epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    logger.info(f"Fold {fold} training completed. Best epoch: {best_epoch}")

    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": training_history,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train MetaGNN model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Root directory for data files"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save checkpoints and results"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Specific fold to train (for k-fold CV)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Determine paths
    data_root = args.data_root or Path(config.get("data_root", "./data"))
    data_root = data_root.resolve()

    output_dir = args.output_dir or Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        # Load graph
        graph_dir = data_root / "graphs"
        cohort = config.get("cohort", "graph")
        graph_path = graph_dir / f"{cohort}.pt"

        if not graph_path.exists():
            raise TrainingError(f"Graph file not found: {graph_path}")

        logger.info(f"Loading graph from {graph_path}")
        hetero_data = torch.load(graph_path)

        # Get number of samples from graph
        num_samples = 0  # Would extract from hetero_data
        logger.info(f"Loaded graph with {num_samples} samples")

        # Create data splits
        split_config = config.get("split", {"mode": "single"})
        seeds = config.get("seeds", [2024, 42, 123])

        # Determine if using k-fold or single split
        if split_config.get("mode") == "kfold":
            # K-fold cross-validation
            num_folds = split_config.get("num_folds", 5)

            # If specific fold requested, train only that fold
            if args.fold is not None:
                folds_to_train = [args.fold]
            else:
                folds_to_train = range(num_folds)

            fold_results = []

            for fold in folds_to_train:
                for seed_idx, seed in enumerate(seeds):
                    set_seed(seed)

                    splits = create_data_splits(num_samples, split_config, seed)

                    fold_data = splits[fold]
                    result = train_fold(
                        fold,
                        fold_data["train"],
                        fold_data["val"],
                        fold_data["test"],
                        config,
                        output_dir
                    )

                    fold_results.append(result)

            # Save results
            results_file = output_dir / "kfold_results.json"
            with open(results_file, "w") as f:
                json.dump(fold_results, f, indent=2)
            logger.info(f"Saved k-fold results to {results_file}")

        else:
            # Single split with multiple seeds
            for seed_idx, seed in enumerate(seeds):
                set_seed(seed)

                splits = create_data_splits(num_samples, split_config, seed)

                result = train_fold(
                    0,
                    splits["train"],
                    splits["val"],
                    splits["test"],
                    config,
                    output_dir / f"seed_{seed}"
                )

                logger.info(f"Completed training with seed {seed}")

        logger.info("=" * 60)
        logger.info("Training completed successfully")
        logger.info(f"Results saved to {output_dir}")
        logger.info("=" * 60)

        return 0

    except TrainingError as e:
        logger.error(f"Training failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
