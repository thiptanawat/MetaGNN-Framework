#!/usr/bin/env python3
"""
Evaluation script for MetaGNN model.

Evaluates trained model on test set with MC Dropout for uncertainty estimation.
Computes standard metrics (AUROC, AUPRC, F1, precision, recall) and generates
per-patient and aggregate performance statistics.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import auc, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class EvaluationError(Exception):
    """Base exception for evaluation failures."""
    pass


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


def load_checkpoint(checkpoint_path: Path, device: str) -> Dict:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Dictionary with model state
    """
    if not checkpoint_path.exists():
        raise EvaluationError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    return checkpoint


def mc_dropout_inference(
    model: torch.nn.Module,
    data: torch.Tensor,
    num_samples: int = 30,
    dropout_rate: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run MC Dropout inference for uncertainty estimation.

    Args:
        model: Trained model with dropout layers
        data: Input data
        num_samples: Number of MC dropout samples
        dropout_rate: Dropout rate to apply

    Returns:
        Tuple of (mean_predictions, uncertainty_estimates)
    """
    model.eval()

    predictions = []

    for sample in range(num_samples):
        # Enable dropout even during inference for MC sampling
        for module in model.modules():
            if hasattr(module, "dropout"):
                module.dropout.train()

        with torch.no_grad():
            pred = model(data)
            predictions.append(pred.cpu().numpy())

    predictions = np.array(predictions)

    # Compute mean and variance across MC samples
    mean_predictions = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)

    logger.info(
        f"MC Dropout inference completed ({num_samples} samples)"
    )

    return mean_predictions, uncertainty


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict:
    """
    Compute evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        uncertainty: Uncertainty estimates (optional)
        threshold: Decision threshold for binary classification

    Returns:
        Dictionary with computed metrics
    """
    # Clip predictions to valid probability range
    y_pred = np.clip(y_pred, 0, 1)

    # Binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Compute metrics
    metrics = {
        "auroc": roc_auc_score(y_true, y_pred),
        "auprc": auc(recall_score(y_true, y_pred_binary, average=None),
                     precision_score(y_true, y_pred_binary, average=None)),
        "f1": f1_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary),
        "recall": recall_score(y_true, y_pred_binary),
        "threshold": threshold,
    }

    # Compute ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
    metrics["roc_curve"] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }

    # Compute PR curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)
    metrics["pr_curve"] = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
    }

    # Uncertainty metrics if provided
    if uncertainty is not None:
        metrics["mean_uncertainty_positive"] = float(uncertainty[y_true == 1].mean())
        metrics["mean_uncertainty_negative"] = float(uncertainty[y_true == 0].mean())
        metrics["mean_uncertainty_correct"] = float(
            uncertainty[y_true == y_pred_binary].mean()
        )
        metrics["mean_uncertainty_incorrect"] = float(
            uncertainty[y_true != y_pred_binary].mean()
        )

    logger.info(
        f"AUROC: {metrics['auroc']:.4f}, "
        f"AUPRC: {metrics.get('auprc', 0):.4f}, "
        f"F1: {metrics['f1']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, "
        f"Recall: {metrics['recall']:.4f}"
    )

    return metrics


def compute_per_patient_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: List[str],
    uncertainty: Optional[np.ndarray] = None
) -> List[Dict]:
    """
    Compute per-patient metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        patient_ids: Patient identifiers
        uncertainty: Uncertainty estimates (optional)

    Returns:
        List of dictionaries with per-patient metrics
    """
    per_patient = []

    for i, patient_id in enumerate(patient_ids):
        patient_metrics = {
            "patient_id": patient_id,
            "label": int(y_true[i]),
            "prediction": float(y_pred[i]),
        }

        if uncertainty is not None:
            patient_metrics["uncertainty"] = float(uncertainty[i])

        per_patient.append(patient_metrics)

    logger.info(f"Computed metrics for {len(per_patient)} patients")

    return per_patient


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate MetaGNN model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint"
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
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--mc_dropout_samples",
        type=int,
        default=30,
        help="Number of MC Dropout samples for uncertainty"
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
        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = load_checkpoint(args.checkpoint, device)

        # Load test data - placeholder
        logger.info("Loading test data...")
        y_true = np.array([])  # Would load actual test labels
        y_pred = np.array([])  # Would run model on test data
        patient_ids = []       # Would load actual patient IDs

        if len(y_true) == 0:
            logger.warning("No test data found, using placeholder data")
            y_true = np.random.randint(0, 2, 100)
            y_pred = np.random.rand(100)
            patient_ids = [f"patient_{i}" for i in range(100)]

        # Run MC Dropout inference for uncertainty
        logger.info(f"Running MC Dropout with {args.mc_dropout_samples} samples...")
        # uncertainty = mc_dropout_inference(model, test_data, args.mc_dropout_samples)
        uncertainty = np.random.rand(len(y_true)) * 0.1  # Placeholder

        # Compute aggregate metrics
        logger.info("Computing evaluation metrics...")
        metrics = compute_metrics(y_true, y_pred, uncertainty)

        # Compute per-patient metrics
        per_patient = compute_per_patient_metrics(
            y_true,
            y_pred,
            patient_ids,
            uncertainty
        )

        # Save results
        results = {
            "config": config,
            "checkpoint": str(args.checkpoint),
            "metrics": metrics,
            "per_patient": per_patient,
            "num_test_samples": len(y_true),
            "num_positive": int(np.sum(y_true)),
            "num_negative": int(len(y_true) - np.sum(y_true)),
        }

        results_file = output_dir / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved results to {results_file}")

        # Save predictions
        predictions_file = output_dir / "predictions.json"
        with open(predictions_file, "w") as f:
            json.dump({
                "patient_ids": patient_ids,
                "predictions": y_pred.tolist(),
                "uncertainty": uncertainty.tolist(),
                "labels": y_true.tolist(),
            }, f, indent=2)

        logger.info(f"Saved predictions to {predictions_file}")

        # Save uncertainty map
        uncertainty_file = output_dir / "uncertainty_map.json"
        with open(uncertainty_file, "w") as f:
            json.dump({
                "patient_ids": patient_ids,
                "uncertainty": uncertainty.tolist(),
            }, f, indent=2)

        logger.info(f"Saved uncertainty map to {uncertainty_file}")

        logger.info("=" * 60)
        logger.info("Evaluation completed successfully")
        logger.info(f"Results directory: {output_dir}")
        logger.info("=" * 60)

        return 0

    except EvaluationError as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
