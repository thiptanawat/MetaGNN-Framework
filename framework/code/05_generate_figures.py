#!/usr/bin/env python3
"""
Figure generation script for MetaGNN pipeline.

Generates publication-quality figures including training curves, ROC curves,
PR curves, uncertainty distributions, and score histograms.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set style for publication-quality figures
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"


class FigureGenerationError(Exception):
    """Base exception for figure generation failures."""
    pass


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


def plot_training_curves(
    history: Dict[str, List[float]],
    output_dir: Path
) -> None:
    """
    Plot training curves (loss, F1, AUROC vs epoch).

    Args:
        history: Dictionary with training history
        output_dir: Directory to save figure
    """
    logger.info("Generating training curves...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    # Loss curve
    if "loss" in history and "val_loss" in history:
        axes[0, 0].plot(history["loss"], label="Train Loss", linewidth=2)
        axes[0, 0].plot(history["val_loss"], label="Val Loss", linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # F1 score
    if "val_f1" in history:
        axes[0, 1].plot(history["val_f1"], label="Val F1", linewidth=2, color="green")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("F1 Score")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])

    # AUROC
    if "val_auroc" in history:
        axes[1, 0].plot(history["val_auroc"], label="Val AUROC", linewidth=2, color="orange")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("AUROC")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0.5, 1.0])

    # Placeholder for additional metrics
    axes[1, 1].text(0.5, 0.5, "Additional metrics", ha="center", va="center")
    axes[1, 1].axis("off")

    plt.tight_layout()

    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved training curves to {output_path}")
    plt.close()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    output_dir: Path
) -> None:
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auroc: Area under ROC curve
        output_dir: Directory to save figure
    """
    logger.info("Generating ROC curve...")

    fig, ax = plt.subplots(figsize=(8, 8))

    # ROC curve
    ax.plot(fpr, tpr, linewidth=2.5, label=f"ROC Curve (AUC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Classifier")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    output_path = output_dir / "roc_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved ROC curve to {output_path}")
    plt.close()


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    auprc: float,
    output_dir: Path
) -> None:
    """
    Plot precision-recall curve.

    Args:
        precision: Precision values
        recall: Recall values
        auprc: Area under PR curve
        output_dir: Directory to save figure
    """
    logger.info("Generating PR curve...")

    fig, ax = plt.subplots(figsize=(8, 8))

    # PR curve
    ax.plot(recall, precision, linewidth=2.5, label=f"PR Curve (AUC = {auprc:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    output_path = output_dir / "pr_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved PR curve to {output_path}")
    plt.close()


def plot_uncertainty_distribution(
    uncertainty: np.ndarray,
    labels: np.ndarray,
    output_dir: Path
) -> None:
    """
    Plot uncertainty distribution by class.

    Args:
        uncertainty: Uncertainty estimates
        labels: Ground truth labels
        output_dir: Directory to save figure
    """
    logger.info("Generating uncertainty distribution plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Separate uncertainty by class
    uncertainty_positive = uncertainty[labels == 1]
    uncertainty_negative = uncertainty[labels == 0]

    # Plot distributions
    ax.hist(uncertainty_negative, bins=30, alpha=0.6, label="Negative", color="blue")
    ax.hist(uncertainty_positive, bins=30, alpha=0.6, label="Positive", color="red")

    ax.set_xlabel("Uncertainty", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Uncertainty Distribution by Class", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    output_path = output_dir / "uncertainty_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved uncertainty distribution to {output_path}")
    plt.close()


def plot_prediction_scores(
    predictions: np.ndarray,
    labels: np.ndarray,
    output_dir: Path
) -> None:
    """
    Plot histograms of prediction scores by class.

    Args:
        predictions: Predicted probabilities
        labels: Ground truth labels
        output_dir: Directory to save figure
    """
    logger.info("Generating prediction score histograms...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Separate predictions by class
    pred_positive = predictions[labels == 1]
    pred_negative = predictions[labels == 0]

    # Plot histograms
    ax.hist(pred_negative, bins=30, alpha=0.6, label="Negative", color="blue")
    ax.hist(pred_positive, bins=30, alpha=0.6, label="Positive", color="red")

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Prediction Score Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim([0, 1])

    output_path = output_dir / "prediction_scores.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved prediction scores to {output_path}")
    plt.close()


def plot_metric_summary(
    metrics: Dict,
    output_dir: Path
) -> None:
    """
    Plot summary of key metrics.

    Args:
        metrics: Dictionary with evaluation metrics
        output_dir: Directory to save figure
    """
    logger.info("Generating metrics summary plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract key metrics
    metric_names = ["AUROC", "AUPRC", "F1", "Precision", "Recall"]
    metric_keys = ["auroc", "auprc", "f1", "precision", "recall"]
    metric_values = [metrics.get(key, 0) for key in metric_keys]

    # Plot bar chart
    colors = sns.color_palette("husl", len(metric_names))
    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f"{value:.3f}",
                ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Performance Metrics Summary", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis="y")

    output_path = output_dir / "metrics_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved metrics summary to {output_path}")
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate figures for MetaGNN evaluation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save figures"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Determine output directory
    output_dir = args.output_dir or args.results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load evaluation results
        results_file = args.results_dir / "evaluation_results.json"
        if not results_file.exists():
            raise FigureGenerationError(f"Results file not found: {results_file}")

        with open(results_file, "r") as f:
            results = json.load(f)

        logger.info(f"Loaded evaluation results from {results_file}")

        # Load predictions
        predictions_file = args.results_dir / "predictions.json"
        if predictions_file.exists():
            with open(predictions_file, "r") as f:
                predictions_data = json.load(f)

            predictions = np.array(predictions_data.get("predictions", []))
            labels = np.array(predictions_data.get("labels", []))
            uncertainty = np.array(predictions_data.get("uncertainty", []))

            # Generate figures
            metrics = results.get("metrics", {})

            # Training curves (if available)
            if "history" in results.get("metrics", {}):
                plot_training_curves(metrics["history"], output_dir)

            # ROC curve
            if "roc_curve" in metrics:
                fpr = np.array(metrics["roc_curve"]["fpr"])
                tpr = np.array(metrics["roc_curve"]["tpr"])
                auroc = metrics.get("auroc", 0)
                plot_roc_curve(fpr, tpr, auroc, output_dir)

            # PR curve
            if "pr_curve" in metrics:
                precision = np.array(metrics["pr_curve"]["precision"])
                recall = np.array(metrics["pr_curve"]["recall"])
                auprc = metrics.get("auprc", 0)
                plot_pr_curve(precision, recall, auprc, output_dir)

            # Uncertainty distribution
            if len(uncertainty) > 0 and len(labels) > 0:
                plot_uncertainty_distribution(uncertainty, labels, output_dir)

            # Prediction scores
            if len(predictions) > 0 and len(labels) > 0:
                plot_prediction_scores(predictions, labels, output_dir)

            # Metrics summary
            plot_metric_summary(metrics, output_dir)

        logger.info("=" * 60)
        logger.info("Figure generation completed successfully")
        logger.info(f"Figures saved to {output_dir}")
        logger.info("=" * 60)

        return 0

    except FigureGenerationError as e:
        logger.error(f"Figure generation failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
