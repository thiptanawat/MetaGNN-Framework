"""Evaluation metrics and threshold analysis."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, f1_score, precision_score, recall_score


class MetaGNNEvaluator:
    """Compute evaluation metrics with threshold sweep capability."""

    def __init__(self, threshold: float = 0.15):
        """Initialize evaluator.
        
        Args:
            threshold: Default decision threshold for binary classification.
        """
        self.threshold = threshold

    def evaluate(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            predictions: Predicted scores (unbounded).
            labels: Ground truth binary labels.
            threshold: Classification threshold. Uses self.threshold if None.
            
        Returns:
            Dictionary with AUROC, AUPRC, F1, Precision, Recall.
        """
        if threshold is None:
            threshold = self.threshold
        
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        predictions = np.asarray(predictions).flatten()
        labels = np.asarray(labels).flatten()
        
        # Sigmoid for probabilities
        probs = 1.0 / (1.0 + np.exp(-predictions))
        preds_binary = (probs > threshold).astype(int)
        
        # Compute metrics
        try:
            auroc = roc_auc_score(labels, probs)
        except ValueError:
            auroc = 0.0
        
        try:
            precision, recall, _ = precision_recall_curve(labels, probs)
            auprc = auc(recall, precision)
        except ValueError:
            auprc = 0.0
        
        f1 = f1_score(labels, preds_binary, zero_division=0)
        precision = precision_score(labels, preds_binary, zero_division=0)
        recall = recall_score(labels, preds_binary, zero_division=0)
        
        return {
            "AUROC": auroc,
            "AUPRC": auprc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
        }

    def threshold_sweep(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        thresholds: Optional[List[float]] = None,
    ) -> Dict[float, Dict[str, float]]:
        """Evaluate metrics across a range of thresholds.
        
        Args:
            predictions: Predicted scores.
            labels: Ground truth labels.
            thresholds: List of thresholds to evaluate. If None, uses linspace(0, 1, 21).
            
        Returns:
            Dictionary mapping thresholds to metric dictionaries.
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 21)
        
        results = {}
        for thresh in thresholds:
            metrics = self.evaluate(predictions, labels, threshold=thresh)
            results[float(thresh)] = metrics
        
        return results

    def report_fold(
        self,
        fold_idx: int,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> str:
        """Generate formatted report for a single fold.
        
        Args:
            fold_idx: Fold index.
            predictions: Predicted scores.
            labels: Ground truth labels.
            
        Returns:
            Formatted report string.
        """
        metrics = self.evaluate(predictions, labels)
        
        report = f"\n--- Fold {fold_idx} ---\n"
        report += f"AUROC:    {metrics['AUROC']:.4f}\n"
        report += f"AUPRC:    {metrics['AUPRC']:.4f}\n"
        report += f"F1:       {metrics['F1']:.4f}\n"
        report += f"Precision: {metrics['Precision']:.4f}\n"
        report += f"Recall:   {metrics['Recall']:.4f}\n"
        
        return report

    def aggregate_folds(
        self,
        fold_results: List[Dict[str, float]],
    ) -> Dict[str, Tuple[float, float]]:
        """Aggregate metrics across folds (mean +/- std).
        
        Args:
            fold_results: List of metric dictionaries from each fold.
            
        Returns:
            Dictionary mapping metric names to (mean, std) tuples.
        """
        metrics_array = {}
        for metrics in fold_results:
            for metric_name, value in metrics.items():
                if metric_name not in metrics_array:
                    metrics_array[metric_name] = []
                metrics_array[metric_name].append(value)
        
        aggregated = {}
        for metric_name, values in metrics_array.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            aggregated[metric_name] = (mean_val, std_val)
        
        return aggregated
