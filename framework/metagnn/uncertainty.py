"""Monte Carlo Dropout for uncertainty estimation."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


class MCDropoutInference:
    """Perform stochastic inference with dropout enabled for uncertainty quantification."""

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 30,
        boundary_threshold_lower: float = 0.3,
        boundary_threshold_upper: float = 0.7,
    ):
        """Initialize MC Dropout inference.
        
        Args:
            model: MetaGNNModel with dropout layers.
            num_samples: Number of stochastic forward passes (T).
            boundary_threshold_lower: Lower threshold for boundary reactions.
            boundary_threshold_upper: Upper threshold for boundary reactions.
        """
        self.model = model
        self.num_samples = num_samples
        self.boundary_lower = boundary_threshold_lower
        self.boundary_upper = boundary_threshold_upper

    def infer(
        self, graph: HeteroData
    ) -> Dict[str, torch.Tensor]:
        """Perform T stochastic forward passes.
        
        Args:
            graph: Heterogeneous graph data.
            
        Returns:
            Dictionary with:
            - 'means': Mean predictions μ_r (n_reactions,)
            - 'stds': Predictive standard deviations σ_r (n_reactions,)
            - 'samples': All T sample predictions (T, n_reactions)
        """
        samples = []
        
        # Enable dropout and perform stochastic passes
        self.model.train()
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(graph, return_attention=False)
                scores = output["scores"]
                probs = torch.sigmoid(scores)
                samples.append(probs)
        
        samples = torch.stack(samples)  # (T, n_reactions)
        
        means = torch.mean(samples, dim=0)
        stds = torch.std(samples, dim=0)
        
        return {
            "means": means,
            "stds": stds,
            "samples": samples,
        }

    def identify_boundary_reactions(
        self, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Identify reactions with uncertain predictions.
        
        Reactions with 0.3 < s_r < 0.7 are classified as boundary reactions.
        
        Args:
            predictions: Mean predictions (n_reactions,).
            
        Returns:
            Boolean mask for boundary reactions.
        """
        return (predictions > self.boundary_lower) & (predictions < self.boundary_upper)

    def confidence_filtered_predictions(
        self,
        graph: HeteroData,
        confidence_threshold: float = 0.9,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get high-confidence predictions for agent validation.
        
        Args:
            graph: Heterogeneous graph data.
            confidence_threshold: Minimum allowed standard deviation percentile.
            
        Returns:
            Tuple of (high_confidence_predictions, confidence_mask).
        """
        inference_results = self.infer(graph)
        means = inference_results["means"]
        stds = inference_results["stds"]
        
        # High confidence = low uncertainty
        confidence_scores = 1.0 - (stds / (stds.max() + 1e-8))
        confidence_mask = confidence_scores > confidence_threshold
        
        return means, confidence_mask
