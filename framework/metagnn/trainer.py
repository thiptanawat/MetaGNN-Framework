"""Training loop with mass-balance regularization."""

from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import HeteroData


logger = logging.getLogger(__name__)


class MetaGNNTrainer:
    """Trainer for MetaGNN with combined BCE + mass-balance loss."""

    def __init__(
        self,
        model: nn.Module,
        graph: HeteroData,
        device: str = "cpu",
        lr: float = 5e-4,
        weight_decay: float = 1e-5,
        lambda_mb: float = 0.2,
        patience: int = 15,
        checkpoint_dir: Optional[Path] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: MetaGNNModel instance.
            graph: Heterogeneous graph data.
            device: Device for training (cpu or cuda).
            lr: Learning rate.
            weight_decay: L2 regularization weight.
            lambda_mb: Mass-balance loss weight.
            patience: Early stopping patience.
            checkpoint_dir: Directory to save checkpoints.
        """
        self.model = model.to(device)
        self.graph = graph.to(device)
        self.device = device
        self.lambda_mb = lambda_mb
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Compute pos_weight for BCEWithLogits
        # Placeholder: assume balanced classes
        self.pos_weight = torch.tensor(1.0, device=device)
        
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.val_f1_history = []

    def compute_mass_balance_loss(
        self,
        reaction_scores: torch.Tensor,
        stoichiometry_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mass-balance regularization loss.
        
        L_MB = (1/|V_M|) * Σ_m (Σ_r S_{m,r} * s_r)^2
        
        Args:
            reaction_scores: Reaction prediction scores.
            stoichiometry_matrix: Stoichiometry matrix S (n_metabolites, n_reactions).
                                 If None, returns zero loss.
            
        Returns:
            Scalar mass-balance loss.
        """
        if stoichiometry_matrix is None:
            return torch.tensor(0.0, device=self.device)
        
        # Matrix multiplication: (n_met,) = (n_met, n_rxn) @ (n_rxn,)
        metabolite_balance = torch.matmul(
            stoichiometry_matrix, reaction_scores
        )
        
        loss_mb = torch.mean(metabolite_balance ** 2)
        return loss_mb

    def train_epoch(
        self,
        train_loader,
        reaction_labels: torch.Tensor,
        stoichiometry_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: DataLoader for training batches.
            reaction_labels: Binary labels for each reaction.
            stoichiometry_matrix: Optional stoichiometry matrix for mass-balance loss.
            
        Returns:
            Dictionary with 'train_loss', 'train_bce', 'train_mb'.
        """
        self.model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_mb = 0.0
        num_batches = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(self.graph, return_attention=False)
            reaction_scores = output["scores"]
            
            # Compute BCE loss
            loss_bce = F.binary_cross_entropy_with_logits(
                reaction_scores,
                reaction_labels.float(),
                pos_weight=self.pos_weight,
            )
            
            # Compute mass-balance loss
            loss_mb = self.compute_mass_balance_loss(
                torch.sigmoid(reaction_scores),
                stoichiometry_matrix,
            )
            
            # Combined loss
            loss = loss_bce + self.lambda_mb * loss_mb
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_mb += loss_mb.item()
            num_batches += 1
        
        return {
            "train_loss": total_loss / num_batches,
            "train_bce": total_bce / num_batches,
            "train_mb": total_mb / num_batches,
        }

    @torch.no_grad()
    def validate(
        self,
        reaction_labels: torch.Tensor,
        stoichiometry_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Validate on full graph.
        
        Args:
            reaction_labels: Binary labels for each reaction.
            stoichiometry_matrix: Optional stoichiometry matrix.
            
        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        
        output = self.model(self.graph)
        reaction_scores = output["scores"]
        
        loss_bce = F.binary_cross_entropy_with_logits(
            reaction_scores,
            reaction_labels.float(),
            pos_weight=self.pos_weight,
        )
        
        loss_mb = self.compute_mass_balance_loss(
            torch.sigmoid(reaction_scores),
            stoichiometry_matrix,
        )
        
        total_loss = loss_bce + self.lambda_mb * loss_mb
        
        # Compute F1 for early stopping
        preds = (torch.sigmoid(reaction_scores) > 0.5).long()
        tp = ((preds == 1) & (reaction_labels == 1)).sum().float()
        fp = ((preds == 1) & (reaction_labels == 0)).sum().float()
        fn = ((preds == 0) & (reaction_labels == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            "val_loss": total_loss.item(),
            "val_bce": loss_bce.item(),
            "val_mb": loss_mb.item(),
            "val_f1": f1.item(),
        }

    def fit(
        self,
        train_loader,
        reaction_labels: torch.Tensor,
        stoichiometry_matrix: Optional[torch.Tensor] = None,
        num_epochs: int = 100,
        eval_freq: int = 1,
    ) -> Dict[str, list]:
        """Train model with early stopping.
        
        Args:
            train_loader: DataLoader for training.
            reaction_labels: Binary labels.
            stoichiometry_matrix: Optional mass-balance matrix.
            num_epochs: Maximum number of epochs.
            eval_freq: Validation frequency (every N epochs).
            
        Returns:
            Dictionary with training history.
        """
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
        }
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(
                train_loader, reaction_labels, stoichiometry_matrix
            )
            
            history["train_loss"].append(train_metrics["train_loss"])
            
            if epoch % eval_freq == 0:
                val_metrics = self.validate(reaction_labels, stoichiometry_matrix)
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_f1"].append(val_metrics["val_f1"])
                
                if val_metrics["val_f1"] > self.best_val_f1:
                    self.best_val_f1 = val_metrics["val_f1"]
                    self.best_epoch = epoch
                    patience_counter = 0
                    
                    # Save checkpoint
                    checkpoint_path = self.checkpoint_dir / "best_model.pt"
                    torch.save(self.model.state_dict(), checkpoint_path)
                else:
                    patience_counter += 1
                
                logger.info(
                    f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                    f"val_loss={val_metrics['val_loss']:.4f}, "
                    f"val_f1={val_metrics['val_f1']:.4f}"
                )
                
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            scheduler.step()
        
        return history
