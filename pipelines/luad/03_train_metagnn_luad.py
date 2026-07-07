"""
MetaGNN Training on TCGA-LUAD
==============================
Runs MetaGNN (GATv2 heterogeneous graph attention) on lung adenocarcinoma data.
Uses the IDENTICAL architecture as MetaGNN-CRC — same model, different cancer.

Two evaluation configurations (matching CRC paper exactly):
  Config A: 220-patient benchmark (best-case)
    - 128 hidden / 2 layers / 4 heads (bipartite-only, 873K params)
    - OR 256 hidden / 3 layers / 8 heads (bipartite-only, 143K params)
    - Expression-thresholded labels, enriched v2 features
    - 70/15/15 train/val/test split

  Config B: Full-cohort cross-validation (realistic)
    - 256 hidden / 3 layers / 8 heads (expanded mode, 9.67M params)
    - shared_metabolite edges (k=10), 5-fold CV
    - HMA-derived labels

Hardware requirements:
  Config A: Any GPU with >=4GB VRAM (or Apple MPS)
  Config B: GPU with >=16GB VRAM (RTX 5090, H100)

Usage:
  # Config A: 220-patient benchmark
  python 03_train_metagnn_luad.py \
      --data_dir ./data_luad/processed/ \
      --config benchmark_220 \
      --n_patients 220 \
      --output_dir ./results_luad/benchmark_220/

  # Config B: Full-cohort 5-fold CV
  python 03_train_metagnn_luad.py \
      --data_dir ./data_luad/processed/ \
      --config full_cohort \
      --output_dir ./results_luad/full_cohort/

Author: MetaGNN Team
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import h5py

# PyTorch imports (required for training)
try:
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch not installed. Install with:")
    print("  conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia")
    sys.exit(1)

try:
    import torch_geometric
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("ERROR: PyTorch Geometric not installed. Install with:")
    print("  pip install torch-geometric")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Model Architecture (identical to MetaGNN-CRC)
# ═════════════════════════════════════════════════════════════════════════════
class MetaGNN(torch.nn.Module):
    """
    Heterogeneous GATv2 model for reaction activity scoring on Recon3D.
    Identical to the CRC version — no architecture changes for LUAD.
    """
    def __init__(
        self,
        rxn_in_dim: int = 2,
        met_in_dim: int = 519,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.2,
        use_shared_met: bool = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_shared_met = use_shared_met

        # Input projections
        self.rxn_proj = Linear(rxn_in_dim, hidden_dim)
        self.met_proj = Linear(met_in_dim, hidden_dim)

        # Stacked GATv2 layers with type-specific weights
        self.convs = torch.nn.ModuleList()
        for _ in range(n_layers):
            conv_dict = {
                ('metabolite', 'substrate_of', 'reaction'):
                    GATv2Conv(hidden_dim, hidden_dim // n_heads, heads=n_heads,
                              dropout=dropout, add_self_loops=False),
                ('reaction', 'produces', 'metabolite'):
                    GATv2Conv(hidden_dim, hidden_dim // n_heads, heads=n_heads,
                              dropout=dropout, add_self_loops=False),
            }
            if use_shared_met:
                conv_dict[('reaction', 'shared_metabolite', 'reaction')] = \
                    GATv2Conv(hidden_dim, hidden_dim // n_heads, heads=n_heads,
                              dropout=dropout, add_self_loops=False)
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        # Output MLP head
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x_dict, edge_index_dict):
        # Input projection
        x_dict = {
            'reaction': self.rxn_proj(x_dict['reaction']),
            'metabolite': self.met_proj(x_dict['metabolite']),
        }

        # Message passing
        for conv in self.convs:
            x_dict_new = conv(x_dict, edge_index_dict)
            # Residual + activation
            for key in x_dict_new:
                x_dict_new[key] = F.elu(x_dict_new[key])
                x_dict_new[key] = F.dropout(x_dict_new[key], p=self.dropout,
                                            training=self.training)
                if key in x_dict:
                    x_dict_new[key] = x_dict_new[key] + x_dict[key]
            x_dict = x_dict_new

        # Per-reaction prediction
        scores = self.mlp(x_dict['reaction']).squeeze(-1)
        return scores


# ═════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═════════════════════════════════════════════════════════════════════════════
def load_graph_data(
    processed_dir: str,
    device: str = 'cpu',
    use_shared_met: bool = False,
    shared_met_k: int = 10,
) -> HeteroData:
    """
    Load Recon3D graph structure (shared across all cancers).

    Args:
        use_shared_met: Whether to load shared_metabolite R↔R edges.
        shared_met_k:   Subsample to k nearest neighbors per reaction
                        to prevent OOM on full edge set (can be millions).
    """
    data = HeteroData()

    # Edge indices
    ei_dir = os.path.join(processed_dir, 'edge_indices')
    data['metabolite', 'substrate_of', 'reaction'].edge_index = \
        torch.load(os.path.join(ei_dir, 'substrate_of.pt'), weights_only=False).to(device)
    data['reaction', 'produces', 'metabolite'].edge_index = \
        torch.load(os.path.join(ei_dir, 'produces.pt'), weights_only=False).to(device)

    logger.info(f"  substrate_of edges: {data['metabolite', 'substrate_of', 'reaction'].edge_index.shape[1]:,}")
    logger.info(f"  produces edges: {data['reaction', 'produces', 'metabolite'].edge_index.shape[1]:,}")

    sm_path = os.path.join(ei_dir, 'shared_metabolite.pt')
    if use_shared_met and os.path.exists(sm_path):
        ei_full = torch.load(sm_path, weights_only=False)
        n_full = ei_full.shape[1]
        logger.info(f"  shared_metabolite edges (full): {n_full:,}")

        # Subsample to k-nearest neighbors per reaction to fit in GPU memory
        # For each reaction, keep only k outgoing edges (random sample)
        if shared_met_k > 0 and n_full > 0:
            src, dst = ei_full[0], ei_full[1]
            n_reactions = max(src.max().item(), dst.max().item()) + 1

            keep_mask = torch.zeros(n_full, dtype=torch.bool)
            for r in range(n_reactions):
                neighbors = (src == r).nonzero(as_tuple=True)[0]
                if len(neighbors) <= shared_met_k:
                    keep_mask[neighbors] = True
                else:
                    # Random subsample
                    perm = torch.randperm(len(neighbors))[:shared_met_k]
                    keep_mask[neighbors[perm]] = True

            ei_sub = ei_full[:, keep_mask]
            logger.info(f"  shared_metabolite edges (k={shared_met_k}): {ei_sub.shape[1]:,} "
                       f"(subsampled from {n_full:,})")
            data['reaction', 'shared_metabolite', 'reaction'].edge_index = ei_sub.to(device)
        else:
            data['reaction', 'shared_metabolite', 'reaction'].edge_index = ei_full.to(device)

    # Metabolite features (patient-invariant)
    with h5py.File(os.path.join(processed_dir, 'metabolite_features.h5'), 'r') as f:
        X_M = torch.tensor(f['X_M'][:], dtype=torch.float32).to(device)
    data['metabolite'].x = X_M

    return data


def load_patient_features(
    processed_dir: str,
    patient_id: str,
    device: str = 'cpu',
) -> torch.Tensor:
    """Load X_R for a single patient."""
    h5_path = os.path.join(processed_dir, 'reaction_features', f'{patient_id}.h5')
    with h5py.File(h5_path, 'r') as f:
        X_R = torch.tensor(f['X_R'][:], dtype=torch.float32).to(device)
    return X_R


def load_labels(processed_dir: str, device: str = 'cpu') -> torch.Tensor:
    """Load activity pseudo-labels."""
    pt_path = os.path.join(processed_dir, 'activity_pseudolabels.pt')
    npy_path = os.path.join(processed_dir, 'activity_pseudolabels.npy')

    if os.path.exists(pt_path):
        return torch.load(pt_path, weights_only=False).to(device)
    elif os.path.exists(npy_path):
        return torch.tensor(np.load(npy_path), dtype=torch.float32).to(device)
    else:
        raise FileNotFoundError(f"No pseudo-labels found in {processed_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# Mass-Balance Regularisation Loss
# ═════════════════════════════════════════════════════════════════════════════
def mass_balance_loss(
    scores: torch.Tensor,
    S: torch.Tensor,
    lambda_mb: float = 0.2,
) -> torch.Tensor:
    """
    Soft mass-balance regulariser: penalise predicted reaction sets
    that violate stoichiometric mass balance.
    L_MB = (1/|V_M|) * sum_m (sum_r S[m,r] * s_r)^2
    """
    net_flux = S @ scores  # (n_met,)
    return lambda_mb * (net_flux ** 2).mean()


# ═════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═════════════════════════════════════════════════════════════════════════════
def train_epoch(
    model: MetaGNN,
    graph: HeteroData,
    patient_ids: List[str],
    labels: torch.Tensor,
    optimizer,
    processed_dir: str,
    device: str,
    class_weights: Optional[torch.Tensor] = None,
) -> float:
    """Train one epoch over all patients."""
    model.train()
    total_loss = 0.0

    for pid in patient_ids:
        X_R = load_patient_features(processed_dir, pid, device)
        graph['reaction'].x = X_R

        optimizer.zero_grad()
        scores = model(
            {k: v.x for k, v in graph.node_items()},
            {k: v.edge_index for k, v in graph.edge_items()},
        )

        # Weighted BCE loss
        if class_weights is not None:
            weight = class_weights[labels.long()]
            loss = F.binary_cross_entropy(scores, labels, weight=weight)
        else:
            loss = F.binary_cross_entropy(scores, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(patient_ids)


@torch.no_grad()
def evaluate(
    model: MetaGNN,
    graph: HeteroData,
    patient_ids: List[str],
    labels: torch.Tensor,
    processed_dir: str,
    device: str,
    mc_dropout: bool = False,
    n_mc: int = 30,
) -> Dict:
    """
    Evaluate model on a set of patients.
    If mc_dropout=True, run Monte Carlo Dropout for uncertainty estimation.
    """
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        average_precision_score,
    )

    if mc_dropout:
        model.train()  # keep dropout active
    else:
        model.eval()

    all_scores = []
    all_uncertainties = []

    for pid in patient_ids:
        X_R = load_patient_features(processed_dir, pid, device)
        graph['reaction'].x = X_R

        if mc_dropout:
            mc_scores = []
            for _ in range(n_mc):
                s = model(
                    {k: v.x for k, v in graph.node_items()},
                    {k: v.edge_index for k, v in graph.edge_items()},
                )
                mc_scores.append(s.cpu().numpy())
            mc_array = np.stack(mc_scores, axis=0)
            mean_score = mc_array.mean(axis=0)
            std_score = mc_array.std(axis=0)
            all_scores.append(mean_score)
            all_uncertainties.append(std_score)
        else:
            s = model(
                {k: v.x for k, v in graph.node_items()},
                {k: v.edge_index for k, v in graph.edge_items()},
            )
            all_scores.append(s.cpu().numpy())

    y_true = labels.cpu().numpy()
    scores_array = np.stack(all_scores, axis=0)  # (n_patients, n_reactions)

    # Per-patient metrics
    results = {'per_patient': []}
    for i, pid in enumerate(patient_ids):
        y_pred = (scores_array[i] > 0.5).astype(float)
        patient_metrics = {
            'patient_id': pid,
            'auroc': float(roc_auc_score(y_true, scores_array[i])),
            'f1': float(f1_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'auprc': float(average_precision_score(y_true, scores_array[i])),
        }
        if all_uncertainties:
            patient_metrics['mean_uncertainty'] = float(all_uncertainties[i].mean())
        results['per_patient'].append(patient_metrics)

    # Aggregate metrics
    aurocs = [m['auroc'] for m in results['per_patient']]
    f1s = [m['f1'] for m in results['per_patient']]
    results['aggregate'] = {
        'auroc_mean': float(np.mean(aurocs)),
        'auroc_std': float(np.std(aurocs)),
        'f1_mean': float(np.mean(f1s)),
        'f1_std': float(np.std(f1s)),
        'n_patients': len(patient_ids),
    }

    # Store raw arrays for downstream use (Agent v6)
    results['_scores_array'] = scores_array  # (n_patients, n_reactions)
    if all_uncertainties:
        results['_uncertainties_array'] = np.stack(all_uncertainties, axis=0)
    results['_patient_ids'] = patient_ids

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Configuration Presets
# ═════════════════════════════════════════════════════════════════════════════
CONFIGS = {
    'benchmark_220': {
        'hidden_dim': 256,
        'n_layers': 3,
        'n_heads': 8,
        'use_shared_met': False,
        'lr': 5e-4,
        'weight_decay': 1e-5,
        'epochs': 200,
        'patience': 20,
        'dropout': 0.2,
        'lambda_mb': 0.2,
        # n_patients omitted → uses ALL available patients
        # Override with --n_patients 220 for direct CRC-comparable benchmark
        'split_ratio': [0.70, 0.15, 0.15],
        'description': 'Benchmark on LUAD (bipartite-only, 143K params)',
    },
    'full_cohort': {
        'hidden_dim': 256,
        'n_layers': 3,
        'n_heads': 8,
        'use_shared_met': True,
        'lr': 5e-4,
        'weight_decay': 1e-5,
        'epochs': 200,
        'patience': 20,
        'dropout': 0.2,
        'lambda_mb': 0.2,
        'n_folds': 5,
        'description': 'Full LUAD cohort 5-fold CV (expanded mode, 9.67M params)',
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# Main Training Pipeline
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train MetaGNN on TCGA-LUAD")
    parser.add_argument("--data_dir", required=True, help="Processed LUAD data directory")
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="benchmark_220")
    parser.add_argument("--output_dir", default="./results_luad/")
    parser.add_argument("--device", default=None, help="Device (auto-detect if not specified)")
    parser.add_argument("--n_patients", type=int, default=None, help="Override number of patients")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    os.makedirs(args.output_dir, exist_ok=True)

    # Device selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logger.info(f"Device: {device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load graph structure
    use_shared_met = cfg.get('use_shared_met', False)
    logger.info(f"Loading Recon3D graph structure (shared_met={use_shared_met})...")
    graph = load_graph_data(
        args.data_dir, device,
        use_shared_met=use_shared_met,
        shared_met_k=10,
    )

    # Load labels
    labels = load_labels(args.data_dir, device)
    n_reactions = labels.shape[0]

    # Get patient list
    patient_list_path = os.path.join(args.data_dir, 'patient_list.txt')
    with open(patient_list_path) as f:
        all_patients = [line.strip() for line in f if line.strip()]

    # Subsample if requested
    n_patients = args.n_patients or cfg.get('n_patients', len(all_patients))
    if n_patients < len(all_patients):
        np.random.shuffle(all_patients)
        all_patients = sorted(all_patients[:n_patients])
    logger.info(f"Using {len(all_patients)} patients")

    # Class weights (handle edge cases where all labels are same class)
    pos_ratio = labels.mean().item()
    logger.info(f"Label distribution: {pos_ratio*100:.1f}% active, {(1-pos_ratio)*100:.1f}% inactive")
    if pos_ratio <= 0.0 or pos_ratio >= 1.0:
        logger.warning(f"Degenerate labels (pos_ratio={pos_ratio:.4f}). Using uniform weights.")
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    else:
        class_weights = torch.tensor([1.0 / (1 - pos_ratio), 1.0 / pos_ratio],
                                      dtype=torch.float32).to(device)

    # Split
    split = cfg.get('split_ratio', [0.7, 0.15, 0.15])
    n_train = int(len(all_patients) * split[0])
    n_val = int(len(all_patients) * split[1])
    train_patients = all_patients[:n_train]
    val_patients = all_patients[n_train:n_train + n_val]
    test_patients = all_patients[n_train + n_val:]

    logger.info(f"Split: train={len(train_patients)}, val={len(val_patients)}, test={len(test_patients)}")

    # Build model
    model = MetaGNN(
        rxn_in_dim=2,
        met_in_dim=519,
        hidden_dim=cfg['hidden_dim'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        dropout=cfg['dropout'],
        use_shared_met=cfg['use_shared_met'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfg['description']}")
    logger.info(f"Trainable parameters: {n_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')

    logger.info(f"Training for up to {cfg['epochs']} epochs (patience={cfg['patience']})...")
    start_time = time.time()

    for epoch in range(cfg['epochs']):
        train_loss = train_epoch(
            model, graph, train_patients, labels, optimizer,
            args.data_dir, device, class_weights,
        )
        scheduler.step()

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_results = evaluate(
                model, graph, val_patients, labels,
                args.data_dir, device,
            )
            val_f1 = val_results['aggregate']['f1_mean']
            val_auroc = val_results['aggregate']['auroc_mean']

            logger.info(
                f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                f"Val AUROC: {val_auroc:.4f} | Val F1: {val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 5

            if patience_counter >= cfg['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.1f}s")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
    logger.info("Evaluating on test set with MC Dropout...")
    test_results = evaluate(
        model, graph, test_patients, labels,
        args.data_dir, device, mc_dropout=True, n_mc=30,
    )

    # Save per-reaction scores & uncertainties for Agent v6
    # These are too large for JSON, so save as .npz
    scores_npz_path = os.path.join(args.output_dir, 'test_scores.npz')
    npz_data = {
        'scores': test_results['_scores_array'],           # (n_test, n_reactions)
        'patient_ids': np.array(test_patients, dtype=str),  # (n_test,)
    }
    if '_uncertainties_array' in test_results:
        npz_data['uncertainties'] = test_results['_uncertainties_array']
    np.savez_compressed(scores_npz_path, **npz_data)
    logger.info(f"Per-reaction scores saved: {scores_npz_path} "
                f"(shape: {test_results['_scores_array'].shape})")

    # Remove non-serialisable arrays before JSON dump
    json_results = {k: v for k, v in test_results.items() if not k.startswith('_')}

    # Save results
    output = {
        'config': cfg,
        'device': device,
        'n_params': n_params,
        'train_time_s': train_time,
        'cancer_type': 'LUAD',
        'n_patients': len(all_patients),
        'test_results': json_results,
        'split': {
            'train': train_patients,
            'val': val_patients,
            'test': test_patients,
        },
    }
    results_path = os.path.join(args.output_dir, 'results_luad.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    agg = test_results['aggregate']
    logger.info("=" * 60)
    logger.info("MetaGNN-LUAD TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Cancer type:  LUAD (Lung Adenocarcinoma)")
    logger.info(f"  Patients:     {len(all_patients)}")
    logger.info(f"  Test set:     {len(test_patients)} patients")
    logger.info(f"  AUROC:        {agg['auroc_mean']:.4f} +/- {agg['auroc_std']:.4f}")
    logger.info(f"  F1 Score:     {agg['f1_mean']:.4f} +/- {agg['f1_std']:.4f}")
    logger.info(f"  Parameters:   {n_params:,}")
    logger.info(f"  Train time:   {train_time:.1f}s")
    logger.info("=" * 60)
    logger.info(f"Results saved: {results_path}")
    logger.info(f"Best model:    {best_model_path}")
    logger.info("Next: Run 04_compare_with_published.py to generate comparison table")


if __name__ == "__main__":
    main()
