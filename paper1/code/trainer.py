#!/usr/bin/env python3
"""
Five-fold cross-validated graph model predictions, one row per patient and reaction
====================================================================================
Modified from exp3_kfold_cv.py and run_full_cohort_experiments.py to:
  1. Run 5-fold stratified CV on the FULL 624-patient HMA cohort
  2. Save **per-reaction predictions** for each fold's test patients
  3. Write the per-reaction CSV the downstream analysis reads:
     patient_id,reaction_id,metagnn_score,metagnn_sigma,hma_label,is_active_predicted
  4. Include MC Dropout uncertainty (metagnn_sigma) via T=30 stochastic passes

Output structure:
    results_kfold/
    ├── fold_0.csv       # Per-reaction predictions for fold 0 test patients
    ├── fold_1.csv
    ├── fold_2.csv
    ├── fold_3.csv
    ├── fold_4.csv
    ├── all_folds.csv    # Combined (all 624 patients, each predicted when held-out)
    ├── fold_summary.json
    └── fold_splits.json # Patient IDs per fold (for reproducibility)

Usage:
    # On RTX 5090 (WSL2):
    cd /path/to/MetaGNN
    python generate_kfold_predictions.py \
        --data_root /path/to/MetaGNN-CRC/data/processed \
        --output_dir results_kfold/

    # On MacBook M4 Max (MPS):
    python generate_kfold_predictions.py \
        --data_root /path/to/MetaGNN-CRC/data/processed \
        --output_dir results_kfold/ \
        --device mps

    # Quick test (1 fold only):
    python generate_kfold_predictions.py \
        --data_root /path/to/data --output_dir results_kfold/ --n_folds 1

Hardware estimates:
    RTX 5090 (32GB): ~55 min/fold × 5 folds ≈ 5 hours
    MacBook M4 Max:  ~2-4 hours/fold × 5 folds ≈ 10-20 hours
    (MPS is slower but works for Config A with 873K params)

Author: Thiptanawat Phongwattana & Jonathan H. Chan
"""

import os
import sys
import csv
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def _journals_root():
    env = os.environ.get("JOURNALS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "shared_data").exists() and (parent / "experiments").exists():
            return parent
    return here.parents[4]


def _resolve_label_path(data_root, label_override):
    data_root = Path(data_root)
    if not label_override:
        return data_root / "activity_pseudolabels.pt"

    override = Path(label_override).expanduser()
    candidates = [override] if override.is_absolute() else [
        data_root / override,
        _journals_root() / "shared_data" / "hma_labels" / override.name,
    ]
    if override.name == "hma_labels.pt":
        candidates.append(
            _journals_root() / "shared_data" / "hma_labels" / "activity_pseudolabels.pt"
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Label override '{label_override}' was requested but was not found. "
        f"Checked: {checked}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION (self-contained; no imports from metagnn_model needed)
# ═══════════════════════════════════════════════════════════════════════════════

class MCDropout(nn.Module):
    """Dropout that remains active during inference for MC uncertainty."""
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)  # Always on


class HGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.2,
                 residual=True, edge_types=None):
        super().__init__()
        self.residual = residual
        conv_dict = {
            ('metabolite', 'substrate_of', 'reaction'): GATv2Conv(
                in_channels['metabolite'], out_channels, heads=heads,
                dropout=dropout, add_self_loops=False, concat=False),
            ('reaction', 'produces', 'metabolite'): GATv2Conv(
                in_channels['reaction'], out_channels, heads=heads,
                dropout=dropout, add_self_loops=False, concat=False),
        }
        if edge_types is None or ('reaction', 'shared_metabolite', 'reaction') in edge_types:
            conv_dict[('reaction', 'shared_metabolite', 'reaction')] = GATv2Conv(
                in_channels['reaction'], out_channels, heads=heads,
                dropout=dropout, add_self_loops=True, concat=False)
        self.conv = HeteroConv(conv_dict, aggr='mean')
        self.norm_rxn = nn.LayerNorm(out_channels)
        self.norm_met = nn.LayerNorm(out_channels)
        self.proj_rxn = (nn.Linear(in_channels['reaction'], out_channels)
                         if in_channels['reaction'] != out_channels else nn.Identity())
        self.proj_met = (nn.Linear(in_channels['metabolite'], out_channels)
                         if in_channels['metabolite'] != out_channels else nn.Identity())
        self.mc_drop = MCDropout(p=dropout)

    def forward(self, x_dict, edge_index_dict):
        out = self.conv(x_dict, edge_index_dict)
        if self.residual:
            out['reaction'] = self.norm_rxn(
                F.elu(out['reaction']) + self.proj_rxn(x_dict['reaction']))
            out['metabolite'] = self.norm_met(
                F.elu(out['metabolite']) + self.proj_met(x_dict['metabolite']))
        else:
            out['reaction'] = self.norm_rxn(F.elu(out['reaction']))
            out['metabolite'] = self.norm_met(F.elu(out['metabolite']))
        out['reaction'] = self.mc_drop(out['reaction'])
        out['metabolite'] = self.mc_drop(out['metabolite'])
        return out


class MetaGNN(nn.Module):
    def __init__(self, rxn_in_dim=2, met_in_dim=519, hidden_dim=128,
                 n_layers=2, heads=4, dropout=0.2, edge_types=None):
        super().__init__()
        self.proj_rxn = nn.Sequential(
            Linear(rxn_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.proj_met = nn.Sequential(
            Linear(met_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.layers = nn.ModuleList([
            HGATLayer({'reaction': hidden_dim, 'metabolite': hidden_dim},
                      hidden_dim, heads=heads, dropout=dropout,
                      edge_types=edge_types)
            for _ in range(n_layers)
        ])
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ELU(),
            MCDropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
        )

    def forward(self, x_dict, edge_index_dict):
        h = {'reaction': self.proj_rxn(x_dict['reaction']),
             'metabolite': self.proj_met(x_dict['metabolite'])}
        for layer in self.layers:
            h = layer(h, edge_index_dict)
        return self.output_head(h['reaction']).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════════════════════

class MetaGNNLoss(nn.Module):
    def __init__(self, stoich_matrix=None, lambda_mb=0.2):
        super().__init__()
        if stoich_matrix is not None:
            self.register_buffer('S', stoich_matrix)
        else:
            self.S = None
        self.lambda_mb = lambda_mb
        self.bce = nn.BCELoss()

    def forward(self, s_r, y_r):
        l_bce = self.bce(s_r, y_r.float())
        if self.S is None or self.lambda_mb == 0:
            return l_bce
        n_rxn = self.S.shape[1]
        if s_r.shape[0] > n_rxn and s_r.shape[0] % n_rxn == 0:
            s_r_2d = s_r.view(-1, n_rxn)
            l_mb = torch.tensor(0.0, device=s_r.device)
            for i in range(s_r_2d.shape[0]):
                net_flux = self.S @ s_r_2d[i]
                l_mb = l_mb + net_flux.pow(2).mean()
            l_mb = l_mb / s_r_2d.shape[0]
        elif s_r.shape[0] == n_rxn:
            net_flux = self.S @ s_r
            l_mb = net_flux.pow(2).mean()
        else:
            l_mb = torch.tensor(0.0, device=s_r.device)
        return l_bce + self.lambda_mb * l_mb


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET (auto-detect MetaGNN-CRC .pt or cohort-220 pipeline .h5 format)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_data_format(data_root):
    pt_dir = os.path.join(data_root, 'patient_features')
    h5_dir = os.path.join(data_root, 'reaction_features')
    if os.path.isdir(pt_dir):
        return 'metagnn_crc'
    elif os.path.isdir(h5_dir):
        return 'cohort_220'
    else:
        raise FileNotFoundError(
            f"Cannot detect data format in {data_root}. "
            f"Expected 'patient_features/' (MetaGNN-CRC) or 'reaction_features/' (cohort-220 pipeline)."
        )


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, patient_ids, data_format='auto',
                 bipartite_only=False, feature_override_dir=None, label_override=None):
        self.data_root = data_root
        self.bipartite_only = bipartite_only
        self.feature_override_dir = feature_override_dir
        self.label_override = label_override
        if data_format == 'auto':
            data_format = detect_data_format(data_root)
        self.data_format = data_format
        self.patient_ids = list(patient_ids)

        if data_format == 'metagnn_crc':
            self._load_metagnn_crc()
        else:
            self._load_cohort_220()

    def _load_metagnn_crc(self):
        graph_path = os.path.join(self.data_root, 'graph_structure.pt')
        graph = torch.load(graph_path, weights_only=False)
        self.edge_index_dict = {}
        for rel in graph.edge_types:
            self.edge_index_dict[rel] = graph[rel].edge_index
        self.X_M = graph['metabolite'].x

        label_path = os.path.join(self.data_root, 'hma_labels_thresholded.pt')
        if not os.path.exists(label_path):
            label_path = os.path.join(self.data_root, 'activity_pseudolabels.pt')
        self.y_r = torch.load(label_path, weights_only=True)

        self.feat_dir = os.path.join(self.data_root, 'patient_features')
        self.feat_ext = '.pt'

    def _load_cohort_220(self):
        with h5py.File(os.path.join(self.data_root, 'metabolite_features.h5'), 'r') as f:
            self.X_M = torch.tensor(f['X_M'][:], dtype=torch.float32)

        edge_dir = os.path.join(self.data_root, 'edge_indices')
        self.edge_index_dict = {
            ('metabolite', 'substrate_of', 'reaction'):
                torch.load(os.path.join(edge_dir, 'substrate_of.pt'), weights_only=True),
            ('reaction', 'produces', 'metabolite'):
                torch.load(os.path.join(edge_dir, 'produces.pt'), weights_only=True),
        }
        shared_path = os.path.join(edge_dir, 'shared_metabolite.pt')
        if os.path.exists(shared_path) and not self.bipartite_only:
            shared_ei = torch.load(shared_path, weights_only=True)
            MAX_K = 10
            n_nodes = shared_ei.max().item() + 1
            if shared_ei.shape[1] > MAX_K * n_nodes:
                n_orig = shared_ei.shape[1]
                perm = torch.randperm(n_orig)
                sort_idx = shared_ei[0][perm].argsort(stable=True)
                src_sorted = shared_ei[0][perm][sort_idx]
                _, counts = torch.unique_consecutive(src_sorted, return_counts=True)
                offsets = counts.cumsum(0)
                starts = torch.zeros_like(offsets)
                starts[1:] = offsets[:-1]
                group_id = torch.repeat_interleave(torch.arange(len(counts)), counts)
                rank = torch.arange(n_orig) - starts[group_id]
                kept_indices = perm[sort_idx[rank < MAX_K]]
                shared_ei = shared_ei[:, kept_indices]
                logger.info(f"  Sparsified shared_metabolite: {n_orig:,} -> {shared_ei.shape[1]:,} edges")
            self.edge_index_dict[('reaction', 'shared_metabolite', 'reaction')] = shared_ei

        # Label loading with optional override
        label_path = _resolve_label_path(self.data_root, self.label_override)
        if self.label_override:
            logger.info(f"  Using label override: {label_path}")
        self.y_r = torch.load(label_path, weights_only=True)

        # Feature directory with optional override (for apple-to-apple ablation)
        if self.feature_override_dir and os.path.isdir(self.feature_override_dir):
            self.feat_dir = self.feature_override_dir
            logger.info(f"  Using feature override dir: {self.feat_dir}")
        else:
            self.feat_dir = os.path.join(self.data_root, 'reaction_features')
        self.feat_ext = '.h5'

        if self.bipartite_only:
            logger.info("  Bipartite-only mode: shared_metabolite edges excluded")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        if self.feat_ext == '.pt':
            X_R = torch.load(
                os.path.join(self.feat_dir, f'{pid}.pt'), weights_only=True)
            if not isinstance(X_R, torch.Tensor):
                X_R = torch.tensor(X_R, dtype=torch.float32)
            X_R = X_R.float()
        else:
            with h5py.File(os.path.join(self.feat_dir, f'{pid}.h5'), 'r') as f:
                X_R = torch.tensor(f['X_R'][:], dtype=torch.float32)

        data = HeteroData()
        data['reaction'].x = X_R
        data['metabolite'].x = self.X_M
        data['reaction'].y = self.y_r
        data['reaction'].pid = pid

        for rel, ei in self.edge_index_dict.items():
            src_type, rel_type, dst_type = rel
            data[src_type, rel_type, dst_type].edge_index = ei

        return data


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD PATIENT IDS
# ═══════════════════════════════════════════════════════════════════════════════

def load_patient_ids(data_root, data_format, stratify_by='msi_status'):
    """Load all patient IDs + stratification labels.

    Parameters
    ----------
    stratify_by : str
        Column in clinical_metadata.tsv to use for stratified subsampling and
        K-fold splits. Defaults to 'msi_status' (220-cohort protocol). For
        the 624 full cohort where 'msi_status' is sparsely populated, pass
        'project' (COAD vs READ, fully populated) or another populated
        categorical column. Empty strings / NaN values are folded into a
        single 'Unknown' class.
    """
    if data_format == 'metagnn_crc':
        meta_path = os.path.join(data_root, 'metadata.csv')
        if os.path.exists(meta_path):
            meta_df = pd.read_csv(meta_path)
            for col in ['tcga_barcode', 'patient_id', 'sample_id']:
                if col in meta_df.columns:
                    id_col = col
                    break
            else:
                id_col = meta_df.columns[0]
            patient_ids = meta_df[id_col].tolist()
            strat = None
            # Prefer caller's choice; fall back to legacy priority list.
            candidate_cols = [stratify_by, 'msi_status', 'tumor_stage', 'stage']
            # Deduplicate while preserving order
            seen = set()
            candidate_cols = [c for c in candidate_cols
                              if not (c in seen or seen.add(c))]
            for col in candidate_cols:
                if col in meta_df.columns:
                    strat = meta_df[col].fillna('Unknown').tolist()
                    logger.info(f"Stratification column: '{col}'")
                    break
        else:
            feat_dir = os.path.join(data_root, 'patient_features')
            patient_ids = sorted([f.replace('.pt', '') for f in os.listdir(feat_dir)
                                  if f.endswith('.pt')])
            strat = None
    else:
        meta_path = os.path.join(data_root, 'clinical_metadata.tsv')
        meta_df = pd.read_csv(meta_path, sep='\t')
        patient_ids = meta_df['tcga_barcode'].tolist()
        if stratify_by not in meta_df.columns:
            logger.warning(
                f"stratify_by='{stratify_by}' not in clinical_metadata.tsv "
                f"columns {list(meta_df.columns)}; falling back to 'msi_status'."
            )
            stratify_by = 'msi_status'
        # Treat blank strings as NaN so they get folded into 'Unknown' too.
        col_series = meta_df[stratify_by].replace(r'^\s*$', pd.NA, regex=True)
        strat = col_series.fillna('Unknown').tolist()
        logger.info(f"Stratification column: '{stratify_by}'")
        # Diagnostics: warn if effectively single-class.
        n_classes = len(set(strat))
        n_unknown = sum(1 for s in strat if s == 'Unknown')
        if n_classes == 1:
            logger.warning(
                f"Stratification column '{stratify_by}' has only one unique "
                f"value across {len(strat)} patients (all '{strat[0]}'); "
                f"stratified subsampling degenerates to simple random sampling. "
                f"Consider --stratify_by project."
            )
        elif n_unknown > 0.5 * len(strat):
            logger.warning(
                f"Stratification column '{stratify_by}' is sparsely populated "
                f"({n_unknown}/{len(strat)} 'Unknown'); stratification quality "
                f"will be limited."
            )

    if strat is None:
        strat = ['A'] * len(patient_ids)

    logger.info(f"Loaded {len(patient_ids)} patients ({data_format} format)")
    return patient_ids, strat


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_pred, y_true, threshold=0.15):
    y_bin = (y_pred >= threshold).astype(int)
    f1 = f1_score(y_true, y_bin, zero_division=0)
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_pred)
    else:
        auroc = float('nan')
    return {'F1': f1, 'AUROC': auroc}


# ═══════════════════════════════════════════════════════════════════════════════
# MC DROPOUT INFERENCE: produces (mean_score, std_score) per reaction
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def mc_dropout_predict(model, data, device, T=30):
    """Run T stochastic forward passes and return mean + std per reaction.

    Args:
        model:  MetaGNN model (MCDropout layers always active)
        data:   HeteroData for one patient
        device: torch device
        T:      number of MC samples (default 30, same as paper)

    Returns:
        scores_mean: np.array [n_rxn], mean predicted score per reaction
        scores_std:  np.array [n_rxn], MC uncertainty (sigma) per reaction
    """
    data = data.to(device)
    samples = []
    for _ in range(T):
        s_r = model(
            x_dict={'reaction': data['reaction'].x,
                    'metabolite': data['metabolite'].x},
            edge_index_dict={rel: data[rel].edge_index
                            for rel in data.edge_types},
        )
        samples.append(s_r.cpu().numpy())

    samples = np.stack(samples, axis=0)  # [T, n_rxn]
    return samples.mean(axis=0), samples.std(axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE FOLD
# ═══════════════════════════════════════════════════════════════════════════════

def train_fold(cfg, fold, train_ids, val_ids, device):
    """Train MetaGNN on one fold. Returns trained model."""
    torch.manual_seed(cfg['seed'] + fold)
    np.random.seed(cfg['seed'] + fold)

    train_ds = UnifiedDataset(cfg['data_root'], train_ids, cfg['data_format'],
                              bipartite_only=cfg.get('bipartite_only', False),
                              feature_override_dir=cfg.get('feature_override_dir'),
                              label_override=cfg.get('label_override'))
    val_ds = UnifiedDataset(cfg['data_root'], val_ids, cfg['data_format'],
                            bipartite_only=cfg.get('bipartite_only', False),
                            feature_override_dir=cfg.get('feature_override_dir'),
                            label_override=cfg.get('label_override'))

    train_loader = PyGLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = PyGLoader(val_ds, batch_size=1, shuffle=False)

    # Determine input dimensions from first sample
    sample = train_ds[0]
    rxn_dim = sample['reaction'].x.shape[1]
    met_dim = sample['metabolite'].x.shape[1]
    edge_types = list(sample.edge_types) if hasattr(sample, 'edge_types') else None

    logger.info(f"  Reaction input dim: {rxn_dim}, Metabolite input dim: {met_dim}")

    model = MetaGNN(
        rxn_in_dim=rxn_dim,
        met_in_dim=met_dim,
        hidden_dim=cfg['hidden_dim'],
        n_layers=cfg['n_layers'],
        heads=cfg['heads'],
        dropout=cfg['dropout'],
        edge_types=edge_types,
    ).to(device)

    # Load stoichiometric matrix for mass-balance regularization
    stoich_path = os.path.join(cfg['data_root'], 'recon3d_stoich.h5')
    if os.path.exists(stoich_path):
        with h5py.File(stoich_path, 'r') as f:
            S = torch.tensor(f['S'][:], dtype=torch.float32).to(device)
        criterion = MetaGNNLoss(stoich_matrix=S, lambda_mb=cfg['lambda_mb'])
    else:
        logger.warning("  recon3d_stoich.h5 not found; training without mass-balance loss")
        criterion = MetaGNNLoss(stoich_matrix=None, lambda_mb=0)

    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['n_epochs'])

    best_val_f1 = 0.0
    patience_counter = 0
    save_path = os.path.join(cfg['output_dir'], f'model_fold_{fold}.pt')

    t0 = time.time()
    for epoch in range(1, cfg['n_epochs'] + 1):
        # Train
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            s_r = model(
                x_dict={'reaction': batch['reaction'].x,
                        'metabolite': batch['metabolite'].x},
                edge_index_dict={rel: batch[rel].edge_index
                                for rel in batch.edge_types},
            )
            loss = criterion(s_r, batch['reaction'].y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                s_r = model(
                    x_dict={'reaction': batch['reaction'].x,
                            'metabolite': batch['metabolite'].x},
                    edge_index_dict={rel: batch[rel].edge_index
                                    for rel in batch.edge_types},
                )
                all_pred.append(s_r.cpu().numpy())
                all_true.append(batch['reaction'].y.cpu().numpy())

        y_pred = np.concatenate(all_pred)
        y_true = np.concatenate(all_true)
        val_metrics = compute_metrics(y_pred, y_true, cfg['threshold'])

        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                f"  [Fold {fold}] Epoch {epoch:3d}  "
                f"loss={total_loss/len(train_loader):.4f}  "
                f"val_F1={val_metrics['F1']:.4f}  "
                f"val_AUROC={val_metrics['AUROC']:.4f}"
            )

        if val_metrics['F1'] > best_val_f1:
            best_val_f1 = val_metrics['F1']
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                logger.info(f"  [Fold {fold}] Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    logger.info(f"  [Fold {fold}] Training complete in {elapsed:.0f}s (best val F1: {best_val_f1:.4f})")

    # Reload best model
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE PREDICTIONS FOR TEST PATIENTS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_fold_predictions(model, test_ids, cfg, device, fold, mc_T=30):
    """Generate per-reaction predictions with MC Dropout uncertainty.

    Returns list of dicts, one per reaction per patient.
    """
    test_ds = UnifiedDataset(cfg['data_root'], test_ids, cfg['data_format'],
                             bipartite_only=cfg.get('bipartite_only', False),
                             feature_override_dir=cfg.get('feature_override_dir'),
                             label_override=cfg.get('label_override'))
    n_rxn = test_ds.y_r.shape[0]

    rows = []
    for i in range(len(test_ds)):
        patient_id = test_ids[i]
        data = test_ds[i]

        # MC Dropout inference
        scores_mean, scores_std = mc_dropout_predict(model, data, device, T=mc_T)
        y_true = data['reaction'].y.cpu().numpy()

        for rxn_idx in range(n_rxn):
            rows.append({
                'patient_id': patient_id,
                'reaction_id': f'R_{rxn_idx:05d}',
                'metagnn_score': float(scores_mean[rxn_idx]),
                'metagnn_sigma': float(scores_std[rxn_idx]),
                'hma_label': int(y_true[rxn_idx]),
                'is_active_predicted': int(scores_mean[rxn_idx] >= cfg['threshold']),
            })

        logger.info(f"  [Fold {fold}] Patient {patient_id}: {n_rxn} reactions, "
                     f"mean_score={scores_mean.mean():.3f}, mean_sigma={scores_std.mean():.4f}")

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Five-fold cross-validated graph model predictions, per patient and reaction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 5-fold CV on 624-patient dataset (RTX 5090, ~5 hours):
  python generate_kfold_predictions.py --data_root ./data/processed

  # On MacBook M4 Max:
  python generate_kfold_predictions.py --data_root ./data/processed --device mps

  # Quick test (1 fold only):
  python generate_kfold_predictions.py --data_root ./data --n_folds 1
        """
    )
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to data directory (MetaGNN-CRC or cohort-220 pipeline format)')
    parser.add_argument('--output_dir', type=str, default='results_kfold',
                        help='Output directory for fold CSVs')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--val_frac', type=float, default=0.15,
                        help='Fraction of training fold for early stopping')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, mps, or cpu')
    parser.add_argument('--mc_T', type=int, default=30,
                        help='MC Dropout samples for uncertainty (default: 30)')
    parser.add_argument('--seed', type=int, default=2024)
    # Architecture (Config A default; fits on both a 5090 and an M4 Max)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--threshold', type=float, default=0.15)
    parser.add_argument('--lambda_mb', type=float, default=0.2)
    # Resume support
    parser.add_argument('--resume_from_fold', type=int, default=0,
                        help='Skip folds before this index (for resuming)')
    # Apple-to-apple ablation support
    parser.add_argument('--bipartite_only', action='store_true',
                        help='Use only bipartite edges (substrate_of + produces), '
                             'ignoring shared_metabolite edges even if present')
    parser.add_argument('--feature_override_dir', type=str, default=None,
                        help='Override reaction feature directory (e.g., enriched 3D features). '
                             'Must contain per-patient .h5 files with X_R dataset')
    parser.add_argument('--label_override', type=str, default=None,
                        help='Override label file name (e.g., hma_labels_union.pt)')
    # Multi-seed robustness support (advisor request: "change the seed for the 200 subject subsets")
    parser.add_argument('--subsample_size', type=int, default=0,
                        help='If >0, stratified-subsample this many patients from the cohort '
                             'before K-fold splitting. Enables the 200-from-220 robustness '
                             'analysis: each --seed yields a different 200-subject subset.')
    parser.add_argument('--stratify_by', type=str, default='msi_status',
                        help='Clinical metadata column for stratified subsampling and K-fold '
                             'splits. Defaults to "msi_status" (220-cohort protocol); use '
                             '"project" for the 624 full cohort where MSI status is sparse.')
    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Detect data format
    data_format = detect_data_format(args.data_root)
    logger.info(f"Data format: {data_format}")

    # Load patient IDs
    patient_ids, strat_labels = load_patient_ids(
        args.data_root, data_format, stratify_by=args.stratify_by,
    )
    patient_ids = np.array(patient_ids)
    strat_labels = np.array(strat_labels)

    # ─── Multi-seed robustness: stratified subsample ─────────────────────────
    # Advisor request: "change the seed for the 200 subject subsets". Each --seed
    # picks a different stratified subsample of size --subsample_size (e.g. 200 of 220);
    # K-fold CV then runs on that subset.
    if args.subsample_size > 0 and args.subsample_size < len(patient_ids):
        rng = np.random.default_rng(args.seed)
        n_keep = args.subsample_size
        # Stratified subsample: keep class proportions balanced across the subset.
        unique, counts = np.unique(strat_labels, return_counts=True)
        proportions = counts / counts.sum()
        keep_idx = []
        for cls, p in zip(unique, proportions):
            cls_idx = np.where(strat_labels == cls)[0]
            # Allocate proportionally; take at least 1 per class if present.
            n_cls = max(1, int(round(p * n_keep)))
            n_cls = min(n_cls, len(cls_idx))
            chosen = rng.choice(cls_idx, size=n_cls, replace=False)
            keep_idx.extend(chosen.tolist())
        # Correct rounding drift so we land on exactly n_keep.
        keep_idx = np.array(sorted(set(keep_idx)))
        if len(keep_idx) > n_keep:
            keep_idx = rng.choice(keep_idx, size=n_keep, replace=False)
        elif len(keep_idx) < n_keep:
            remaining = np.setdiff1d(np.arange(len(patient_ids)), keep_idx)
            extra = rng.choice(remaining, size=n_keep - len(keep_idx), replace=False)
            keep_idx = np.concatenate([keep_idx, extra])
        keep_idx = np.sort(keep_idx)
        n_orig = len(patient_ids)
        patient_ids = patient_ids[keep_idx]
        strat_labels = strat_labels[keep_idx]
        logger.info(
            f"Stratified subsample: {n_orig} → {len(patient_ids)} patients "
            f"(seed={args.seed})"
        )
    elif args.subsample_size > 0:
        logger.info(
            f"--subsample_size={args.subsample_size} >= cohort size "
            f"{len(patient_ids)}; no subsampling applied."
        )

    cfg = {
        'data_root': args.data_root,
        'output_dir': args.output_dir,
        'data_format': data_format,
        'seed': args.seed,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'heads': args.heads,
        'dropout': args.dropout,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'n_epochs': args.n_epochs,
        'patience': args.patience,
        'threshold': args.threshold,
        'lambda_mb': args.lambda_mb,
        # Apple-to-apple ablation flags
        'bipartite_only': getattr(args, 'bipartite_only', False),
        'feature_override_dir': getattr(args, 'feature_override_dir', None),
        'label_override': getattr(args, 'label_override', None),
        'subsample_size': getattr(args, 'subsample_size', 0),
        'n_patients_after_subsample': int(len(patient_ids)),
    }

    if cfg['bipartite_only']:
        logger.info("ABLATION MODE: bipartite-only (shared_metabolite edges excluded)")
    if cfg['feature_override_dir']:
        logger.info(f"ABLATION MODE: feature override from {cfg['feature_override_dir']}")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    fold_splits = {}
    fold_metrics = []
    all_rows = []
    total_start = time.time()

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(patient_ids, strat_labels)):
        test_ids = patient_ids[test_idx].tolist()
        trainval_ids = patient_ids[trainval_idx]
        trainval_strat = strat_labels[trainval_idx]

        # Split trainval → train + val
        train_ids, val_ids = train_test_split(
            trainval_ids, test_size=args.val_frac,
            stratify=trainval_strat, random_state=args.seed + fold_idx,
        )
        train_ids = train_ids.tolist()
        val_ids = val_ids.tolist()

        fold_splits[f'fold_{fold_idx}'] = {
            'train': train_ids, 'val': val_ids, 'test': test_ids
        }

        logger.info(f"\n{'═'*60}")
        logger.info(f"  FOLD {fold_idx} / {args.n_folds - 1}  "
                     f"(train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)})")
        logger.info(f"{'═'*60}")

        fold_csv = os.path.join(args.output_dir, f'fold_{fold_idx}.csv')

        # Resume support: skip already-completed folds
        if fold_idx < args.resume_from_fold:
            if os.path.exists(fold_csv):
                logger.info(f"  Skipping fold {fold_idx} (already exists, resuming)")
                # Load existing rows for all_folds.csv
                with open(fold_csv) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_rows.append(row)
                continue
            else:
                logger.warning(f"  Fold {fold_idx} CSV not found despite resume, retraining")

        # Train
        model, train_time = train_fold(cfg, fold_idx, train_ids, val_ids, device)

        # Generate predictions with MC Dropout
        logger.info(f"  Generating predictions for {len(test_ids)} test patients (MC T={args.mc_T})...")
        rows = generate_fold_predictions(model, test_ids, cfg, device, fold_idx, mc_T=args.mc_T)

        # Save fold CSV
        fieldnames = ['patient_id', 'reaction_id', 'metagnn_score', 'metagnn_sigma',
                      'hma_label', 'is_active_predicted']
        with open(fold_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"  Saved {len(rows)} predictions to {fold_csv}")

        # Compute fold-level metrics
        scores = np.array([r['metagnn_score'] for r in rows])
        labels = np.array([r['hma_label'] for r in rows])
        sigmas = np.array([r['metagnn_sigma'] for r in rows])
        metrics = compute_metrics(scores, labels, cfg['threshold'])
        n_boundary = np.sum((scores >= 0.3) & (scores <= 0.7))
        metrics['n_test_patients'] = len(test_ids)
        metrics['n_reactions'] = len(rows)
        metrics['n_boundary'] = int(n_boundary)
        metrics['boundary_pct'] = float(n_boundary / len(rows) * 100)
        metrics['mean_sigma'] = float(sigmas.mean())
        metrics['training_time_s'] = train_time
        metrics['fold'] = fold_idx
        fold_metrics.append(metrics)

        logger.info(f"  [Fold {fold_idx}] AUROC={metrics['AUROC']:.4f}  F1={metrics['F1']:.4f}  "
                     f"Boundary={n_boundary} ({metrics['boundary_pct']:.1f}%)")

        all_rows.extend(rows)

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ─── Save combined CSV ───────────────────────────────────────────────────
    all_csv = os.path.join(args.output_dir, 'all_folds.csv')
    with open(all_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info(f"\nSaved {len(all_rows)} total predictions to {all_csv}")

    # ─── Save splits ─────────────────────────────────────────────────────────
    splits_path = os.path.join(args.output_dir, 'fold_splits.json')
    with open(splits_path, 'w') as f:
        json.dump(fold_splits, f, indent=2)

    # ─── Save summary ────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_folds': args.n_folds,
        'n_patients_total': len(patient_ids),
        'n_reactions_total': len(all_rows),
        'total_time_s': total_time,
        'device': str(device),
        'config': {k: v for k, v in cfg.items()
                   if isinstance(v, (int, float, str, bool))},
        'per_fold': fold_metrics,
        'aggregate': {},
    }

    if fold_metrics:
        for key in ['F1', 'AUROC']:
            vals = [m[key] for m in fold_metrics if not np.isnan(m.get(key, float('nan')))]
            if vals:
                summary['aggregate'][f'{key}_mean'] = float(np.mean(vals))
                summary['aggregate'][f'{key}_std'] = float(np.std(vals))
        boundary_counts = [m['n_boundary'] for m in fold_metrics]
        summary['aggregate']['total_boundary'] = int(sum(boundary_counts))
        summary['aggregate']['mean_boundary_pct'] = float(np.mean(
            [m['boundary_pct'] for m in fold_metrics]))

    summary_path = os.path.join(args.output_dir, 'fold_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # ─── Print results ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"K-FOLD PREDICTION GENERATION COMPLETE")
    print(f"{'═'*60}")
    print(f"  Patients: {len(patient_ids)}")
    print(f"  Folds:    {args.n_folds}")
    print(f"  Total predictions: {len(all_rows)}")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print()
    print(f"  {'Fold':<6} {'AUROC':>8} {'F1':>8} {'Boundary':>10} {'Time':>8}")
    print(f"  {'-'*44}")
    for m in fold_metrics:
        print(f"  {m['fold']:<6} {m['AUROC']:.4f}  {m['F1']:.4f}  "
              f"{m['n_boundary']:>6} ({m['boundary_pct']:.0f}%)  {m['training_time_s']:.0f}s")
    if fold_metrics:
        agg = summary['aggregate']
        print(f"  {'-'*44}")
        print(f"  {'Mean':<6} {agg.get('AUROC_mean',0):.4f}  {agg.get('F1_mean',0):.4f}  "
              f"{agg.get('total_boundary',0):>6} total")

    print(f"\n  Output files:")
    print(f"    {all_csv}")
    print(f"    {splits_path}")
    print(f"    {summary_path}")
    print(f"\n  Next: the per-cell analyses read {args.output_dir} directly; see")
    print(f"    docs/REPRODUCTION_PAPER1.md for the command that consumes it.")
    print(f"{'═'*60}")


if __name__ == '__main__':
    main()
