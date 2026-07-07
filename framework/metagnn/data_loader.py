"""Data loading and patient tensor management."""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.loader import DataLoader


class PatientDataLoader:
    """Load and manage pre-computed patient reaction activity tensors.
    
    Supports creation of train/val/test splits with optional stratification,
    and integrates with PyG DataLoader.
    """

    def __init__(
        self,
        data_root: Path,
        dataset_name: str = "CRC",
    ):
        """Initialize patient data loader.
        
        Args:
            data_root: Root directory containing .pt files and metadata.
            dataset_name: Dataset name (CRC, BRCA, LUAD).
            
        Raises:
            FileNotFoundError: If data directory doesn't exist.
        """
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")
        
        # Load patient tensor files
        self.patient_files = sorted(self.data_root.glob("*.pt"))
        self.patient_ids = [f.stem for f in self.patient_files]
        self.num_patients = len(self.patient_files)

    def load_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load all patient tensors and labels.
        
        Returns:
            Tuple of (patient_tensors, labels) where:
            - patient_tensors: (n_patients, n_reactions)
            - labels: (n_patients,) binary class labels
        """
        tensors = []
        labels = []
        
        for patient_file in self.patient_files:
            data = torch.load(patient_file)
            
            # Extract tensor and label
            if isinstance(data, dict):
                tensor = data.get("reactions", data.get("scores"))
                label = data.get("label", 0)
            else:
                tensor = data
                label = 0
            
            tensors.append(tensor)
            labels.append(label)
        
        patient_tensors = torch.stack(tensors)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return patient_tensors, labels

    def create_train_val_test_split(
        self,
        patient_tensors: torch.Tensor,
        labels: torch.Tensor,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        stratify: bool = True,
        random_state: int = 42,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Create train/val/test splits with optional stratification.
        
        Args:
            patient_tensors: Patient reaction activity tensors.
            labels: Patient labels for stratification.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            stratify: Whether to stratify by labels.
            random_state: Random seed.
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices).
        """
        n_samples = patient_tensors.shape[0]
        test_ratio = 1.0 - train_ratio - val_ratio
        
        # First split: train + val vs test
        stratify_labels = labels if stratify else None
        train_val_idx, test_idx = train_test_split(
            np.arange(n_samples),
            test_size=test_ratio,
            stratify=stratify_labels,
            random_state=random_state,
        )
        
        # Second split: train vs val
        train_val_labels = labels[train_val_idx] if stratify else None
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio / (train_ratio + val_ratio),
            stratify=train_val_labels,
            random_state=random_state,
        )
        
        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    def create_stratified_folds(
        self,
        labels: torch.Tensor,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> List[Tuple[List[int], List[int]]]:
        """Create stratified k-fold cross-validation splits.
        
        Args:
            labels: Patient labels for stratification.
            n_splits: Number of folds.
            random_state: Random seed.
            
        Returns:
            List of (train_indices, val_indices) tuples for each fold.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = []
        
        for train_idx, val_idx in skf.split(
            np.zeros(labels.shape[0]), labels.numpy()
        ):
            folds.append((train_idx.tolist(), val_idx.tolist()))
        
        return folds

    def get_data_loaders(
        self,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyG DataLoaders for train/val/test.
        
        Args:
            train_indices: Indices for training set.
            val_indices: Indices for validation set.
            test_indices: Indices for test set.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of worker processes.
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        patient_tensors, labels = self.load_tensors()
        
        # Create simple data objects
        train_data = [
            (patient_tensors[i], labels[i]) for i in train_indices
        ]
        val_data = [
            (patient_tensors[i], labels[i]) for i in val_indices
        ]
        test_data = [
            (patient_tensors[i], labels[i]) for i in test_indices
        ]
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
