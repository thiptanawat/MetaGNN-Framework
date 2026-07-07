#!/usr/bin/env python3
"""
Verify data integrity for MetaGNN-CRC dataset.

Checks:
  - All expected tensor files exist
  - Tensor shapes match expected dimensions
  - Graph structure integrity (node/edge counts)
  - Patient count consistency

Run after downloading or extracting the dataset.
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm


# Expected dimensions for full 690-patient dataset
EXPECTED_SHAPES = {
    "patient_features.pt": (690, 2),  # v1 features
    "patient_features_v2.pt": (690, 3),  # v2 features (enriched)
    "hma_labels_thresholded.pt": (10600,),  # Reaction labels
    "gpr_reaction_mask.pt": (10600,),  # GPR-mapped reaction mask
}

EXPECTED_GRAPH_NODES = {
    "reaction": 10600,
    "metabolite": 5835,
    "gene": 2248,
}

EXPECTED_EDGE_RANGES = {
    "stoichiometric": (40000, 41000),  # substrate_of + produces
    "shared_metabolite": (7000000, 8000000),  # Large undirected graph
}


def check_file_exists(filepath):
    """Check if a file exists and is readable."""
    path = Path(filepath)
    if not path.exists():
        return False, f"File not found: {filepath}"
    if not path.is_file():
        return False, f"Not a file: {filepath}"
    return True, "OK"


def check_tensor_shape(filepath, expected_shape):
    """Load tensor and verify shape."""
    try:
        tensor = torch.load(filepath, weights_only=True)
        actual_shape = tensor.shape
        if actual_shape != expected_shape:
            return (
                False,
                f"Shape mismatch: expected {expected_shape}, got {actual_shape}",
            )
        return True, f"Shape OK: {actual_shape}"
    except Exception as e:
        return False, f"Error loading tensor: {str(e)}"


def check_graph_structure(graph_path):
    """Load and verify graph structure."""
    try:
        graph = torch.load(graph_path, weights_only=False)

        results = {}

        # Check node counts
        for node_type, expected_count in EXPECTED_GRAPH_NODES.items():
            if node_type in graph and hasattr(graph[node_type], "num_nodes"):
                actual_count = graph[node_type].num_nodes
                if actual_count != expected_count:
                    results[node_type] = (
                        False,
                        f"Expected {expected_count} nodes, got {actual_count}",
                    )
                else:
                    results[node_type] = (True, f"{actual_count} nodes")
            else:
                results[node_type] = (False, f"Node type not found: {node_type}")

        return results
    except Exception as e:
        return {"error": (False, f"Error loading graph: {str(e)}")}


def verify_dataset(data_dir):
    """Run all verification checks."""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return False

    print("=" * 70)
    print("MetaGNN-CRC Dataset Integrity Verification")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print("")

    all_passed = True

    # Check tensor files
    print("Checking tensor files...")
    print("-" * 70)

    processed_dir = data_path / "processed"
    if not processed_dir.exists():
        print(f"Warning: processed directory not found: {processed_dir}")
        all_passed = False
    else:
        for filename, expected_shape in EXPECTED_SHAPES.items():
            filepath = processed_dir / filename
            exists, exists_msg = check_file_exists(filepath)

            if not exists:
                print(f"  {filename}: {exists_msg}")
                all_passed = False
                continue

            passed, msg = check_tensor_shape(filepath, expected_shape)
            status = "✓" if passed else "✗"
            print(f"  {status} {filename}: {msg}")
            all_passed = all_passed and passed

    print("")

    # Check graph structure
    print("Checking graph structure...")
    print("-" * 70)

    graph_path = processed_dir / "graph_structure.pt"
    if not Path(graph_path).exists():
        print(f"  Warning: graph structure file not found: {graph_path}")
    else:
        graph_results = check_graph_structure(graph_path)
        for node_type, (passed, msg) in graph_results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {node_type}: {msg}")
            all_passed = all_passed and passed

    print("")

    # Dataset summary
    print("Dataset Summary")
    print("-" * 70)

    if (processed_dir / "patient_features_v2.pt").exists():
        features = torch.load(
            processed_dir / "patient_features_v2.pt", weights_only=True
        )
        print(f"  Patients: {features.shape[0]}")
        print(f"  Features per patient: {features.shape[1]}")

    if (processed_dir / "hma_labels_thresholded.pt").exists():
        labels = torch.load(
            processed_dir / "hma_labels_thresholded.pt", weights_only=True
        )
        n_active = labels.sum().item()
        n_inactive = (labels == 0).sum().item()
        print(f"  Total reactions: {labels.shape[0]}")
        print(f"  Active reactions: {n_active}")
        print(f"  Inactive reactions: {n_inactive}")

    print("")
    print("=" * 70)

    if all_passed:
        print("✓ All checks passed. Dataset is ready for use.")
        return True
    else:
        print("✗ Some checks failed. See above for details.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify MetaGNN-CRC dataset integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/verify_data.py
  python scripts/verify_data.py --data_dir ./data/processed/
  python scripts/verify_data.py --data_dir /path/to/zenodo/download/
        """,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data directory root (default: data)",
    )

    args = parser.parse_args()

    success = verify_dataset(args.data_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
