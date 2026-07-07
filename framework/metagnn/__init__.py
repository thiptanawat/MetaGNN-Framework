"""MetaGNN: Metabolic Graph Neural Networks for Context-Aware Cancer Biomarker Discovery."""

__version__ = "1.0.0"
__author__ = "MetaGNN Contributors"
__description__ = "Graph neural networks for heterogeneous metabolic networks"

from metagnn.model import MetaGNNModel
from metagnn.graph_builder import ReconGraphBuilder
from metagnn.features import ReactionFeatures, MetaboliteFeatures
from metagnn.data_loader import PatientDataLoader
from metagnn.trainer import MetaGNNTrainer
from metagnn.evaluator import MetaGNNEvaluator
from metagnn.uncertainty import MCDropoutInference

__all__ = [
    "MetaGNNModel",
    "ReconGraphBuilder",
    "ReactionFeatures",
    "MetaboliteFeatures",
    "PatientDataLoader",
    "MetaGNNTrainer",
    "MetaGNNEvaluator",
    "MCDropoutInference",
]
