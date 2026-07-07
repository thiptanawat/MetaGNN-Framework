# MetaGNN: Heterogeneous Graph Attention Network for Topology-Aware Metabolic Reaction Activity Scoring

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXXX (add after minting).svg)](https://doi.org/10.5281/zenodo.XXXXXXXX (add after minting))

MetaGNN applies Graph Attention Networks (GATv2) to heterogeneous metabolic networks for predicting reaction activity across cancer patients. The framework integrates enzyme expression, metabolite properties, and reaction topology to score metabolic reactions in a context-aware manner. Multi-cancer training enables transfer learning across colorectal, breast, and lung adenocarcinoma cohorts.

**Code Archive (Zenodo):** [https://doi.org/10.5281/zenodo.XXXXXXXX (add after minting)](https://doi.org/10.5281/zenodo.XXXXXXXX (add after minting))
**Companion Dataset:** archived on Zenodo — add the data-record DOI here after minting.

## Key Features

- **GATv2 Architecture**: Multi-head graph attention mechanism for adaptive aggregation of heterogeneous metabolic nodes
- **Multi-Cancer Support**: Train on CRC, BRCA, LUAD independently or in combination
- **MC Dropout Uncertainty**: Bayesian-style confidence estimates via Monte Carlo dropout during inference
- **Agentic Validation**: Six agent variants (v1-v6) for LLM-driven reaction validation using knowledge graphs and retrieval-augmented generation
- **Reproducible Benchmarks**: Pre-configured setups for 220-patient and 624-patient CRC cohorts

## Architecture Overview

MetaGNN operates on a bipartite metabolic graph with reaction and metabolite nodes. Reaction nodes aggregate signals from connected metabolites using GATv2 attention; metabolite nodes similarly aggregate from reactions. Patient-specific enzyme expression and metabolite abundance provide node features. The framework includes Figure 2 from the paper depicting the full graph structure and information flow.

## Quick Start

Set up the conda environment and run a standard benchmark:

```bash
conda env create -f environment.yml
conda activate metagnn
bash run.sh configs/crc_220.yaml
```

This executes the full pipeline: data acquisition, graph construction, model training, evaluation, and figure generation.

## Installation

### Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate metagnn
```

### Pip

```bash
pip install -r requirements.txt
pip install -e .
```

To include optional agent dependencies:

```bash
pip install -r requirements.txt
pip install -e ".[agent]"
```

## Usage

### Configuration-Based Training

MetaGNN uses YAML configurations to specify cohort, features, and hyperparameters. Run with any config file:

```bash
bash run.sh configs/brca.yaml      # BRCA cohort
bash run.sh configs/luad.yaml      # LUAD cohort
bash run.sh configs/crc_624.yaml   # Larger CRC cohort
```

### Direct Python Usage

```python
from metagnn import MetaGNNModel, PatientDataLoader, MetaGNNTrainer
import yaml

# Load configuration
with open('configs/crc_220.yaml') as f:
    config = yaml.safe_load(f)

# Load data
loader = PatientDataLoader(config['data_root'])
train_graphs, train_labels = loader.load_split('train')

# Initialize model
model = MetaGNNModel(
    hidden_dim=config['hidden_dim'],
    num_layers=config['layers'],
    num_heads=config['heads']
)

# Train
trainer = MetaGNNTrainer(model, config['learning_rate'])
trainer.fit(train_graphs, train_labels, epochs=config['epochs'])
```

### Inference with Uncertainty

```python
from metagnn import MCDropoutInference

# Generate predictions with dropout-based uncertainty
inference = MCDropoutInference(model, n_passes=50)
predictions, uncertainties = inference.predict(test_graphs)
```

## Module Overview

| Module | Purpose |
|--------|---------|
| **model.py** | GATv2-based reaction activity prediction network |
| **graph_builder.py** | Heterogeneous graph construction from metabolic models |
| **features.py** | Reaction and metabolite feature engineering |
| **data_loader.py** | Patient-specific graph and label loading |
| **trainer.py** | Training loop with early stopping and validation |
| **evaluator.py** | Performance metrics (AUROC, AUPR, F1) and visualization |
| **uncertainty.py** | Monte Carlo dropout inference for confidence estimates |

## Agent-Based Validation

Six agent variants implement increasingly sophisticated validation logic. Run individual versions or the full pipeline:

```bash
# Individual agent versions
python code/agent/v1_bare_llm.py              # Bare LLM queries
python code/agent/v2_enriched_prompt.py       # Context-enriched prompts
python code/agent/v3_advocate_resolver.py     # Advocate-resolver pattern
python code/agent/v4_patient_rag.py           # Patient-specific RAG
python code/agent/v5_faiss_retrieval.py       # FAISS vector retrieval
python code/agent/v6_langgraph_kg.py          # LangGraph knowledge graph

# Full agent pipeline
python code/agent/run_agent.py configs/agent.yaml
```

Agent outputs include per-reaction validation scores and confidence estimates based on supporting evidence.

## Configuration Reference

Common configuration parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `cancer_type` | str | CRC, BRCA, or LUAD |
| `hidden_dim` | int | GATv2 hidden dimension (default: 256) |
| `layers` | int | Number of graph attention layers (default: 3) |
| `heads` | int | Attention heads per layer (default: 8) |
| `epochs` | int | Maximum training epochs (default: 200) |
| `learning_rate` | float | Adam learning rate (default: 1e-3) |
| `batch_size` | int | Training batch size (default: 32) |
| `patience` | int | Early stopping patience in epochs (default: 15) |
| `features` | str | enriched (3D) or scalar (2D) node features |

Full configuration details are documented in each YAML file.

## Supported Cancer Types

- **CRC**: Colorectal cancer (benchmark 220, extended 624 cohorts)
- **BRCA**: Breast cancer
- **LUAD**: Lung adenocarcinoma

## Data Repositories

Pre-built patient cohorts and metabolic models are available at:

- [MetaGNN-CRC](https://github.com/example/metagnn-crc) - Colorectal cancer datasets
- [MetaGNN-BRCA](https://github.com/example/metagnn-brca) - Breast cancer datasets
- [MetaGNN-LUAD](https://github.com/example/metagnn-luad) - Lung adenocarcinoma datasets

## Citation

If you use MetaGNN in your research, please cite:

```
Phongwattana T, et al. MetaGNN: Heterogeneous Graph Attention Networks 
for Topology-Aware Metabolic Reaction Activity Scoring. preprint. 2025.
```

## License

MIT License. See LICENSE file for details.

## Zenodo

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXXX (add after minting).svg)](https://doi.org/10.5281/zenodo.XXXXXXXX (add after minting))
