"""Build heterogeneous bipartite graphs from genome-scale metabolic models."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch
from torch_geometric.data import HeteroData
import cobra


class ReconGraphBuilder:
    """Constructs heterogeneous bipartite reaction-metabolite graphs from SBML models.
    
    Parses Recon3D v3 via COBRApy, builds directed substrate_of and produces edges,
    and optional reaction-to-reaction shared_metabolite edges with sparsification.
    """

    # Known currency metabolites with degree > 150 in full model
    CURRENCY_METABOLITES = {
        "atp_c", "adp_c", "amp_c",  # Energy carriers
        "gtp_c", "gdp_c",
        "nad_c", "nadh_c",  # Redox carriers
        "nadp_c", "nadph_c",
        "fad_c", "fadh2_c",  # Flavin cofactors
        "coa_c", "accoa_c",  # CoA derivatives
        "h_c", "h2o_c",  # Basic metabolites
    }

    def __init__(self, sbml_path: Path, data_root: Optional[Path] = None):
        """Initialize graph builder with SBML model.
        
        Args:
            sbml_path: Path to SBML model file (Recon3D v3).
            data_root: Base data directory (optional, for relative paths).
            
        Raises:
            FileNotFoundError: If SBML file does not exist.
        """
        sbml_path = Path(sbml_path)
        if not sbml_path.exists():
            raise FileNotFoundError(f"SBML file not found: {sbml_path}")
        
        self.sbml_path = sbml_path
        self.data_root = Path(data_root) if data_root else Path.cwd()
        self.model = cobra.io.read_sbml_model(str(sbml_path))
        
        self.reaction_id_to_idx = {rxn.id: i for i, rxn in enumerate(self.model.reactions)}
        self.metabolite_id_to_idx = {met.id: i for i, met in enumerate(self.model.metabolites)}

    def build_graph(
        self,
        include_shared_metabolite: bool = True,
        shared_metabolite_k: int = 10,
    ) -> HeteroData:
        """Construct complete heterogeneous graph.
        
        Args:
            include_shared_metabolite: Include reaction-to-reaction edges.
            shared_metabolite_k: Top-k sparsification for shared_metabolite edges.
            
        Returns:
            PyG HeteroData object with node and edge attributes.
        """
        # Extract stoichiometry matrix and identify edge types
        substrate_edges, substrate_attrs = self._build_substrate_edges()
        produces_edges, produces_attrs = self._build_produces_edges()
        
        hetero_data = HeteroData()
        
        # Add node counts
        num_reactions = len(self.model.reactions)
        num_metabolites = len(self.model.metabolites)
        
        hetero_data["reaction"].num_nodes = num_reactions
        hetero_data["metabolite"].num_nodes = num_metabolites
        
        # Add edges
        hetero_data["metabolite", "substrate_of", "reaction"].edge_index = substrate_edges
        hetero_data["metabolite", "substrate_of", "reaction"].edge_attr = substrate_attrs
        
        hetero_data["reaction", "produces", "metabolite"].edge_index = produces_edges
        hetero_data["reaction", "produces", "metabolite"].edge_attr = produces_attrs
        
        if include_shared_metabolite:
            shared_edges, shared_attrs = self._build_shared_metabolite_edges(k=shared_metabolite_k)
            hetero_data["reaction", "shared_metabolite", "reaction"].edge_index = shared_edges
            hetero_data["reaction", "shared_metabolite", "reaction"].edge_attr = shared_attrs
        
        return hetero_data

    def _build_substrate_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build metabolite → reaction edges for substrates (S_{m,r} < 0).
        
        Returns:
            Tuple of (edge_index, edge_attributes) tensors.
        """
        substrate_sources = []
        substrate_targets = []
        substrate_weights = []
        substrate_reversible = []
        
        for rxn in self.model.reactions:
            rxn_idx = self.reaction_id_to_idx[rxn.id]
            
            for met, coeff in rxn.metabolites.items():
                if coeff < 0:  # Substrate
                    met_idx = self.metabolite_id_to_idx[met.id]
                    substrate_sources.append(met_idx)
                    substrate_targets.append(rxn_idx)
                    substrate_weights.append(abs(coeff))
                    substrate_reversible.append(float(rxn.reversibility))
        
        edge_index = torch.tensor(
            [substrate_sources, substrate_targets], dtype=torch.long
        )
        
        edge_attr = torch.stack([
            torch.tensor(substrate_weights, dtype=torch.float),
            torch.tensor(substrate_reversible, dtype=torch.float),
        ], dim=1)
        
        return edge_index, edge_attr

    def _build_produces_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build reaction → metabolite edges for products (S_{m,r} > 0).
        
        Returns:
            Tuple of (edge_index, edge_attributes) tensors.
        """
        produces_sources = []
        produces_targets = []
        produces_weights = []
        produces_reversible = []
        
        for rxn in self.model.reactions:
            rxn_idx = self.reaction_id_to_idx[rxn.id]
            
            for met, coeff in rxn.metabolites.items():
                if coeff > 0:  # Product
                    met_idx = self.metabolite_id_to_idx[met.id]
                    produces_sources.append(rxn_idx)
                    produces_targets.append(met_idx)
                    produces_weights.append(abs(coeff))
                    produces_reversible.append(float(rxn.reversibility))
        
        edge_index = torch.tensor(
            [produces_sources, produces_targets], dtype=torch.long
        )
        
        edge_attr = torch.stack([
            torch.tensor(produces_weights, dtype=torch.float),
            torch.tensor(produces_reversible, dtype=torch.float),
        ], dim=1)
        
        return edge_index, edge_attr

    def _build_shared_metabolite_edges(
        self, k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build reaction-to-reaction edges for shared non-currency metabolites.
        
        Uses top-k sparsification to limit edges per reaction pair.
        
        Args:
            k: Maximum number of shared metabolites to keep per reaction pair.
            
        Returns:
            Tuple of (edge_index, edge_attributes) tensors.
        """
        reaction_metabolites = {}
        for rxn in self.model.reactions:
            non_currency = {
                met.id for met in rxn.metabolites 
                if met.id not in self.CURRENCY_METABOLITES
            }
            reaction_metabolites[rxn.id] = non_currency
        
        shared_sources = []
        shared_targets = []
        shared_weights = []
        
        for i, rxn_i in enumerate(self.model.reactions):
            mets_i = reaction_metabolites[rxn_i.id]
            if not mets_i:
                continue
            
            overlaps = []
            for j, rxn_j in enumerate(self.model.reactions):
                if i >= j:
                    continue
                
                mets_j = reaction_metabolites[rxn_j.id]
                shared = mets_i & mets_j
                
                if shared:
                    overlaps.append((j, len(shared)))
            
            # Keep top-k overlaps
            overlaps.sort(key=lambda x: x[1], reverse=True)
            for j, count in overlaps[:k]:
                shared_sources.append(i)
                shared_targets.append(j)
                shared_weights.append(float(count))
                
                # Add reverse edge
                shared_sources.append(j)
                shared_targets.append(i)
                shared_weights.append(float(count))
        
        if shared_sources:
            edge_index = torch.tensor(
                [shared_sources, shared_targets], dtype=torch.long
            )
            edge_attr = torch.tensor(shared_weights, dtype=torch.float).unsqueeze(1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        
        return edge_index, edge_attr

    def get_reaction_ids(self) -> List[str]:
        """Get ordered reaction IDs."""
        return [rxn.id for rxn in self.model.reactions]

    def get_metabolite_ids(self) -> List[str]:
        """Get ordered metabolite IDs."""
        return [met.id for met in self.model.metabolites]

    def get_currency_metabolite_mask(self) -> torch.Tensor:
        """Get boolean mask for currency metabolites."""
        mask = torch.zeros(len(self.model.metabolites), dtype=torch.bool)
        for met_id, idx in self.metabolite_id_to_idx.items():
            if met_id in self.CURRENCY_METABOLITES:
                mask[idx] = True
        return mask
