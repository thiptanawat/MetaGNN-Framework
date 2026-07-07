"""Feature engineering for reaction and metabolite nodes."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
import cobra


class ReactionFeatures:
    """Engineer reaction node features from expression and proteomics data."""

    def __init__(self, model: cobra.Model):
        """Initialize reaction feature extractor.
        
        Args:
            model: COBRA metabolic model.
        """
        self.model = model
        self.reaction_ids = [rxn.id for rxn in model.reactions]
        self.num_reactions = len(model.reactions)

    def compute_v1_features(
        self,
        expression_data: torch.Tensor,
        gpr_rules: Optional[Dict[str, str]] = None,
    ) -> torch.Tensor:
        """Compute v1 features (2D): expression + proteomics.
        
        Args:
            expression_data: Expression values (n_reactions,).
            gpr_rules: Optional GPR rules dict. If None, uses model.reactions[].gene_reaction_rule.
            
        Returns:
            Feature tensor of shape (n_reactions, 2).
        """
        if gpr_rules is None:
            gpr_rules = {
                rxn.id: rxn.gene_reaction_rule for rxn in self.model.reactions
            }
        
        aggregated_expr = self._aggregate_gpr(expression_data, gpr_rules)
        
        # Placeholder proteomics (zero-filled)
        proteomics = torch.zeros_like(aggregated_expr)
        
        features = torch.stack([aggregated_expr, proteomics], dim=1)
        return features

    def compute_v2_features(
        self,
        expression_data: torch.Tensor,
        gpr_rules: Optional[Dict[str, str]] = None,
    ) -> torch.Tensor:
        """Compute v2 features (3D): mean, max, fraction above median.
        
        Args:
            expression_data: Expression values (n_reactions,).
            gpr_rules: Optional GPR rules dict.
            
        Returns:
            Feature tensor of shape (n_reactions, 3).
        """
        if gpr_rules is None:
            gpr_rules = {
                rxn.id: rxn.gene_reaction_rule for rxn in self.model.reactions
            }
        
        aggregated_expr = self._aggregate_gpr(expression_data, gpr_rules)
        
        # Placeholder for per-reaction statistics
        median_expr = torch.median(aggregated_expr)
        mean_expr = aggregated_expr.unsqueeze(1)
        max_expr = aggregated_expr.unsqueeze(1)
        frac_above_median = (aggregated_expr > median_expr).float().unsqueeze(1)
        
        features = torch.cat([mean_expr, max_expr, frac_above_median], dim=1)
        return features

    def _aggregate_gpr(
        self, expression_data: torch.Tensor, gpr_rules: Dict[str, str]
    ) -> torch.Tensor:
        """Aggregate gene expression via GPR rules using AND/OR logic.
        
        Uses Zur et al. convention: AND → min, OR → max.
        
        Args:
            expression_data: Gene expression tensor (n_genes,).
            gpr_rules: Dict mapping reaction IDs to GPR rule strings.
            
        Returns:
            Aggregated expression per reaction (n_reactions,).
        """
        aggregated = torch.zeros(self.num_reactions, dtype=expression_data.dtype)
        
        for i, rxn_id in enumerate(self.reaction_ids):
            rule = gpr_rules.get(rxn_id, "")
            if not rule or rule.strip() == "":
                aggregated[i] = 0.0
                continue
            
            # Simple parsing: split by OR, then by AND
            or_groups = rule.split(" or ")
            or_values = []
            
            for or_group in or_groups:
                and_genes = or_group.replace("(", "").replace(")", "").split(" and ")
                and_values = []
                
                for gene_id in and_genes:
                    gene_id = gene_id.strip()
                    if gene_id:
                        # Placeholder: assume gene index maps to expression_data
                        try:
                            gene_idx = int(gene_id.split("_")[-1]) % expression_data.shape[0]
                            and_values.append(expression_data[gene_idx])
                        except (ValueError, IndexError):
                            and_values.append(torch.tensor(0.0, dtype=expression_data.dtype))
                
                if and_values:
                    or_values.append(torch.min(torch.stack(and_values)))
            
            if or_values:
                aggregated[i] = torch.max(torch.stack(or_values))
        
        # Normalize: log2(x + 1), then rank-quantile to (0, 1)
        aggregated = torch.log2(aggregated + 1)
        sorted_vals, indices = torch.sort(aggregated)
        ranks = torch.argsort(indices).float()
        quantiles = ranks / (self.num_reactions - 1)
        aggregated = quantiles
        
        return aggregated


class MetaboliteFeatures:
    """Engineer metabolite node features from physico-chemical properties."""

    # Physico-chemical descriptor functions
    _DESCRIPTOR_FUNCTIONS = [
        ("MolWt", Descriptors.MolWt),
        ("LogP", Crippen.MolLogP),
        ("HBD", Descriptors.NumHDonors),
        ("HBA", Descriptors.NumHAcceptors),
        ("PSA", Descriptors.TPSA),
        ("RotBonds", Descriptors.NumRotatableBonds),
        ("AromaticRings", Descriptors.NumAromaticRings),
    ]

    def __init__(self, model: cobra.Model):
        """Initialize metabolite feature extractor.
        
        Args:
            model: COBRA metabolic model.
        """
        self.model = model
        self.metabolite_ids = [met.id for met in model.metabolites]

    def compute_features(self, smiles_dict: Optional[Dict[str, str]] = None) -> torch.Tensor:
        """Compute metabolite features: 7 physico-chemical + 512-bit Morgan fingerprints.
        
        Args:
            smiles_dict: Dict mapping metabolite IDs to SMILES strings.
                        If None, attempts to extract from model.metabolites[].annotation.
            
        Returns:
            Feature tensor of shape (n_metabolites, 519).
        """
        if smiles_dict is None:
            smiles_dict = self._extract_smiles_from_model()
        
        num_mets = len(self.metabolite_ids)
        features = torch.zeros((num_mets, 519), dtype=torch.float)
        
        for i, met_id in enumerate(self.metabolite_ids):
            smiles = smiles_dict.get(met_id)
            
            # Compute physico-chemical descriptors
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    for j, (name, func) in enumerate(self._DESCRIPTOR_FUNCTIONS):
                        try:
                            val = func(mol)
                            features[i, j] = float(val)
                        except Exception:
                            features[i, j] = 0.0
                    
                    # Compute Morgan fingerprint
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
                    features[i, 7:] = torch.tensor(fp, dtype=torch.float)
        
        return features

    def _extract_smiles_from_model(self) -> Dict[str, str]:
        """Extract SMILES strings from model metadata if available."""
        smiles_dict = {}
        for met in self.model.metabolites:
            inchi = met.annotation.get("inchi", "")
            if inchi:
                # Try to convert InChI to SMILES if RDKit supports it
                try:
                    mol = Chem.MolFromInchi(inchi)
                    if mol:
                        smiles_dict[met.id] = Chem.MolToSmiles(mol)
                except Exception:
                    pass
        return smiles_dict
