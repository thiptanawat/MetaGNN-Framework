"""
Subsystem knowledge graph for metabolic pathway context.

Builds directed graph from Recon3D subsystem annotations.
Supports two traversal modes:
- Direct GPR path: reaction -> genes -> evidence
- Proxy path: reaction -> subsystem -> related genes -> evidence (for GPR-orphans)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import networkx as nx
except ImportError:
    nx = None


logger = logging.getLogger(__name__)


class SubsystemKG:
    """Knowledge graph for subsystem-level metabolic analysis."""

    def __init__(self, recon3d_path: str = "./data/recon3d.json"):
        """Initialize subsystem KG from Recon3D data.

        Args:
            recon3d_path: Path to Recon3D metabolic model JSON.

        Raises:
            ImportError: If networkx not installed.
            FileNotFoundError: If Recon3D file not found.
        """
        if nx is None:
            raise ImportError(
                "networkx required. Install with: pip install networkx"
            )

        self.recon3d_path = Path(recon3d_path)
        self.graph = nx.DiGraph()
        self.subsystem_index = {}

        self._load_recon3d()
        logger.info(f"Initialized subsystem KG with {len(self.graph)} nodes")

    def _load_recon3d(self):
        """Load Recon3D model and build KG.

        Expects JSON format with reactions, metabolites, genes arrays.
        Each reaction has: id, name, subsystem, gpr, metabolites.
        """
        if not self.recon3d_path.exists():
            logger.warning(f"Recon3D file not found: {self.recon3d_path}")
            return

        try:
            with open(self.recon3d_path, 'r') as f:
                model = json.load(f)

            # Extract reactions
            reactions = model.get("reactions", [])
            logger.info(f"Loaded {len(reactions)} reactions from Recon3D")

            # Build subsystem index
            for reaction in reactions:
                rxn_id = reaction.get("id", "")
                subsystem = reaction.get("subsystem", "Unknown")

                # Add to subsystem index
                if subsystem not in self.subsystem_index:
                    self.subsystem_index[subsystem] = []
                self.subsystem_index[subsystem].append(rxn_id)

                # Add nodes to graph
                self.graph.add_node(f"rxn:{rxn_id}", type="reaction", subsystem=subsystem)

                # Add gene nodes (from GPR)
                gpr = reaction.get("gpr", "")
                if gpr:
                    genes = self._parse_gpr(gpr)
                    for gene in genes:
                        self.graph.add_node(f"gene:{gene}", type="gene")
                        # Gene -> Reaction (expression support)
                        self.graph.add_edge(f"gene:{gene}", f"rxn:{rxn_id}", relation="encodes")

                # Add subsystem node
                self.graph.add_node(f"subsys:{subsystem}", type="subsystem")
                self.graph.add_edge(f"rxn:{rxn_id}", f"subsys:{subsystem}", relation="belongs_to")

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load Recon3D: {e}")

    def _parse_gpr(self, gpr: str) -> List[str]:
        """Parse GPR string to extract gene IDs.

        Simple parsing: assumes genes are alphanumeric tokens separated by
        whitespace, 'and', 'or'.

        Args:
            gpr: GPR string.

        Returns:
            List of gene IDs.
        """
        import re
        # Remove 'and'/'or' keywords and split on whitespace
        cleaned = re.sub(r'\s+(and|or)\s+', ' ', gpr)
        genes = [g.strip() for g in cleaned.split() if g.strip()]
        return genes

    def traverse_gpr_path(self, genes: List[str]) -> str:
        """Traverse direct GPR path: genes -> reactions -> evidence.

        Args:
            genes: List of gene IDs.

        Returns:
            Summary of found evidence.
        """
        if not genes:
            return "No genes provided"

        evidence_reactions = set()
        for gene in genes:
            gene_node = f"gene:{gene}"
            if gene_node in self.graph:
                # Find reactions encoded by this gene
                for neighbor in self.graph.successors(gene_node):
                    if neighbor.startswith("rxn:"):
                        evidence_reactions.add(neighbor)

        if evidence_reactions:
            return f"Found {len(evidence_reactions)} reactions: {', '.join(list(evidence_reactions)[:5])}"
        else:
            return f"No direct evidence for genes: {', '.join(genes[:3])}"

    def traverse_subsystem_proxy(self, subsystem: str) -> str:
        """Traverse proxy path: subsystem -> reactions -> genes.

        Used for GPR-orphan reactions to find related genes.

        Args:
            subsystem: Subsystem name.

        Returns:
            Summary of proxy evidence.
        """
        if subsystem not in self.subsystem_index:
            return f"Subsystem not found: {subsystem}"

        related_genes = set()
        related_reactions = self.subsystem_index[subsystem]

        for rxn_id in related_reactions[:10]:
            rxn_node = f"rxn:{rxn_id}"
            # Find genes that encode reactions in this subsystem
            for predecessor in self.graph.predecessors(rxn_node):
                if predecessor.startswith("gene:"):
                    related_genes.add(predecessor.replace("gene:", ""))

        if related_genes:
            return f"Subsystem '{subsystem}' includes genes: {', '.join(list(related_genes)[:5])}"
        else:
            return f"No gene evidence for subsystem: {subsystem}"

    def get_subsystem_neighbors(self, subsystem: str, limit: int = 10) -> List[str]:
        """Get neighboring reactions in same subsystem.

        Args:
            subsystem: Subsystem name.
            limit: Maximum number of neighbors to return.

        Returns:
            List of neighboring reaction IDs.
        """
        if subsystem not in self.subsystem_index:
            return []

        neighbors = self.subsystem_index[subsystem]
        return neighbors[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get KG statistics.

        Returns:
            Dictionary with graph metrics.
        """
        return {
            "total_nodes": len(self.graph),
            "total_edges": len(self.graph.edges()),
            "subsystems": len(self.subsystem_index),
            "avg_reactions_per_subsystem": len(self.graph) / max(1, len(self.subsystem_index)),
        }
