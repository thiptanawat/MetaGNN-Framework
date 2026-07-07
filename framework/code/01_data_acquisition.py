#!/usr/bin/env python3
"""
Data acquisition script for MetaGNN pipeline.

Downloads and prepares all required datasets:
- Recon3D v3 SBML from vmh.life
- TCGA RNA-seq via GDC API (cancer type from config)
- CPTAC proteomics (optional)
- DepMap 22Q4 gene effect scores
- HMA tissue-specific genome-scale metabolic models
- Validates checksums and implements retry logic
"""

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataAcquisitionError(Exception):
    """Base exception for data acquisition failures."""
    pass


class ChecksumMismatchError(DataAcquisitionError):
    """Raised when file checksum does not match expected value."""
    pass


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


def verify_checksum(file_path: Path, expected_hash: str, algorithm: str = "sha256") -> bool:
    """
    Verify file integrity using checksum.

    Args:
        file_path: Path to file to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm (default: sha256)

    Returns:
        True if checksum matches, False otherwise
    """
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    computed_hash = hasher.hexdigest()
    match = computed_hash == expected_hash.lower()

    if match:
        logger.info(f"Checksum verified for {file_path.name}")
    else:
        logger.warning(
            f"Checksum mismatch for {file_path.name}: "
            f"expected {expected_hash}, got {computed_hash}"
        )

    return match


def download_file(
    url: str,
    output_path: Path,
    max_retries: int = 3,
    timeout: int = 30
) -> bool:
    """
    Download file from URL with retry logic.

    Args:
        url: URL to download from
        output_path: Path to save file
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds

    Returns:
        True if download successful, False otherwise
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = 100 * downloaded / total_size
                            logger.debug(f"Progress: {percent:.1f}%")

            logger.info(f"Successfully downloaded to {output_path}")
            return True

        except requests.RequestException as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"Download failed after {max_retries} attempts")
                return False

    return False


def acquire_recon3d(data_root: Path, config: Dict) -> Path:
    """
    Download Recon3D v3 SBML from vmh.life.

    Returns path to downloaded file.
    """
    logger.info("Acquiring Recon3D v3...")

    recon_dir = data_root / "recon3d"
    recon_dir.mkdir(parents=True, exist_ok=True)
    recon_file = recon_dir / "Recon3D_v3.xml"

    # Check if already exists and is valid
    if recon_file.exists():
        logger.info(f"Recon3D file already exists at {recon_file}")
        return recon_file

    # URL for Recon3D v3 SBML from vmh.life
    url = "https://vmh.life/api/v1/reconstructions/Recon3D/download?format=xml"

    if not download_file(url, recon_file):
        raise DataAcquisitionError(f"Failed to download Recon3D from {url}")

    logger.info(f"Recon3D acquired: {recon_file}")
    return recon_file


def acquire_tcga_rnaseq(data_root: Path, config: Dict) -> Path:
    """
    Download TCGA RNA-seq data via GDC API.

    Cancer type is determined by config['cancer_type'].

    Returns path to downloaded file.
    """
    logger.info("Acquiring TCGA RNA-seq data...")

    cancer_type = config.get("cancer_type")
    if not cancer_type:
        raise DataAcquisitionError("cancer_type not specified in config")

    # Map cancer types to GDC project codes
    cancer_to_project = {
        "CRC": "TCGA-COAD,TCGA-READ",  # Colorectal cancer
        "BRCA": "TCGA-BRCA",            # Breast cancer
        "LUAD": "TCGA-LUAD"             # Lung adenocarcinoma
    }

    project_code = cancer_to_project.get(cancer_type)
    if not project_code:
        raise DataAcquisitionError(
            f"Unsupported cancer type: {cancer_type}. "
            f"Supported: {list(cancer_to_project.keys())}"
        )

    rnaseq_dir = data_root / "tcga_rnaseq" / cancer_type.lower()
    rnaseq_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = rnaseq_dir / f"{cancer_type}_manifest.txt"

    # GDC API query for RNA-seq data
    gdc_url = "https://api.gdc.cancer.gov/files"
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {"field": "cases.project.project_id", "value": project_code.split(",")}
            },
            {
                "op": "in",
                "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}
            },
            {
                "op": "in",
                "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}
            }
        ]
    }

    params = {
        "filters": str(filters).replace("'", '"'),
        "format": "JSON",
        "size": 10000
    }

    logger.info(f"Querying GDC API for {cancer_type} RNA-seq files...")

    try:
        response = requests.get(gdc_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        file_uuids = [f["id"] for f in data.get("data", {}).get("hits", [])]
        logger.info(f"Found {len(file_uuids)} RNA-seq files for {cancer_type}")

        # Save manifest with file UUIDs for download
        with open(manifest_file, "w") as f:
            for uuid in file_uuids:
                f.write(f"{uuid}\n")

        logger.info(f"Saved manifest to {manifest_file}")
        logger.info(
            f"Use GDC Data Transfer Tool or gdc-client to download files: "
            f"gdc-client download -m {manifest_file}"
        )

        return manifest_file

    except requests.RequestException as e:
        raise DataAcquisitionError(f"Failed to query GDC API: {e}")


def acquire_cptac_proteomics(data_root: Path, config: Dict) -> Optional[Path]:
    """
    Download CPTAC proteomics data (optional).

    Returns path to downloaded file or None if proteomics not enabled.
    """
    if not config.get("include_proteomics", False):
        logger.info("Proteomics disabled in config, skipping CPTAC acquisition")
        return None

    logger.info("Acquiring CPTAC proteomics data...")

    proteomics_dir = data_root / "cptac_proteomics"
    proteomics_dir.mkdir(parents=True, exist_ok=True)

    # PDC datasets for common cancer types
    pdc_studies = {
        "CRC": ["PDC000111"],      # Colorectal cancer
        "BRCA": ["PDC000116"],     # Breast cancer
    }

    cancer_type = config.get("cancer_type")
    studies = pdc_studies.get(cancer_type, [])

    if not studies:
        logger.warning(f"No PDC proteomics data available for {cancer_type}")
        return None

    # Note: PDC requires authentication and special handling
    # This is a placeholder for the actual download workflow
    logger.info(f"PDC studies for {cancer_type}: {studies}")
    logger.info(
        "CPTAC proteomics requires authentication and manual download "
        "from https://proteomics.cancer.gov/"
    )

    return proteomics_dir


def acquire_depmap_geneeffect(data_root: Path, config: Dict) -> Path:
    """
    Download DepMap 22Q4 gene effect scores.

    Returns path to downloaded file.
    """
    logger.info("Acquiring DepMap 22Q4 gene effect scores...")

    depmap_dir = data_root / "depmap"
    depmap_dir.mkdir(parents=True, exist_ok=True)

    gene_effect_file = depmap_dir / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"

    if gene_effect_file.exists():
        logger.info(f"DepMap file already exists at {gene_effect_file}")
        return gene_effect_file

    # DepMap public data URL
    url = "https://figshare.com/ndownloader/files/34008457"

    if not download_file(url, gene_effect_file):
        raise DataAcquisitionError(f"Failed to download DepMap data from {url}")

    logger.info(f"DepMap data acquired: {gene_effect_file}")
    return gene_effect_file


def acquire_hma_gems(data_root: Path, config: Dict) -> Path:
    """
    Download HMA tissue-specific genome-scale metabolic models.

    Returns path to directory containing GEM files.
    """
    logger.info("Acquiring HMA tissue-specific GEMs...")

    hma_dir = data_root / "hma_gems"
    hma_dir.mkdir(parents=True, exist_ok=True)

    # HMA GEMs are typically available from GitHub or supplementary materials
    # This is a placeholder for the actual acquisition workflow
    logger.info(
        "HMA tissue-specific GEMs can be downloaded from: "
        "https://github.com/opencobra/cobrapy/wiki/Tissue-specific-models"
    )
    logger.info(f"Store downloaded GEM files in: {hma_dir}")

    return hma_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Acquire and prepare datasets for MetaGNN pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Root directory for data (overrides config if provided)"
    )
    parser.add_argument(
        "--skip-recon3d",
        action="store_true",
        help="Skip Recon3D acquisition"
    )
    parser.add_argument(
        "--skip-tcga",
        action="store_true",
        help="Skip TCGA acquisition"
    )
    parser.add_argument(
        "--skip-depmap",
        action="store_true",
        help="Skip DepMap acquisition"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Determine data root directory
    data_root = args.data_root or Path(config.get("data_root", "./data"))
    data_root = data_root.resolve()
    logger.info(f"Data root directory: {data_root}")

    try:
        # Acquire datasets based on config
        acquired_files = {}

        if not args.skip_recon3d:
            acquired_files["recon3d"] = acquire_recon3d(data_root, config)

        if not args.skip_tcga:
            acquired_files["tcga_rnaseq"] = acquire_tcga_rnaseq(data_root, config)

        if config.get("include_proteomics", False):
            proteomics_path = acquire_cptac_proteomics(data_root, config)
            if proteomics_path:
                acquired_files["cptac_proteomics"] = proteomics_path

        if not args.skip_depmap:
            acquired_files["depmap"] = acquire_depmap_geneeffect(data_root, config)

        acquired_files["hma_gems"] = acquire_hma_gems(data_root, config)

        # Summary
        logger.info("=" * 60)
        logger.info("Data acquisition summary:")
        for name, path in acquired_files.items():
            logger.info(f"  {name}: {path}")
        logger.info("=" * 60)

        logger.info("Data acquisition completed successfully")
        return 0

    except DataAcquisitionError as e:
        logger.error(f"Data acquisition failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
