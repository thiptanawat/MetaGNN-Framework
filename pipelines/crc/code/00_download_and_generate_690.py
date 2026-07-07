#!/usr/bin/env python3
"""
MetaGNN-CRC: Full 690-Patient Dataset Generation Pipeline
==========================================================
Downloads TCGA-COAD/READ RNA-seq from GDC, processes CPTAC proteomics,
and builds PyTorch Geometric graph tensors for all 690 patients.

PREREQUISITES:
  1. Python >= 3.9 with packages: pip install -r requirements.txt
  2. RDKit:  conda install -c conda-forge rdkit
  3. gdc-client:  https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
  4. ~80 GB free disk space (raw downloads ~50 GB + processed ~8 GB)
  5. ~16 GB RAM (for VST normalisation; 32 GB recommended)

USAGE:
  # Step 0: Download raw data (interactive — see instructions below)
  # Step 1: Run this script
  python 00_download_and_generate_690.py \
      --gdc_dir ./raw_downloads/gdc_star_counts \
      --gdc_manifest ./raw_downloads/gdc_manifest.tsv \
      --cptac_tsv ./raw_downloads/cptac_pdc_protein_abundance.tsv \
      --cptac_clinical ./raw_downloads/cptac_clinical.tsv \
      --recon3d_mat ./raw_downloads/Recon3DModel_301.mat \
      --hma_mat ./raw_downloads/hma_tissue_gems/11models.mat \
      --pubchem_props ./raw_downloads/pubchem_metabolite_props.tsv \
      --output_dir ./processed_690

  # Step 2: Run experiments
  python run_full_cohort_experiments.py --data_root ./processed_690

RAW DATA DOWNLOAD INSTRUCTIONS:
================================

╔═══════════════════════════════════════════════════════════════╗
║  1. TCGA RNA-seq (GDC Data Portal)                           ║
║     ~40 GB download, ~30 min on fast connection               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Option A: GDC Data Portal UI (recommended for first time)    ║
║  ─────────────────────────────────────────────────────────────║
║  a) Go to: https://portal.gdc.cancer.gov/repository          ║
║  b) Apply filters:                                            ║
║     - Data Category: Transcriptome Profiling                  ║
║     - Data Type: Gene Expression Quantification               ║
║     - Experimental Strategy: RNA-Seq                          ║
║     - Workflow Type: STAR - Counts                            ║
║     - Project: TCGA-COAD, TCGA-READ                          ║
║     - Sample Type: Primary Tumor                              ║
║  c) Click "Add All Files to Cart"                             ║
║  d) Go to Cart → Download Manifest                            ║
║  e) Download Clinical data (TSV) from Cart → Clinical tab     ║
║  f) Use gdc-client:                                           ║
║     gdc-client download -m gdc_manifest.txt -d gdc_downloads  ║
║                                                               ║
║  Option B: GDC API (programmatic)                             ║
║  ─────────────────────────────────────────────────────────────║
║  See function download_gdc_manifest() below — it queries      ║
║  the GDC API to get file UUIDs, then uses gdc-client.         ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║  Expected: ~707 STAR count files (690 after QC filtering)     ║
╚═══════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════╗
║  2. CPTAC Proteomics (PDC Data Portal)                       ║
║     ~200 MB download                                          ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  a) Go to: https://pdc.cancer.gov/pdc/                        ║
║  b) Search for studies:                                       ║
║     - PDC000116 (CPTAC COAD Discovery)                        ║
║     - PDC000220 (CPTAC READ Discovery)                        ║
║  c) For each study:                                           ║
║     - Click "Biospecimen" → Export to TSV (clinical mapping)  ║
║     - Click "Quantitative Data" → Download protein abundance  ║
║  d) The protein abundance file is a TSV with:                 ║
║     - Rows: protein/gene identifiers                          ║
║     - Columns: sample aliquot IDs                             ║
║     - Values: log2 TMT reporter ion ratios                    ║
║                                                               ║
║  NOTE: Only ~95 TCGA-CRC patients have matched CPTAC data.   ║
║  The remaining 595 patients get zero-filled proteomics.       ║
╚═══════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════╗
║  3. Recon3D v3.0 MATLAB Model                                ║
║     ~320 MB download                                          ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Download from VMH:                                           ║
║  https://www.vmh.life/#downloadview                           ║
║  → Recon3DModel_301.mat (MATLAB COBRA format)                 ║
║                                                               ║
║  Alternative (BiGG, for SBML/XML only — NOT for this script): ║
║  http://bigg.ucsd.edu/static/models/Recon3D.xml.gz            ║
║  (Use this ONLY for FBA experiments, not for graph building)   ║
╚═══════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════╗
║  4. HMA Tissue-Specific GEMs                                 ║
║     ~410 MB download                                          ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Download from Metabolic Atlas:                               ║
║  https://metabolicatlas.org/downloads                          ║
║  → Human1_GEMs_v2.0.zip  (contains 98 tissue GEMs)           ║
║  → Extract the MATLAB .mat file for use as --hma_mat          ║
║                                                               ║
║  These are used ONLY for generating activity pseudo-labels.   ║
╚═══════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════╗
║  5. PubChem Metabolite Properties                            ║
║     Generated by this script automatically if not provided    ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  If --pubchem_props is not provided, this script will:        ║
║  a) Extract metabolite BiGG IDs from Recon3D                  ║
║  b) Query PubChem REST API for SMILES + physico-chem props    ║
║  c) Save to pubchem_metabolite_props.tsv                      ║
║  This takes ~30 min due to API rate limits.                   ║
╚═══════════════════════════════════════════════════════════════╝

Author: Thiptanawat Phongwattana
Affiliation: School of Information Technology, KMUTT
"""

import os
import sys
import json
import logging
import argparse
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('generate_690_pipeline.log'),
    ]
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 0: Helpers for downloading GDC data programmatically
# ═══════════════════════════════════════════════════════════════════════════════

def download_gdc_manifest(output_manifest: str) -> pd.DataFrame:
    """
    Query GDC API to get file UUIDs for TCGA-COAD/READ STAR gene counts.
    Saves manifest compatible with gdc-client download.

    After running this, use:
        gdc-client download -m <output_manifest> -d gdc_downloads/
    """
    import requests

    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id",
                                     "value": ["TCGA-COAD", "TCGA-READ"]}},
            {"op": "=",  "content": {"field": "data_category",
                                     "value": "Transcriptome Profiling"}},
            {"op": "=",  "content": {"field": "data_type",
                                     "value": "Gene Expression Quantification"}},
            {"op": "=",  "content": {"field": "analysis.workflow_type",
                                     "value": "STAR - Counts"}},
            {"op": "=",  "content": {"field": "cases.samples.sample_type",
                                     "value": "Primary Tumor"}},
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.case_id,cases.submitter_id,"
                  "cases.samples.sample_type,file_size,md5sum",
        "format": "JSON",
        "size": "1000",
    }

    url = "https://api.gdc.cancer.gov/files"
    logger.info("Querying GDC API for TCGA-COAD/READ STAR count files...")
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    hits = data['data']['hits']
    logger.info(f"GDC API returned {len(hits)} files (expect ~707)")

    rows = []
    for h in hits:
        case_id = h['cases'][0]['submitter_id'] if h.get('cases') else 'UNKNOWN'
        rows.append({
            'id': h['file_id'],
            'filename': h['file_name'],
            'md5': h.get('md5sum', ''),
            'size': h.get('file_size', 0),
            'state': 'released',
            'case_id': case_id,
        })

    manifest_df = pd.DataFrame(rows)
    manifest_df[['id', 'filename', 'md5', 'size', 'state']].to_csv(
        output_manifest, sep='\t', index=False
    )
    # Also save with case mapping for later use
    manifest_df.to_csv(
        output_manifest.replace('.tsv', '_with_cases.tsv'), sep='\t', index=False
    )
    logger.info(f"GDC manifest saved: {output_manifest} ({len(manifest_df)} files)")
    return manifest_df


def fetch_pubchem_properties(
    bigg_ids: List[str],
    output_tsv: str,
    bigg_to_chebi: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Fetch physico-chemical properties and SMILES from PubChem for
    Recon3D metabolites. Uses PubChem REST API with rate limiting.

    Args:
        bigg_ids:      list of BiGG metabolite IDs (e.g. 'atp_c')
        output_tsv:    path to save the properties TSV
        bigg_to_chebi: optional mapping BiGG ID → ChEBI/PubChem CID

    Returns:
        props_df: DataFrame indexed by base BiGG ID
    """
    import requests as req
    import asyncio
    try:
        import aiohttp
        HAS_AIOHTTP = True
    except ImportError:
        HAS_AIOHTTP = False

    # Strip compartment suffixes to get unique metabolite base IDs
    # Handle both underscore notation (atp_c) and bracket notation (atp[c])
    import re
    def _strip_compartment(mid):
        mid = re.sub(r'\[[a-z]+\]$', '', mid)      # atp[c] → atp
        return mid.rsplit('_', 1)[0] if '_' in mid else mid  # atp_c → atp
    base_ids = sorted(set(_strip_compartment(mid) for mid in bigg_ids))
    logger.info(f"Fetching PubChem properties for {len(base_ids)} unique metabolites...")

    # ── Resume support: load partial results from previous run ──
    partial_tsv = output_tsv + '.partial'
    already_fetched = set()
    results = []
    if os.path.exists(partial_tsv):
        try:
            cached = pd.read_csv(partial_tsv, sep='\t', index_col='bigg_id')
            already_fetched = set(cached.index.tolist())
            results = cached.reset_index().to_dict('records')
            logger.info(f"Resuming: {len(already_fetched)} cached, "
                        f"{len(base_ids) - len(already_fetched)} remaining")
        except Exception as e:
            logger.warning(f"Could not read partial cache: {e} — starting fresh")

    to_fetch = [bid for bid in base_ids if bid not in already_fetched]
    if not to_fetch:
        logger.info("All metabolites already cached — skipping fetch.")
    elif HAS_AIOHTTP:
        # ══════════════════════════════════════════════════════════════════════
        #  ASYNC PARALLEL FETCH  (aiohttp available → ~8x faster)
        # ══════════════════════════════════════════════════════════════════════
        BIGG_CONCURRENCY = 10   # concurrent BiGG requests
        PUBCHEM_CONCURRENCY = 5 # concurrent PubChem requests (stricter rate limit)

        async def _async_fetch_all(bids_to_fetch):
            """Fetch metabolite properties concurrently via BiGG + PubChem."""
            bigg_sem = asyncio.Semaphore(BIGG_CONCURRENCY)
            pubchem_sem = asyncio.Semaphore(PUBCHEM_CONCURRENCY)
            tcp_conn = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)
            timeout = aiohttp.ClientTimeout(total=20)
            local_results = []
            counter = {'done': 0, 'resolved': 0, 'bigg_ok': 0}

            async def _pubchem_query(session, name_or_cid, by='name'):
                if by == 'cid':
                    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                           f"{name_or_cid}/property/MolecularWeight,XLogP,"
                           f"HBondAcceptorCount,HBondDonorCount,TPSA,CanonicalSMILES/JSON")
                else:
                    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                           f"{name_or_cid}/property/MolecularWeight,XLogP,"
                           f"HBondAcceptorCount,HBondDonorCount,TPSA,CanonicalSMILES/JSON")
                async with pubchem_sem:
                    async with session.get(url) as r:
                        if r.status == 200:
                            data = await r.json()
                            return data['PropertyTable']['Properties'][0]
                        if r.status == 503:
                            await asyncio.sleep(5)
                        return None

            async def _fetch_one(session, bid):
                props = None
                try:
                    # Step 1: BiGG API → get real name + PubChem CID
                    async with bigg_sem:
                        async with session.get(
                            f"http://bigg.ucsd.edu/api/v2/universal/metabolites/{bid}"
                        ) as br:
                            if br.status == 200:
                                bigg_info = await br.json()
                                counter['bigg_ok'] += 1
                                bigg_name = bigg_info.get('name', '')
                                db_links = bigg_info.get('database_links', {})
                                pubchem_ids = (db_links.get('PubChem Compound', [])
                                               or db_links.get('PubChem Substance', []))
                                # Try CID first
                                if pubchem_ids:
                                    cid = pubchem_ids[0].get('id', '')
                                    if cid:
                                        props = await _pubchem_query(session, cid, by='cid')
                                # Fallback: try BiGG name
                                if props is None and bigg_name:
                                    props = await _pubchem_query(session, bigg_name, by='name')

                    # Step 2: direct BiGG-ID lookup as last resort
                    if props is None:
                        props = await _pubchem_query(session, bid, by='name')

                except asyncio.TimeoutError:
                    pass
                except Exception:
                    pass

                counter['done'] += 1
                if props is not None:
                    counter['resolved'] += 1
                    local_results.append({
                        'bigg_id': bid,
                        'smiles': props.get('CanonicalSMILES', ''),
                        'mol_weight': props.get('MolecularWeight', 0.0),
                        'xlogp': props.get('XLogP', 0.0),
                        'hbond_acceptor': props.get('HBondAcceptorCount', 0),
                        'hbond_donor': props.get('HBondDonorCount', 0),
                        'tpsa': props.get('TPSA', 0.0),
                        'ring_count': 0,
                        'formal_charge': 0,
                    })

                # Progress + checkpoint
                d = counter['done']
                if d % 200 == 0 or d == len(bids_to_fetch):
                    logger.info(
                        f"  Progress: {d}/{len(bids_to_fetch)} "
                        f"(resolved={counter['resolved']}, bigg_ok={counter['bigg_ok']})"
                    )
                    if local_results:
                        _all = results + local_results
                        pd.DataFrame(_all).set_index('bigg_id').to_csv(partial_tsv, sep='\t')

            async with aiohttp.ClientSession(connector=tcp_conn, timeout=timeout) as session:
                # Process in batches to avoid overwhelming servers
                BATCH = 50
                for start in range(0, len(bids_to_fetch), BATCH):
                    batch = bids_to_fetch[start:start + BATCH]
                    await asyncio.gather(*[_fetch_one(session, bid) for bid in batch])
                    await asyncio.sleep(0.5)  # brief pause between batches

            return local_results

        logger.info(f"Using async parallel fetch (aiohttp) — "
                    f"BiGG×{BIGG_CONCURRENCY}, PubChem×{PUBCHEM_CONCURRENCY} concurrency")
        new_results = asyncio.run(_async_fetch_all(to_fetch))
        results.extend(new_results)
        logger.info(f"Async fetch complete: {len(new_results)} new results")

    else:
        # ══════════════════════════════════════════════════════════════════════
        #  THREADED PARALLEL FETCH  (no aiohttp → use ThreadPoolExecutor)
        # ══════════════════════════════════════════════════════════════════════
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        N_WORKERS = 8
        lock = threading.Lock()
        counter = {'done': 0, 'resolved': 0, 'bigg_ok': 0}
        local_results = []
        session = req.Session()
        adapter = req.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        def _pubchem_query_sync(name_or_cid, by='name'):
            if by == 'cid':
                url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                       f"{name_or_cid}/property/MolecularWeight,XLogP,"
                       f"HBondAcceptorCount,HBondDonorCount,TPSA,CanonicalSMILES/JSON")
            else:
                url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                       f"{name_or_cid}/property/MolecularWeight,XLogP,"
                       f"HBondAcceptorCount,HBondDonorCount,TPSA,CanonicalSMILES/JSON")
            r = session.get(url, timeout=15)
            if r.status_code == 200:
                return r.json()['PropertyTable']['Properties'][0]
            if r.status_code == 503:
                time.sleep(5)
            return None

        def _fetch_one_sync(bid):
            props = None
            try:
                br = session.get(
                    f"http://bigg.ucsd.edu/api/v2/universal/metabolites/{bid}", timeout=10)
                if br.status_code == 200:
                    bigg_info = br.json()
                    with lock:
                        counter['bigg_ok'] += 1
                    bigg_name = bigg_info.get('name', '')
                    db_links = bigg_info.get('database_links', {})
                    pubchem_ids = (db_links.get('PubChem Compound', [])
                                   or db_links.get('PubChem Substance', []))
                    if pubchem_ids:
                        cid = pubchem_ids[0].get('id', '')
                        if cid:
                            props = _pubchem_query_sync(cid, by='cid')
                    if props is None and bigg_name:
                        props = _pubchem_query_sync(bigg_name, by='name')
                if props is None:
                    props = _pubchem_query_sync(bid, by='name')
            except Exception:
                pass

            with lock:
                counter['done'] += 1
                if props is not None:
                    counter['resolved'] += 1
                    local_results.append({
                        'bigg_id': bid,
                        'smiles': props.get('CanonicalSMILES', ''),
                        'mol_weight': props.get('MolecularWeight', 0.0),
                        'xlogp': props.get('XLogP', 0.0),
                        'hbond_acceptor': props.get('HBondAcceptorCount', 0),
                        'hbond_donor': props.get('HBondDonorCount', 0),
                        'tpsa': props.get('TPSA', 0.0),
                        'ring_count': 0,
                        'formal_charge': 0,
                    })
                d = counter['done']
                if d % 200 == 0:
                    logger.info(
                        f"  Progress: {d}/{len(to_fetch)} "
                        f"(resolved={counter['resolved']}, bigg_ok={counter['bigg_ok']})")
                    if local_results:
                        _all = results + local_results
                        pd.DataFrame(_all).set_index('bigg_id').to_csv(partial_tsv, sep='\t')

        logger.info(f"Using threaded parallel fetch — {N_WORKERS} workers")
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(_fetch_one_sync, bid): bid for bid in to_fetch}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass

        session.close()
        results.extend(local_results)
        logger.info(f"Threaded fetch complete: {len(local_results)} new results")

    # ── Save final cache ──
    if results:
        pd.DataFrame(results).set_index('bigg_id').to_csv(partial_tsv, sep='\t')

    # Fill missing metabolites with defaults
    if not results:
        logger.warning(f"No PubChem results for any of {len(base_ids)} metabolites. "
                       "Creating default properties.")
        results = [{'bigg_id': bid, 'smiles': '', 'mol_weight': 0.0,
                     'xlogp': 0.0, 'hbond_acceptor': 0, 'hbond_donor': 0,
                     'tpsa': 0.0, 'ring_count': 0, 'formal_charge': 0}
                    for bid in base_ids]
    else:
        found_ids = {r['bigg_id'] for r in results}
        missing = [bid for bid in base_ids if bid not in found_ids]
        for bid in missing:
            results.append({'bigg_id': bid, 'smiles': '', 'mol_weight': 0.0,
                            'xlogp': 0.0, 'hbond_acceptor': 0, 'hbond_donor': 0,
                            'tpsa': 0.0, 'ring_count': 0, 'formal_charge': 0})
        logger.info(f"PubChem: {len(found_ids)}/{len(base_ids)} resolved, "
                    f"{len(missing)} filled with defaults")

    props_df = pd.DataFrame(results).set_index('bigg_id')

    # Fill ring_count and formal_charge via RDKit if available
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        for bid in props_df.index:
            smi = props_df.loc[bid, 'smiles']
            if pd.notna(smi) and smi:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    props_df.loc[bid, 'ring_count'] = float(Descriptors.RingCount(mol))
                    props_df.loc[bid, 'formal_charge'] = float(Chem.GetFormalCharge(mol))
    except ImportError:
        logger.warning("RDKit not available — ring_count and formal_charge will be 0")

    props_df.to_csv(output_tsv, sep='\t')
    logger.info(f"PubChem properties saved: {output_tsv} ({len(props_df)} metabolites)")
    return props_df


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: RNA-seq preprocessing (adapted from 01_preprocess_tcga_rnaseq.py)
# ═══════════════════════════════════════════════════════════════════════════════

def merge_star_counts(
    gdc_dir: str,
    manifest_tsv: str,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Merge GDC STAR count files into gene × patient matrix.
    Returns (counts_df, file_to_barcode mapping).
    """
    manifest = pd.read_csv(manifest_tsv, sep='\t')
    counts_dict = {}
    file_to_barcode = {}

    # Try to load case mapping if available
    case_manifest = manifest_tsv.replace('.tsv', '_with_cases.tsv')
    if os.path.exists(case_manifest):
        case_df = pd.read_csv(case_manifest, sep='\t')
        file_to_case = dict(zip(case_df['id'], case_df['case_id']))
    else:
        file_to_case = {}

    for _, row in manifest.iterrows():
        file_id = row['id']
        filename = row['filename']
        file_path = os.path.join(gdc_dir, file_id, filename)

        if not os.path.exists(file_path):
            # Try alternative directory structure
            alt_path = os.path.join(gdc_dir, filename)
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                logger.warning(f"Missing: {file_path}")
                continue

        try:
            # Auto-detect GDC STAR Count format by scanning the first
            # lines.  We need to count how many non-data rows to skip
            # (comments starting with '#', the header row 'gene_id ...',
            # and N_ summary rows) and detect the number of columns.
            skip_rows = 0
            n_cols = 0
            with open(file_path) as _fh:
                for line in _fh:
                    stripped = line.strip()
                    if not stripped:
                        skip_rows += 1
                        continue
                    first_field = stripped.split('\t', 1)[0]
                    if first_field.startswith('#') or first_field == 'gene_id' or first_field.startswith('N_'):
                        skip_rows += 1
                        continue
                    # First real data line — count columns
                    n_cols = len(stripped.split('\t'))
                    break

            if n_cols >= 9:
                # Current GDC format: gene_id, gene_name, gene_type, unstranded, ...
                df = pd.read_csv(file_path, sep='\t',
                                 names=['gene_id', 'gene_name', 'gene_type',
                                        'unstranded', 'stranded_first',
                                        'stranded_second', 'tpm_unstranded',
                                        'fpkm_unstranded', 'fpkm_uq_unstranded'],
                                 skiprows=skip_rows)
            else:
                # Legacy 7-column format
                df = pd.read_csv(file_path, sep='\t',
                                 names=['gene_id', 'unstranded', 'stranded_first',
                                        'stranded_second', 'tpm_unstranded',
                                        'fpkm_unstranded', 'fpkm_uq_unstranded'],
                                 skiprows=skip_rows)
            df = df.set_index('gene_id')
            # Ensure unstranded is numeric (guard against mis-parsed columns)
            df['unstranded'] = pd.to_numeric(df['unstranded'], errors='coerce').fillna(0).astype(int)
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            continue

        barcode = file_to_case.get(file_id, filename.split('.')[0])
        counts_dict[barcode] = df['unstranded']
        file_to_barcode[file_id] = barcode

    counts_df = pd.DataFrame(counts_dict)
    counts_df.index.name = 'gene_id'
    logger.info(f"Merged STAR counts: {counts_df.shape[0]:,} genes × "
                f"{counts_df.shape[1]:,} patients")
    return counts_df, file_to_barcode


def qc_filter_patients(
    counts_df: pd.DataFrame,
    min_genes_detected: int = 5000,
    min_total_reads: int = 1_000_000,
) -> pd.DataFrame:
    """
    QC filter: remove patients with too few detected genes or total reads.
    This reduces ~707 to ~690 patients (removing 17 QC failures).
    """
    n_before = counts_df.shape[1]
    genes_detected = (counts_df > 0).sum(axis=0)
    total_reads = counts_df.sum(axis=0)

    keep = (genes_detected >= min_genes_detected) & (total_reads >= min_total_reads)
    counts_filtered = counts_df.loc[:, keep]

    logger.info(
        f"QC filter: {n_before} → {counts_filtered.shape[1]} patients "
        f"(removed {n_before - counts_filtered.shape[1]} with <{min_genes_detected} genes "
        f"or <{min_total_reads:,} reads)"
    )
    return counts_filtered


def compute_tpm(counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log2(TPM + 1) normalisation.
    This is the normalisation used in the data_processing manuscript for 690 patients.
    (Different from cohort-220 pipeline which uses DESeq2 VST on 220 patients.)
    """
    # Simple TPM: divide by library size, multiply by 1e6
    lib_sizes = counts_df.sum(axis=0)
    tpm = counts_df.div(lib_sizes, axis=1) * 1e6
    log2_tpm = np.log2(tpm + 1)
    logger.info(f"TPM normalisation complete: shape {log2_tpm.shape}")
    return log2_tpm


def filter_low_expression(
    expr_df: pd.DataFrame,
    min_expr: float = 0.5,
    min_frac: float = 0.10,
) -> pd.DataFrame:
    """Remove genes expressed below threshold in too few patients."""
    n_before = expr_df.shape[0]
    min_patients = int(min_frac * expr_df.shape[1])
    mask = (expr_df > min_expr).sum(axis=1) >= min_patients
    filtered = expr_df.loc[mask]
    logger.info(f"Gene filter: {n_before:,} → {filtered.shape[0]:,} genes")
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: CPTAC proteomics (adapted from 02_preprocess_cptac_proteomics.py)
# ═══════════════════════════════════════════════════════════════════════════════

def process_cptac_proteomics(
    cptac_tsv: Optional[str],
    cptac_clinical: Optional[str],
    patient_barcodes: List[str],
) -> pd.DataFrame:
    """
    Process CPTAC proteomics. If files not provided, returns zero-filled matrix.

    For the 690-patient data_processing dataset:
      - 95 patients have matched CPTAC proteomics
      - 595 patients are zero-filled
      - Use proteomics_available_mask.pt for sensitivity analyses
    """
    if cptac_tsv is None or not os.path.exists(cptac_tsv):
        logger.warning("No CPTAC data provided — all patients will have zero proteomics")
        return pd.DataFrame(0.0, index=[], columns=patient_barcodes)

    # Load proteomics matrix
    prot_df = pd.read_csv(cptac_tsv, sep='\t', index_col=0)
    numeric_cols = prot_df.select_dtypes(include=[np.number]).columns.tolist()
    prot_df = prot_df[numeric_cols]
    logger.info(f"Raw CPTAC matrix: {prot_df.shape[0]} genes × {prot_df.shape[1]} samples")

    # ── Build ID mapping (CPTAC internal → TCGA barcode) ─────────────────
    aliquot_to_tcga = {}

    # Method 1: Clinical file with standard columns
    if cptac_clinical is not None and os.path.exists(cptac_clinical):
        clin_df = pd.read_csv(cptac_clinical, sep='\t')
        # Handle PDC-exported clinical format (Case Submitter ID = TCGA barcode)
        if 'Case Submitter ID' in clin_df.columns:
            # PDC format — but no direct mapping to CPTAC internal IDs
            logger.info("PDC clinical file found (TCGA barcodes only, no CPTAC ID mapping)")
        elif 'aliquot_id' in clin_df.columns and 'tcga_barcode' in clin_df.columns:
            aliquot_to_tcga = clin_df.set_index('aliquot_id')['tcga_barcode'].to_dict()
            logger.info(f"Clinical mapping loaded: {len(aliquot_to_tcga)} entries")

    # Method 2: Check if columns are already TCGA barcodes
    tcga_cols = [c for c in prot_df.columns if str(c).startswith('TCGA-')]
    if tcga_cols:
        logger.info(f"Proteomics already uses TCGA barcodes ({len(tcga_cols)} found)")

    # ── Identify tumour samples ──────────────────────────────────────────
    # Different CPTAC data sources use different naming conventions:
    #   PDC000111 (Label Free): CPTAC internal IDs like "01CO005", normal = "01CO005.N"
    #   PDC000116 (TMT10): may use ".T" suffix for tumor
    #   cptac package: may use TCGA barcodes directly
    tumour_cols = []
    if tcga_cols:
        # Already TCGA barcodes — keep all non-normal
        tumour_cols = [c for c in tcga_cols if not str(c).endswith('.N')]
    else:
        # CPTAC internal IDs: normal samples end with ".N"
        normal_cols = [c for c in prot_df.columns if str(c).endswith('.N')]
        tumour_cols = [c for c in prot_df.columns if not str(c).endswith('.N')]
        # Also check for ".T" suffix convention
        if not tumour_cols:
            tumour_cols = [c for c in prot_df.columns
                           if str(c).endswith('.T') or 'Tumor' in str(c)]
        logger.info(f"Identified {len(tumour_cols)} tumour, {len(normal_cols)} normal samples")

    if not tumour_cols:
        tumour_cols = list(prot_df.columns)  # fallback: use all

    prot_tumour = prot_df[tumour_cols].copy()

    # Apply ID mapping if available
    if aliquot_to_tcga:
        prot_tumour.columns = [aliquot_to_tcga.get(c, c) for c in prot_tumour.columns]
    elif not tcga_cols:
        # No mapping available — CPTAC internal IDs won't match TCGA barcodes
        # Log this clearly so the user knows
        logger.warning(
            f"CPTAC uses internal IDs (e.g. '{tumour_cols[0]}') — "
            f"cannot map to TCGA barcodes without a clinical mapping file.\n"
            f"  Proteomics will be zero-filled for all 690 patients.\n"
            f"  This is acceptable: only ~90/690 have CPTAC data anyway."
        )

    # Median centering + KNN imputation
    sample_medians = prot_tumour.median(axis=0)
    prot_centred = prot_tumour.subtract(sample_medians, axis=1)

    # Completeness filter
    completeness = prot_centred.notna().mean(axis=1)
    prot_filtered = prot_centred.loc[completeness >= 0.70].copy()

    # KNN imputation
    try:
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        imputed = imputer.fit_transform(prot_filtered.T).T
        prot_imputed = pd.DataFrame(
            imputed, index=prot_filtered.index, columns=prot_filtered.columns
        )
    except ImportError:
        # Fallback: fill NaN with row minimum / 2
        prot_imputed = prot_filtered.copy()
        for gene in prot_imputed.index:
            row = prot_imputed.loc[gene]
            prot_imputed.loc[gene] = row.fillna(row.min() / 2.0)

    matched = [b for b in patient_barcodes if b in prot_imputed.columns]
    logger.info(f"CPTAC proteomics: {len(matched)} / {len(patient_barcodes)} "
                f"patients have matched data")

    return prot_imputed


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Graph construction (adapted from 03_construct_hetero_graph.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_recon3d(mat_path: str) -> dict:
    """Load Recon3D v3.0 from MATLAB COBRA .mat file."""
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # Handle different variable names in different Recon3D releases
    model = None
    for key in ['Recon3DModel', 'Recon3D', 'model']:
        if key in mat:
            model = mat[key]
            break
    if model is None:
        raise ValueError(
            f"Could not find model in .mat file. Keys found: {list(mat.keys())}"
        )

    S = np.array(model.S.todense()) if hasattr(model.S, 'todense') else np.array(model.S)

    result = {
        'S': S,
        'rxns': list(model.rxns),
        'mets': list(model.mets),
        'grRules': list(model.grRules),
        'lb': np.array(model.lb),
        'ub': np.array(model.ub),
    }
    if hasattr(model, 'genes'):
        result['genes'] = list(model.genes)

    logger.info(f"Recon3D loaded: {S.shape[1]:,} reactions × {S.shape[0]:,} metabolites")
    return result


def parse_gpr_rules(gr_rules: List[str]) -> List[List[List[str]]]:
    """
    Parse COBRA GPR rule strings into nested list structure.
    'gene1 and gene2 or gene3' → [['gene1','gene2'], ['gene3']]
    (AND groups within OR groups)
    """
    parsed = []
    for rule in gr_rules:
        rule = str(rule).strip()
        if not rule or rule == 'nan':
            parsed.append([])
            continue

        # Split by 'or' first (outer), then 'and' (inner)
        or_groups = rule.split(' or ')
        gene_sets = []
        for og in or_groups:
            genes = [g.strip().strip('()') for g in og.split(' and ')]
            genes = [g for g in genes if g and g != 'nan']
            if genes:
                gene_sets.append(genes)
        parsed.append(gene_sets)
    return parsed


def build_edge_indices(S: np.ndarray, output_dir: str):
    """Build and save edge index tensors from stoichiometric matrix."""
    ei_dir = os.path.join(output_dir, 'edge_indices')
    os.makedirs(ei_dir, exist_ok=True)

    S_dense = np.array(S) if not isinstance(S, np.ndarray) else S

    # substrate_of: M → R
    met_sub, rxn_sub = np.where(S_dense < 0)
    ei_sub = torch.tensor(np.stack([met_sub, rxn_sub]), dtype=torch.long)
    torch.save(ei_sub, os.path.join(ei_dir, 'substrate_of.pt'))
    logger.info(f"  substrate_of edges: {ei_sub.shape[1]:,}")

    # produces: R → M
    met_prod, rxn_prod = np.where(S_dense > 0)
    ei_prod = torch.tensor(np.stack([rxn_prod, met_prod]), dtype=torch.long)
    torch.save(ei_prod, os.path.join(ei_dir, 'produces.pt'))
    logger.info(f"  produces edges: {ei_prod.shape[1]:,}")

    # shared_metabolite: R ↔ R
    P = (S_dense != 0).astype(np.float32)
    shared = P.T @ P
    np.fill_diagonal(shared, 0)
    r1, r2 = np.where(shared > 0)
    ei_shared = torch.tensor(np.stack([r1, r2]), dtype=torch.long)
    torch.save(ei_shared, os.path.join(ei_dir, 'shared_metabolite.pt'))
    logger.info(f"  shared_metabolite edges: {ei_shared.shape[1]:,}")


def build_metabolite_features(
    met_ids: List[str],
    pubchem_tsv: str,
    output_h5: str,
):
    """Build X_M (n_met × 519) from PubChem properties + Morgan FP."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem
        HAS_RDKIT = True
    except ImportError:
        HAS_RDKIT = False
        logger.warning("RDKit not available — metabolite features will be zero-filled")

    props_df = pd.read_csv(pubchem_tsv, sep='\t', index_col='bigg_id')
    X_M = np.zeros((len(met_ids), 519), dtype=np.float32)

    for i, mid in enumerate(met_ids):
        base_id = mid.rsplit('_', 1)[0]
        if base_id not in props_df.index:
            continue

        row = props_df.loc[base_id]
        physico = np.array([
            row.get('mol_weight', 0.0),
            row.get('xlogp', 0.0),
            row.get('hbond_acceptor', 0.0),
            row.get('hbond_donor', 0.0),
            row.get('tpsa', 0.0),
            row.get('ring_count', 0.0),
            row.get('formal_charge', 0.0),
        ], dtype=np.float32)

        fp = np.zeros(512, dtype=np.float32)
        if HAS_RDKIT and pd.notna(row.get('smiles', None)):
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is not None:
                fp = np.array(
                    AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512),
                    dtype=np.float32,
                )
        X_M[i] = np.concatenate([physico, fp])

    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('X_M', data=X_M, compression='gzip')
        f.create_dataset('met_ids', data=np.array(met_ids, dtype='S'), compression='gzip')
        f.attrs['shape'] = str(X_M.shape)
    logger.info(f"  Metabolite features: {X_M.shape} → {output_h5}")


def build_reaction_features(
    patient_ids: List[str],
    rna_df: pd.DataFrame,
    prot_df: pd.DataFrame,
    gpr_parsed: List[List[List[str]]],
    n_reactions: int,
    output_dir: str,
) -> List[str]:
    """
    Build per-patient X_R (n_rxn × 2) via GPR mapping.
    Returns list of patient IDs that had features built.
    """
    rxn_dir = os.path.join(output_dir, 'reaction_features')
    os.makedirs(rxn_dir, exist_ok=True)

    built = []
    for pi, pid in enumerate(patient_ids):
        if pi % 100 == 0:
            logger.info(f"  Building X_R: {pi}/{len(patient_ids)}")

        X_R = np.zeros((n_reactions, 2), dtype=np.float32)

        rna_expr = rna_df[pid].to_dict() if pid in rna_df.columns else {}
        prot_expr = prot_df[pid].to_dict() if pid in prot_df.columns else {}

        for rxn_idx, gene_sets in enumerate(gpr_parsed):
            if not gene_sets:
                continue

            # RNA-seq GPR score (AND→min, OR→max)
            rna_cmplx = [min(rna_expr.get(g, 0.0) for g in grp)
                         for grp in gene_sets if grp]
            if rna_cmplx:
                X_R[rxn_idx, 0] = max(rna_cmplx)

            # Proteomics GPR score
            prt_cmplx = [min(prot_expr.get(g, 0.0) for g in grp)
                         for grp in gene_sets if grp]
            if prt_cmplx:
                X_R[rxn_idx, 1] = max(prt_cmplx)

        out_path = os.path.join(rxn_dir, f'{pid}.h5')
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('X_R', data=X_R, compression='gzip')
            f.attrs['patient_id'] = pid
            f.attrs['shape'] = str(X_R.shape)
            f.attrs['cols'] = 'col0=GPR_rnaseq, col1=GPR_proteomics'
        built.append(pid)

    logger.info(f"  Reaction features built for {len(built)} patients")
    return built


def build_activity_labels(
    hma_mat_path: str,
    n_reactions: int,
    output_pt: str,
    recon_rxn_ids: List[str] = None,
):
    """Binary activity pseudo-labels from HMA tissue GEMs.

    Supports three HMA .mat layouts:
      1. Single key holding an array of models (e.g. 'models', 'tissueModels')
      2. Single key holding one model struct
      3. Multiple named keys, each holding one tissue model (e.g. 11models.mat)

    When tissue models are subsets of Recon3D (fewer reactions), we do
    ID-based matching using recon_rxn_ids to map each model's reactions
    onto the full Recon3D reaction vector.
    """
    mat = sio.loadmat(hma_mat_path, squeeze_me=True, struct_as_record=False)

    # Auto-detect the key containing tissue-specific models
    user_keys = [k for k in mat.keys() if not k.startswith('__')]
    logger.info(f"  HMA .mat keys: {user_keys}")

    hma_models = None
    for candidate in ['models', 'tissueModels', 'model', 'RECON']:
        if candidate in mat:
            hma_models = mat[candidate]
            logger.info(f"  Using HMA key: '{candidate}'")
            break

    if hma_models is None:
        # Check if every user key is an individual model struct (11models.mat layout)
        all_are_models = all(
            hasattr(mat[k], 'rxns') or hasattr(mat[k], 'lb')
            for k in user_keys
        ) if user_keys else False

        if all_are_models and len(user_keys) > 1:
            hma_models = [mat[k] for k in user_keys]
            logger.info(f"  Using {len(hma_models)} named model keys: {user_keys}")
        elif user_keys:
            candidate = user_keys[0]
            hma_models = mat[candidate]
            logger.info(f"  Using fallback HMA key: '{candidate}'")
        else:
            raise ValueError(f"HMA .mat file has no usable keys. Found: {list(mat.keys())}")

    # Handle both single model and array of models
    if not hasattr(hma_models, '__iter__') or hasattr(hma_models, 'rxns'):
        hma_models = [hma_models]

    # Build a reaction-ID → index lookup for ID-based matching
    rxn_to_idx = {}
    if recon_rxn_ids is not None:
        rxn_to_idx = {str(r): i for i, r in enumerate(recon_rxn_ids)}
        logger.info(f"  Recon3D reaction index built: {len(rxn_to_idx)} entries")

    active_union = np.zeros(n_reactions, dtype=np.float32)
    n_matched = 0
    for m in hma_models:
        model_name = str(getattr(m, 'id', getattr(m, 'description', '?')))

        def _to_dense_1d(x):
            """Convert sparse matrix / array to dense 1-D float array."""
            if hasattr(x, 'toarray'):       # scipy sparse
                return np.asarray(x.toarray(), dtype=float).flatten()
            if hasattr(x, 'todense'):       # np.matrix
                return np.asarray(x).flatten().astype(float)
            return np.asarray(x, dtype=float).flatten()

        # Strategy 1: exact-length match (model has same # reactions as Recon3D)
        rxn_active = None
        for attr in ['activeRxns', 'lb']:
            val = getattr(m, attr, None)
            if val is not None:
                arr = _to_dense_1d(val)
                if attr == 'lb':
                    ub_val = getattr(m, 'ub', None)
                    ub = _to_dense_1d(ub_val) if ub_val is not None else np.zeros_like(arr)
                    rxn_active = ((np.abs(arr) > 1e-9) | (np.abs(ub) > 1e-9)).astype(float)
                else:
                    rxn_active = (arr > 0).astype(float) if len(arr) == n_reactions else None
                if rxn_active is not None and len(rxn_active) == n_reactions:
                    break
                rxn_active = None

        if rxn_active is not None and len(rxn_active) == n_reactions:
            active_union = np.maximum(active_union, rxn_active)
            n_matched += 1
            logger.info(f"    {model_name}: exact-length match, {int(rxn_active.sum())} active rxns")
            continue

        # Strategy 2: ID-based matching for subset models (e.g. tINIT tissue GEMs)
        if rxn_to_idx and hasattr(m, 'rxns'):
            model_rxns = [str(r) for r in m.rxns]
            # Determine active reactions by lb/ub bounds
            if hasattr(m, 'lb'):
                lb = _to_dense_1d(m.lb)
                ub_val = getattr(m, 'ub', None)
                ub = _to_dense_1d(ub_val) if ub_val is not None else np.zeros_like(lb)
                is_active = (np.abs(lb) > 1e-9) | (np.abs(ub) > 1e-9)
            else:
                # All reactions present in the model are considered active
                is_active = np.ones(len(model_rxns), dtype=bool)

            mapped = 0
            for j, rid in enumerate(model_rxns):
                if is_active[j] and rid in rxn_to_idx:
                    active_union[rxn_to_idx[rid]] = 1.0
                    mapped += 1
            if mapped > 0:
                n_matched += 1
                logger.info(f"    {model_name}: ID-matched {mapped}/{len(model_rxns)} rxns "
                            f"({int(is_active.sum())} active in model)")

    if n_matched == 0:
        logger.warning(f"  No HMA models matched {n_reactions} reactions. "
                       f"Using Recon3D flux-capacity heuristic instead.")
        # Fallback: mark all reactions as potentially active (conservative)
        active_union = np.ones(n_reactions, dtype=np.float32)

    y_r = torch.tensor(active_union, dtype=torch.float32)
    torch.save(y_r, output_pt)
    logger.info(f"  Activity labels: {int(y_r.sum())} active / "
                f"{n_reactions - int(y_r.sum())} inactive "
                f"(from {n_matched} HMA models)")
    return y_r


def build_proteomics_mask(
    patient_ids: List[str],
    prot_df: pd.DataFrame,
    output_pt: str,
):
    """Save boolean mask indicating which patients have proteomics data."""
    mask = torch.tensor(
        [pid in prot_df.columns for pid in patient_ids],
        dtype=torch.bool,
    )
    torch.save(mask, output_pt)
    n_with = int(mask.sum())
    logger.info(f"  Proteomics mask: {n_with}/{len(patient_ids)} patients have data")


def build_clinical_metadata(
    patient_ids: List[str],
    output_tsv: str,
):
    """
    Fetch clinical metadata from GDC API for all patients.
    Creates clinical_metadata.tsv with columns needed for stratified splitting.
    """
    import requests

    logger.info("Fetching clinical metadata from GDC API...")

    filters = {
        "op": "in",
        "content": {
            "field": "cases.project.project_id",
            "value": ["TCGA-COAD", "TCGA-READ"]
        }
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "submitter_id,demographic.gender,diagnoses.tumor_stage,"
                  "diagnoses.primary_diagnosis,project.project_id",
        "format": "JSON",
        "size": "1000",
    }

    resp = requests.get("https://api.gdc.cancer.gov/cases", params=params)
    resp.raise_for_status()
    cases = resp.json()['data']['hits']

    rows = []
    for c in cases:
        barcode = c.get('submitter_id', '')
        if barcode not in set(patient_ids):
            continue
        diag = c.get('diagnoses', [{}])[0] if c.get('diagnoses') else {}
        demo = c.get('demographic', {})
        rows.append({
            'tcga_barcode': barcode,
            'project': c.get('project', {}).get('project_id', ''),
            'gender': demo.get('gender', ''),
            'tumor_stage': diag.get('tumor_stage', ''),
            'primary_diagnosis': diag.get('primary_diagnosis', ''),
            'msi_status': '',   # MSI status requires separate molecular data
        })

    meta_df = pd.DataFrame(rows)
    meta_df.to_csv(output_tsv, sep='\t', index=False)
    logger.info(f"  Clinical metadata: {len(meta_df)} patients → {output_tsv}")
    return meta_df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    """Full 690-patient dataset generation pipeline."""

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MetaGNN-CRC: 690-Patient Dataset Generation Pipeline")
    logger.info("=" * 70)

    # ─── Phase 1: RNA-seq ──────────────────────────────────────────────────
    logger.info("\n[Phase 1/6] Processing TCGA RNA-seq...")

    counts_df, file_to_barcode = merge_star_counts(args.gdc_dir, args.gdc_manifest)
    counts_df = qc_filter_patients(counts_df)
    logger.info(f"After QC: {counts_df.shape[1]} patients (target: 690)")

    if counts_df.shape[1] == 0:
        raise RuntimeError(
            f"QC filter removed ALL {len(file_to_barcode)} patients. "
            f"This usually means the STAR Count file format was not parsed correctly. "
            f"Check that the GDC files have the expected column layout."
        )

    # TPM normalisation (as described in data_processing manuscript)
    tpm_df = compute_tpm(counts_df)
    tpm_filtered = filter_low_expression(tpm_df)

    # Save RNA-seq matrix
    rna_csv = os.path.join(output_dir, 'tcga_rna_seq.csv')
    tpm_filtered.to_csv(rna_csv)
    logger.info(f"RNA-seq saved: {tpm_filtered.shape} → {rna_csv}")

    # Also save as HDF5 for graph construction
    rna_h5 = os.path.join(output_dir, 'tcga_crc_rnaseq.h5')
    with h5py.File(rna_h5, 'w') as f:
        f.create_dataset('vst_expression', data=tpm_filtered.values.astype(np.float32),
                         compression='gzip', compression_opts=4)
        f.create_dataset('gene_ids', data=np.array(tpm_filtered.index, dtype='S'),
                         compression='gzip')
        f.create_dataset('patient_ids', data=np.array(tpm_filtered.columns, dtype='S'),
                         compression='gzip')

    patient_ids = tpm_filtered.columns.tolist()

    # ─── Phase 2: Proteomics ──────────────────────────────────────────────
    logger.info("\n[Phase 2/6] Processing CPTAC proteomics...")

    prot_df = process_cptac_proteomics(
        args.cptac_tsv, args.cptac_clinical, patient_ids
    )

    # Save proteomics HDF5
    prot_h5 = os.path.join(output_dir, 'cptac_crc_protein.h5')
    if prot_df.shape[0] > 0:
        with h5py.File(prot_h5, 'w') as f:
            f.create_dataset('protein_abundance', data=prot_df.values.astype(np.float32),
                             compression='gzip')
            f.create_dataset('gene_ids', data=np.array(prot_df.index, dtype='S'),
                             compression='gzip')
            f.create_dataset('patient_ids', data=np.array(prot_df.columns, dtype='S'),
                             compression='gzip')
    else:
        # Create empty proteomics file
        with h5py.File(prot_h5, 'w') as f:
            f.create_dataset('protein_abundance', data=np.zeros((0, len(patient_ids)),
                             dtype=np.float32))
            f.create_dataset('gene_ids', data=np.array([], dtype='S'))
            f.create_dataset('patient_ids', data=np.array(patient_ids, dtype='S'))

    # ─── Phase 3: Load Recon3D ─────────────────────────────────────────────
    logger.info("\n[Phase 3/6] Loading Recon3D and building graph topology...")

    recon = load_recon3d(args.recon3d_mat)
    S = recon['S']
    n_rxn = S.shape[1]
    n_met = S.shape[0]

    # Build edge indices
    build_edge_indices(S, output_dir)

    # Save stoichiometric matrix
    stoich_h5 = os.path.join(output_dir, 'recon3d_stoich.h5')
    with h5py.File(stoich_h5, 'w') as f:
        f.create_dataset('S', data=S.astype(np.float32), compression='gzip')
        f.create_dataset('rxn_ids', data=np.array(recon['rxns'], dtype='S'), compression='gzip')
        f.create_dataset('met_ids', data=np.array(recon['mets'], dtype='S'), compression='gzip')

    # ─── Phase 4: Metabolite features ──────────────────────────────────────
    logger.info("\n[Phase 4/6] Building metabolite features...")

    pubchem_path = args.pubchem_props
    if pubchem_path is None or not os.path.exists(pubchem_path):
        pubchem_path = os.path.join(output_dir, 'pubchem_metabolite_props.tsv')
        logger.info("PubChem properties not provided — fetching from API...")
        fetch_pubchem_properties(recon['mets'], pubchem_path)

    build_metabolite_features(
        recon['mets'], pubchem_path,
        os.path.join(output_dir, 'metabolite_features.h5'),
    )

    # ─── Phase 5: Per-patient reaction features ───────────────────────────
    logger.info("\n[Phase 5/6] Building per-patient reaction features (GPR mapping)...")

    # Parse GPR rules
    gpr_parsed = parse_gpr_rules(recon['grRules'])

    # Also save GPR table for compatibility with existing scripts
    gpr_rows = []
    for i, gs in enumerate(gpr_parsed):
        gpr_rows.append({'rxn_idx': i, 'gene_sets_str': str(gs)})
    gpr_df = pd.DataFrame(gpr_rows)
    gpr_tsv = os.path.join(output_dir, 'gpr_table.tsv')
    gpr_df.to_csv(gpr_tsv, sep='\t', index=False)

    # Build X_R for all 690 patients
    built_ids = build_reaction_features(
        patient_ids, tpm_filtered, prot_df,
        gpr_parsed, n_rxn, output_dir,
    )

    # Proteomics availability mask
    build_proteomics_mask(
        patient_ids, prot_df,
        os.path.join(output_dir, 'proteomics_available_mask.pt'),
    )

    # ─── Phase 6: Activity pseudo-labels + clinical metadata ──────────────
    logger.info("\n[Phase 6/6] Building activity labels and clinical metadata...")

    build_activity_labels(
        args.hma_mat, n_rxn,
        os.path.join(output_dir, 'activity_pseudolabels.pt'),
        recon_rxn_ids=recon['rxns'],
    )

    build_clinical_metadata(
        patient_ids,
        os.path.join(output_dir, 'clinical_metadata.tsv'),
    )

    # ─── Summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total patients:  {len(patient_ids)}")
    logger.info(f"RNA-seq genes:   {tpm_filtered.shape[0]:,}")
    logger.info(f"Reactions:       {n_rxn:,}")
    logger.info(f"Metabolites:     {n_met:,}")
    logger.info(f"CPTAC patients:  {sum(1 for p in patient_ids if p in prot_df.columns)}")
    logger.info("")
    logger.info("Generated files:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        if os.path.isdir(fpath):
            n_files = len(os.listdir(fpath))
            logger.info(f"  {f}/ ({n_files} files)")
        else:
            size_mb = os.path.getsize(fpath) / (1024**2)
            logger.info(f"  {f} ({size_mb:.1f} MB)")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Verify with: python 05_validate_dataset.py --data_dir " + output_dir)
    logger.info("  2. Run experiments: python run_full_cohort_experiments.py --data_root " + output_dir)
    logger.info("  3. Upload to Zenodo for data_processing submission")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Generate full 690-patient MetaGNN-CRC dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required inputs
    p.add_argument('--gdc_dir', required=True,
                   help='Path to GDC download directory (contains UUID folders)')
    p.add_argument('--gdc_manifest', required=True,
                   help='GDC file manifest TSV')
    p.add_argument('--recon3d_mat', required=True,
                   help='Recon3D v3.0 MATLAB .mat file')
    p.add_argument('--hma_mat', required=True,
                   help='HMA tissue-specific GEMs .mat file')

    # Optional inputs
    p.add_argument('--cptac_tsv', default=None,
                   help='CPTAC PDC protein abundance TSV (optional; 95 patients)')
    p.add_argument('--cptac_clinical', default=None,
                   help='CPTAC PDC clinical table TSV (required if --cptac_tsv provided)')
    p.add_argument('--pubchem_props', default=None,
                   help='PubChem metabolite properties TSV (auto-generated if absent)')

    # Output
    p.add_argument('--output_dir', default='./processed_690',
                   help='Output directory for processed dataset')

    # Optional: download GDC manifest
    p.add_argument('--download_manifest', action='store_true',
                   help='Download GDC manifest via API (then run gdc-client separately)')

    args = p.parse_args()

    if args.download_manifest:
        manifest_path = args.gdc_manifest or 'gdc_manifest.tsv'
        download_gdc_manifest(manifest_path)
        print(f"\nManifest saved to: {manifest_path}")
        print(f"Now run: gdc-client download -m {manifest_path} -d gdc_downloads/")
        sys.exit(0)

    run_pipeline(args)
