#!/usr/bin/env bash
# Recreate the feasibility data directory the audit reads, on a fresh Linux machine with
# internet access (needs: git, curl, python3 + pip, md5sum, and enough disk -- ~2.2 GB).
#
# It clones the two source repositories at the exact commits the audit was validated against,
# downloads the five DepMap Public 24Q4 v1 files from Figshare (md5-verified), and the published
# DeepMeta checkpoint from Google Drive (md5-verified), producing exactly this layout:
#
#   <FEAS>/                                    (default: ../feas_deepmeta relative to this file)
#   |- DeepMeta.pt                             checkpoint, md5 6bdd383581a53c7df72d066e0735e328
#   |- DeepMeta/                               github.com/XSLiuLab/DeepMeta @ 44c62dc0...
#   |  |- data/                                train_genes.csv, *_cell_info.csv, test_dtV2.csv,
#   |  |                                       test_preV2.csv, enz_gene_mapping.rds, cpg_gene.rds,
#   |  |                                       kegg_all_pathway.rds, HGM_all_gene.rds, the GTEx
#   |  |                                       .gct, meta_net/EnzGraphs/*_enzymes_based_graph.tsv
#   |  \- scripts/model/cell_net.py            PyG dataset class the predictor imports
#   |- PreDeepMeta/                            github.com/wt12318/PreDeepMeta @ 6ec0baee...
#   |  \- data/model_gene_order.rda            the 7993 expression-encoder genes, in order
#   \- depmap_24q4/                            Model.csv, CRISPRGeneEffect.csv,
#                                              OmicsExpressionProteinCodingGenesTPMLogp1.csv,
#                                              CRISPRGeneDependency.csv, README.txt
#
# Point the audit at a non-default location with  DEEPMETA_FEAS=<FEAS>  (the audit's config.py
# reads the same variable). Idempotent: every file is skipped once its md5 verifies, so re-runs
# only fetch what is missing or corrupt.
#
#   FEAS=/data/feas_deepmeta bash fetch_inputs.sh
set -uo pipefail

FEAS="${DEEPMETA_FEAS:-${FEAS:-$(cd "$(dirname "$0")/.." && pwd)/feas_deepmeta}}"
DEP="$FEAS/depmap_24q4"
mkdir -p "$DEP"
echo "== target feasibility directory: $FEAS"

DEEPMETA_COMMIT=44c62dc0d58605cf58e1dd56393789abdf5b5f9d
PREDEEPMETA_COMMIT=6ec0baee399547b38bf45937a247c8e019b54392
DRIVE_ID=1ZQAaSeOgmgBy-dE23i5qu5pieaAdCCUd
CKPT_MD5=6bdd383581a53c7df72d066e0735e328

fail=0

md5_of() { md5sum "$1" 2>/dev/null | awk '{print $1}'; }

verify() {  # verify <file> <expected_md5>  -> 0 if present and matching
  local f="$1" want="$2"
  [ -f "$f" ] || return 1
  [ -z "$want" ] && return 0                    # no md5 given: existence is enough
  local got; got="$(md5_of "$f")"
  [ "$got" = "$want" ]
}

# ---- Figshare: fetch <figshare_file_id> <dest> <expected_md5> ----------------------------
fetch_figshare() {
  local id="$1" dest="$2" want="$3" name; name="$(basename "$dest")"
  if verify "$dest" "$want"; then echo "  [skip] $name (md5 ok)"; return 0; fi
  echo "  [get ] $name  <- https://ndownloader.figshare.com/files/$id"
  local tmp="$dest.part"
  if ! curl -fSL --retry 5 --retry-delay 5 -o "$tmp" \
        "https://ndownloader.figshare.com/files/$id"; then
    echo "  [FAIL] download of $name failed"; rm -f "$tmp"; fail=1; return 1
  fi
  if [ -n "$want" ] && [ "$(md5_of "$tmp")" != "$want" ]; then
    echo "  [FAIL] $name md5 mismatch: got $(md5_of "$tmp"), want $want"
    rm -f "$tmp"; fail=1; return 1
  fi
  mv -f "$tmp" "$dest"; echo "  [ ok ] $name"
}

# ---- git: clone_at <url> <dir> <commit> ---------------------------------------------------
clone_at() {
  local url="$1" dir="$2" commit="$3"
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" = "$commit" ]; then
    echo "  [skip] $(basename "$dir") already at $commit"; return 0
  fi
  if [ ! -d "$dir/.git" ]; then
    echo "  [get ] cloning $url"
    # shallow clone of the tip; the pinned commit is fetched explicitly below so this works
    # even after the default branch has moved past it.
    git clone --depth 1 "$url" "$dir" || { echo "  [FAIL] clone $url"; fail=1; return 1; }
  fi
  # Fetch the exact commit by SHA (GitHub allows reachable-SHA1 want); fall back to unshallow.
  if ! git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$dir" fetch --depth 1 origin "$commit" 2>/dev/null \
      || git -C "$dir" fetch --unshallow 2>/dev/null \
      || git -C "$dir" fetch --tags origin 2>/dev/null || true
  fi
  if ! git -C "$dir" checkout -q "$commit" 2>/dev/null; then
    echo "  [FAIL] could not check out $commit in $(basename "$dir")"; fail=1; return 1
  fi
  echo "  [ ok ] $(basename "$dir") @ $(git -C "$dir" rev-parse --short HEAD)"
}

# ---- Google Drive checkpoint (confirm-token method) ---------------------------------------
fetch_checkpoint() {
  local dest="$FEAS/DeepMeta.pt"
  if verify "$dest" "$CKPT_MD5"; then echo "  [skip] DeepMeta.pt (md5 ok)"; return 0; fi
  echo "  [get ] DeepMeta.pt  <- Google Drive $DRIVE_ID"
  local jar tmp html url
  jar="$(mktemp)"; tmp="$dest.part"; html="$(mktemp)"
  local base="https://drive.usercontent.google.com/download"
  # 1) first request: for a large file Drive returns an HTML interstitial with a confirm form.
  curl -fsSL -c "$jar" -o "$html" "${base}?id=${DRIVE_ID}&export=download" || true
  if head -c 16 "$html" | grep -qi '<!DOCTYPE\|<html'; then
    # 2) pull confirm + uuid out of the form and re-request.
    local confirm uuid
    confirm="$(grep -o 'name="confirm" value="[^"]*"' "$html" | head -1 | sed -E 's/.*value="([^"]*)".*/\1/')"
    uuid="$(grep -o 'name="uuid" value="[^"]*"'    "$html" | head -1 | sed -E 's/.*value="([^"]*)".*/\1/')"
    confirm="${confirm:-t}"
    url="${base}?id=${DRIVE_ID}&export=download&confirm=${confirm}"
    [ -n "$uuid" ] && url="${url}&uuid=${uuid}"
    echo "         confirm=${confirm} uuid=${uuid:-<none>}"
    curl -fSL --retry 5 --retry-delay 5 -b "$jar" -c "$jar" -o "$tmp" "$url" || true
  else
    mv -f "$html" "$tmp"                          # small-file case: first request was the file
  fi
  rm -f "$jar" "$html"
  if verify "$tmp" "$CKPT_MD5"; then
    mv -f "$tmp" "$dest"; echo "  [ ok ] DeepMeta.pt"
  else
    echo "  [FAIL] DeepMeta.pt md5 mismatch or Drive refused (got $(md5_of "$tmp" 2>/dev/null || echo none))"
    rm -f "$tmp"; fail=1
    cat <<EOF
         Manual fallback for the checkpoint:
           - In a browser (signed in), open
               https://drive.google.com/file/d/${DRIVE_ID}/view
             click "Download anyway", and save the file to  $dest
           - or with the gdown helper:
               pip install gdown && gdown "$DRIVE_ID" -O "$dest"
           - then re-run this script; it verifies md5 ($CKPT_MD5) and skips everything else.
EOF
  fi
}

echo "== 1/4 source repositories"
clone_at https://github.com/XSLiuLab/DeepMeta   "$FEAS/DeepMeta"    "$DEEPMETA_COMMIT"
clone_at https://github.com/wt12318/PreDeepMeta "$FEAS/PreDeepMeta" "$PREDEEPMETA_COMMIT"

echo "== 2/4 DepMap Public 24Q4 v1 (Figshare 27993248)"
fetch_figshare 51065297 "$DEP/Model.csv"                                    675210d17675f3517b0ce39a3c274f16
fetch_figshare 51064667 "$DEP/CRISPRGeneEffect.csv"                         6edf7ade09b9b34199210b559d4745d3
fetch_figshare 51065489 "$DEP/OmicsExpressionProteinCodingGenesTPMLogp1.csv" 71794802b750ce77c422dad0720a40af
fetch_figshare 51064631 "$DEP/CRISPRGeneDependency.csv"                     0d3bdadf0c59264e39f7fbadf232ccdb
fetch_figshare 51065795 "$DEP/README.txt"                                   54628c15a10b9ff6536db245cfed5231

echo "== 3/4 checkpoint"
fetch_checkpoint

echo "== 4/4 python dependency"
if python3 -c "import pyreadr" 2>/dev/null; then
  echo "  [skip] pyreadr already importable"
else
  echo "  [get ] pip install pyreadr"
  python3 -m pip install --quiet pyreadr || { echo "  [FAIL] pip install pyreadr"; fail=1; }
fi

# ---- final verification of every file the audit scripts read ------------------------------
echo "== verification (files the audit reads)"
check() { if verify "$1" "$2"; then echo "  ok   ${1#$FEAS/}"; else echo "  MISS ${1#$FEAS/}"; fail=1; fi; }
check "$FEAS/DeepMeta.pt"                                                        "$CKPT_MD5"
check "$DEP/Model.csv"                                    675210d17675f3517b0ce39a3c274f16
check "$DEP/CRISPRGeneEffect.csv"                         6edf7ade09b9b34199210b559d4745d3
check "$DEP/OmicsExpressionProteinCodingGenesTPMLogp1.csv" 71794802b750ce77c422dad0720a40af
check "$DEP/README.txt"                                   54628c15a10b9ff6536db245cfed5231
# CRISPRGeneDependency.csv is fetched for completeness but is NOT read by the audit scripts.
for f in \
  DeepMeta/data/train_genes.csv \
  DeepMeta/data/all_cell_info.csv \
  DeepMeta/data/test_cell_info.csv \
  DeepMeta/data/test_dtV2.csv \
  DeepMeta/data/test_preV2.csv \
  DeepMeta/data/enz_gene_mapping.rds \
  DeepMeta/data/cpg_gene.rds \
  DeepMeta/data/kegg_all_pathway.rds \
  DeepMeta/data/HGM_all_gene.rds \
  DeepMeta/data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct \
  DeepMeta/scripts/model/cell_net.py \
  PreDeepMeta/data/model_gene_order.rda ; do
  check "$FEAS/$f" ""
done
# 19 tissue enzyme graphs the supported lineages use (all 20 non-renal graphs ship in the repo).
for net in adrenal_gland bone_marrow brain iBreastCancer1746 iCervicalCancer1611 \
  iColorectalCancer1750 iEndometrialCancer1713 iHeadNeckCancer1628 iLiverCancer1788 \
  iLungCancer1490 iOvarianCancer1620 iPancreaticCancer1613 iProstateCancer1560 \
  iSkinCancer1386 iStomachCancer1511 iTestisCancer1483 iThyroidCancer1710 \
  iUrothelialCancer1532 kidney ; do
  check "$FEAS/DeepMeta/data/meta_net/EnzGraphs/${net}_enzymes_based_graph.tsv" ""
done

if [ "$fail" -eq 0 ]; then
  echo "== DONE: all inputs present and verified. Point the audit at:  DEEPMETA_FEAS=$FEAS"
else
  echo "== INCOMPLETE: some files are missing or failed verification (see [FAIL]/MISS above)."
  exit 1
fi
