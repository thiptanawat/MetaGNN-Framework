#!/bin/bash
# Re-issue every model call of the study against OpenAI-compatible endpoints.
#
#   ENDPOINT=http://<host>/<model>/v1/chat/completions MODEL=<model> bash collect.sh
#
# The main backbone is $ENDPOINT/$MODEL. The two further backbones are collected when their
# endpoints are set, which is how they were served here (one local vLLM process at a time):
#   ENDPOINT2/MODEL2   the second backbone (Gemma 4 31B-it)
#   ENDPOINT3/MODEL3   the third backbone (Mistral Small 3.2 24B Instruct). This one needs
#                      --plain_chat, which PLAIN3 supplies; leave PLAIN3 empty for a backbone
#                      whose server accepts the thinking-control keyword.
#
# Every collection the manuscript reports as primary is issued here, interleaved and fully
# archived, with the seeds and donor seeds the archives record. Runs from this directory; needs
# rxn_context.npz, recon3d_aligned.json(.gz) and clinical_metadata_msi.tsv beside it, and for the
# external-cohort block the frozen panels under results/external/. Every script is resumable:
# rerunning skips finished cells, so an interrupted collection is continued by rerunning this file.
# The scripts that actually ran on the study's host are kept verbatim under code/provenance/; they
# hardcode that host's paths and are not the reproduction path.
set -euo pipefail
cd "$(dirname "$0")"
: "${ENDPOINT:?set ENDPOINT to the chat-completions URL}"
MODEL=${MODEL:-qwen3-8-27b}
[ -f recon3d_aligned.json ] || gunzip -k recon3d_aligned.json.gz
P="python3 code/llm_probe.py --url $ENDPOINT --model $MODEL"

# ---- secondary, blocked-order collections (one arm at a time) ---------------------------------
$P --n 300 --patients 10 --sampling random --out results/repr
$P --n 300 --patients 40 --patients_from results/repr/design.json --sampling random --out results/repr40
$P --n 240 --patients 8  --sampling stratified --out results/pilot
$P --n 240 --patients 8  --sampling stratified --format categorical --arms patient shuffled --out results/fmt_cat
$P --n 240 --patients 8  --sampling stratified --format percentile  --arms patient shuffled --out results/fmt_pct
# the stratified sampler takes n // 4 reactions per stratum, so --n 50 yields the 48 reactions reported
$P --n 50  --patients 4  --sampling stratified --arms patient patient_r2 patient_r3 --out results/det_patient
$P --n 80  --patients 4  --sampling stratified --think --max_tokens 1400 --workers 8 --out results/think
python3 code/positive_control.py --url $ENDPOINT --model $MODEL --n 150 --patients 4 --tasks read compare --out results/pc
python3 code/positive_control.py --url $ENDPOINT --model $MODEL --n 150 --patients 4 --tasks echo --out results/pc_echo_b1
python3 code/msi_probe.py --url $ENDPOINT --model $MODEL --out results/msi

# ---- the primary collections: arms interleaved within a patient, replies archived in full ------
# Methods, "Collection order, the archive, and which collection is primary": these are what the
# tables and figures report under each design's name, and the blocked runs above are the contrast.
$P --n 300 --patients 10 --sampling random --interleave --full_archive --out results/repr_il
$P --n 300 --patients 40 --patients_from results/repr/design.json --sampling random \
   --interleave --full_archive --out results/repr40_il
# the second donor schedule, which is what the donor-schedule spread is computed over
$P --n 300 --patients 10 --sampling random --donor_seed 4048 --interleave --full_archive --out results/repr_il_d2
$P --n 300 --patients 40 --patients_from results/repr/design.json --sampling random --donor_seed 4048 \
   --interleave --full_archive --out results/repr40_il_d2
python3 code/msi_probe.py --url $ENDPOINT --model $MODEL --full_archive --out results/msi_il
python3 code/msi_probe.py --url $ENDPOINT --model $MODEL --interleave --full_archive --out results/msi_int
python3 code/positive_control.py --url $ENDPOINT --model $MODEL --n 150 --patients 4 --tasks echoval \
   --full_archive --out results/pc_echoval

# ---- the frozen percentile-only interface, three cohorts by three configurations ---------------
# frozen_scorer.py fits the supervised reference on the development cohort and freezes it before any
# external patient is scored; external_cohort.py builds the two external panels. Both are offline.
python3 code/frozen_scorer.py
python3 code/external_cohort.py
mkdir -p results/msi2
msi2 () {   # url model served_name extra_flags
  local URL=$1 M=$2 NAME=$3 EXTRA=${4:-}
  for cfg in zero_shot evidence tool; do
    python3 code/msi_probe2.py --url "$URL" --model "$M" $EXTRA --cohort tcga --endpoint nonmsih \
      --config $cfg --donor_seed 11 --workers 10 --out results/msi2/tcga_${NAME}_${cfg}_s11
    python3 code/msi_probe2.py --url "$URL" --model "$M" $EXTRA --cohort gse39582 \
      --config $cfg --donor_seed 11 --workers 10 --out results/msi2/gse39582_${NAME}_${cfg}_s11
    python3 code/msi_probe2.py --url "$URL" --model "$M" $EXTRA --cohort gse13294 \
      --config $cfg --donor_seed 11 --workers 10 --out results/msi2/gse13294_${NAME}_${cfg}_s11
  done
}
msi2 "$ENDPOINT" "$MODEL" qwen

# ---- the second backbone -----------------------------------------------------------------------
if [ -n "${ENDPOINT2:-}" ]; then
  M2=${MODEL2:-gemma-4-31b-it}; P2="python3 code/llm_probe.py --url $ENDPOINT2 --model $M2"
  $P2 --n 300 --patients 10 --sampling random --out results/repr_b2
  $P2 --n 300 --patients 10 --sampling random --interleave --full_archive --out results/repr_b2il
  python3 code/positive_control.py --url $ENDPOINT2 --model $M2 --n 150 --patients 4 --tasks read compare --out results/pc_b2
  python3 code/positive_control.py --url $ENDPOINT2 --model $M2 --n 150 --patients 4 --tasks echo --out results/pc_echo_b2
  python3 code/msi_probe.py --url $ENDPOINT2 --model $M2 --out results/msi_b2
  python3 code/msi_probe.py --url $ENDPOINT2 --model $M2 --interleave --full_archive --out results/msi_b2int
  msi2 "$ENDPOINT2" "$M2" "$M2"
fi

# ---- the third backbone ------------------------------------------------------------------------
if [ -n "${ENDPOINT3:-}" ]; then
  M3=${MODEL3:-mistral-small-3.2-24b-instruct}; PLAIN3=${PLAIN3:---plain_chat}
  P3="python3 code/llm_probe.py --url $ENDPOINT3 --model $M3 $PLAIN3"
  $P3 --n 300 --patients 10 --sampling random --interleave --full_archive --out results/repr_b3il
  python3 code/positive_control.py --url $ENDPOINT3 --model $M3 $PLAIN3 --n 150 --patients 4 \
    --tasks read compare --interleave --full_archive --out results/pc_b3il
  python3 code/msi_probe.py --url $ENDPOINT3 --model $M3 $PLAIN3 --interleave --full_archive --out results/msi_b3int
  msi2 "$ENDPOINT3" "$M3" "$M3" "$PLAIN3"
fi
echo "collection complete; now: bash ../reproduce.sh analysis"
