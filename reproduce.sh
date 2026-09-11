#!/bin/bash
# Regenerate every number and figure in both manuscripts from the committed results.
# No GPU and no network required.
#   bash reproduce.sh analysis   statistics, macros and figures
#   bash reproduce.sh paper      the above, then typeset both manuscripts
set -euo pipefail
cd "$(dirname "$0")"
MODE=${1:-analysis}
echo "== paper 1 =="
python3 paper1/code/split_audit.py
# the label-blind reaction families the family-disjoint split rests on, rebuilt from the reference
# network alone (about forty seconds on CPU); the committed maps are its output
python3 paper1/code/build_families.py >/dev/null
# the identical audit against the family-disjoint folds, and the matched indices the clean-subset
# rescoring drops; both are inputs to build_results_p1.py
python3 paper1/code/split_audit.py --family_map paper1/data/families/families_union.json \
    --out paper1/results/split_audit_fam.json --matched_out paper1/results/fam_matched.json >/dev/null
python3 paper1/code/inference_p1.py
python3 paper1/code/build_results_p1.py
python3 paper1/code/make_numbers_p1.py
python3 paper1/code/make_figures_p1.py
python3 paper1/code/check_macros_p1.py
python3 paper1/code/make_ledger.py >/dev/null
echo "== paper 2 =="
# Every generator whose output make_numbers.py reads runs here, so that deleting a derived file and
# rerunning this driver puts it back. msi2_stats.py resamples the external-cohort contrasts with
# 10,000-draw nulls over 21 runs and crossed_boot_10k.py draws 10,000 crossed bootstrap replicates,
# so the paper 2 block takes roughly fifteen minutes; nothing in it needs a GPU or the network.
( cd paper2 && python3 code/stats.py && python3 code/msi_stats.py >/dev/null \
  && python3 code/collection_manifest.py >/dev/null \
  && python3 code/tie_audit.py >/dev/null \
  && python3 code/crossed_boot_10k.py >/dev/null \
  && python3 code/msi2_stats.py >/dev/null \
  && python3 code/make_numbers.py && python3 code/check_macros.py \
  && python3 code/make_results_md.py && python3 code/make_figures.py )
echo "== README tables =="
python3 make_readme.py
if [ "$MODE" = "paper" ]; then
  for p in paper1 paper2; do
    # the combined document a reader gets: article, references, then the supplement
    ( cd $p/manuscript && for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error $p.tex >/dev/null 2>&1 \
        || { echo "pdflatex failed on $p.tex; see $p/manuscript/$p.log"; exit 1; }; done )
    # the two files a journal takes: the article alone and the supplement alone. Their cross-references
    # into each other resolve through xr, which reads the other document's .aux, so each is built twice
    # in turn and the article once more to pick up the supplement's final numbering.
    python3 tools/make_supp.py $p >/dev/null || { echo "make_supp.py failed on $p"; exit 1; }
    ( cd $p/manuscript \
      && for i in 1 2; do pdflatex -interaction=nonstopmode ${p}_main.tex >/dev/null 2>&1; \
                          pdflatex -interaction=nonstopmode ${p}_supp.tex >/dev/null 2>&1; done \
      && pdflatex -interaction=nonstopmode -halt-on-error ${p}_main.tex >/dev/null 2>&1 \
        || { echo "pdflatex failed on ${p}_main.tex; see $p/manuscript/${p}_main.log"; exit 1; } )
    for f in $p.pdf ${p}_main.pdf ${p}_supp.pdf; do
      [ -s "$p/manuscript/$f" ] || { echo "missing $p/manuscript/$f"; exit 1; }
    done
  done
  echo "wrote, for each paper: <paper>.pdf (combined), <paper>_main.pdf (article) and <paper>_supp.pdf (supplement)"
fi
echo done
