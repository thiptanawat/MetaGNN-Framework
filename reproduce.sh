#!/bin/bash
# Regenerate every number and figure in both manuscripts from the committed results.
# No GPU and no network required.
#   bash reproduce.sh analysis   statistics, macros and figures (the public checkout runs this)
#   bash reproduce.sh paper      the above, then typeset both manuscripts; needs the manuscript
#                                sources, which are withheld from the public checkout until publication
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
# the ledger hashes every result and data file, so it runs after the last script that writes one
# (build_results_p1.py writes data/results_p1.json) and before the macros that quote its counts; the
# macro file, the figures and the ledger itself are not hashed, so nothing hashed changes after this
python3 paper1/code/make_ledger.py >/dev/null
python3 paper1/code/make_numbers_p1.py
python3 paper1/code/make_figures_p1.py
python3 paper1/code/check_macros_p1.py
# every hash in the ledger is checked against the file it names, so a stale ledger fails the run
python3 paper1/code/verify_ledger.py
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
  && python3 code/deployment_records.py >/dev/null \
  && python3 code/make_numbers.py && python3 code/check_macros.py \
  && python3 code/make_results_md.py && python3 code/make_figures.py )
echo "== README tables =="
python3 make_readme.py
if [ "$MODE" = "paper" ]; then
  # the manuscript sources are withheld from the public checkout until publication; without them
  # this mode has nothing to typeset and says so rather than failing inside pdflatex
  for p in paper1 paper2; do
    [ -f "$p/manuscript/$p.tex" ] || { echo "paper mode needs the manuscript sources ($p/manuscript/$p.tex), which are not part of the public checkout; the analysis mode above has completed"; exit 2; }
  done
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
  # the table-and-figure maps of the reproduction guides are read from the documents just built
  python3 tools/make_repro_map.py paper1 && python3 tools/make_repro_map.py paper2
fi
# the release check: the versioned required-artifact manifest, then every path the manuscripts name
python3 tools/check_release.py >/dev/null || { echo "tools/check_release.py reports a problem; run it for the list"; exit 1; }
echo done
