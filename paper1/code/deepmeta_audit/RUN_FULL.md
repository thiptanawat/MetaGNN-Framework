# Running the full DeepMeta external audit

The pipeline is seven scripts in `deepmeta_audit/`. It reads data from the feasibility
directory (`../feas_deepmeta` by default, override with `DEEPMETA_FEAS=/path`) and writes
everything else inside `deepmeta_audit/` (override with `DEEPMETA_AUDIT_OUT=/path`). Move the
two directories together and nothing else has to change.

## Getting the input data

On a fresh machine, `bash fetch_inputs.sh` recreates the feasibility data directory the audit
reads: it clones `XSLiuLab/DeepMeta` @ `44c62dc0…` and `wt12318/PreDeepMeta` @ `6ec0baee…`,
downloads the five DepMap 24Q4 v1 files from Figshare (md5-verified) and the checkpoint from
Google Drive (md5-verified), and pip-installs `pyreadr` if missing. It is idempotent (skips any
file whose md5 already verifies) and prints the exact layout plus a final verification of every
file the audit reads. Set `FEAS=/path` (or `DEEPMETA_FEAS=/path`) to place it somewhere other
than `../feas_deepmeta`. See PROVENANCE.md for the full file list and md5s.

## 0. Environment

Python 3.11 with the versions this was developed against:

    torch 2.13.0+cpu (or a CUDA build), torch_geometric 2.8.0, networkx 3.6.1,
    numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, pyreadr, tqdm

Two settings are not optional:

* `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`: torch >= 2.6 refuses to unpickle both the published
  checkpoint and the PyG `Data` objects otherwise. `run_arms.py` and `native_repro.py` set it
  for the child process; export it yourself if you call `pred_enzyme_nograd.py` directly.
* `-a cpu` / `--device cpu` on a machine without CUDA (`-a cuda` with a GPU).

Sanity check before a long run:

    python3 -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)"
    ls -l ../feas_deepmeta/DeepMeta.pt   # md5 6bdd383581a53c7df72d066e0735e328

## 1..7 Commands (24 CPUs + one GPU)

```bash
cd deepmeta_audit
export OMP_NUM_THREADS=8                     # torch CPU threads for the ops PyG runs on host

# 1. manifest: eligible lines, held-out set, gene panel, gene families, input md5s
python3 manifest.py                                        # ~5 min first time, ~40 s cached

# 2. donor schedules for seeds 11 / 22 / 33
python3 arms.py                                            # ~15 s

# 3. inputs for all six arms of the primary schedule (134 recipients)
python3 build_arms.py --seed 11 --batch_size 20            # ~15 min, ~9 GB

# 4. inference, six arms x 134 lines
python3 run_arms.py --seed 11 --batch_size 8 --cores 8 --device cuda      # 30-60 min
#    CPU-only fallback (24 cores):
#    python3 run_arms.py --seed 11 --batch_size 8 --cores 8 --device cpu  # 1-2 h

# 5. the authors' own benchmark, reproduced on 24Q4 inputs (75 of their 76 test lines)
python3 native_repro.py --batch_size 8 --cores 8 --device cuda            # 3-5 min

# 6. all statistics, 2000 + 2000 bootstrap replicates, four baselines
python3 metrics.py --seed 11 --rf_jobs 24                  # ~10 min

# 7. sensitivity to the schedule: repeat the donor arms for the other two seeds
for S in 22 33; do
  python3 build_arms.py --seed $S --arms within cross within_expr within_graph --batch_size 20
  python3 run_arms.py  --seed $S --arms within cross within_expr within_graph \
                       --batch_size 8 --cores 8 --device cuda
  # `own` does not depend on the schedule: copy it in rather than recomputing it
  cp preds/seed11/preds_own.csv preds/seed$S/preds_own.csv
  python3 metrics.py --seed $S --arms own within cross within_expr within_graph \
                     --rf_jobs 24 --out audit_results_seed$S.json
done
```

`run_donor_arms.sh` is the launcher that produced the donor arms the results carry: it runs
stages 3, 4 and 6 for the four donor arms under seeds 11, 22 and 33 in one pass, copying the
schedule-independent `own` and `mean` predictions into each seed's directory. The commands above
and that script agree; the script is what ran.

`smoke.sh` runs stages 1-4 and 6 on six held-out lines and two arms in about two minutes on
2 CPUs; run it first on any new machine. `selfcheck.py` checks the concordance and bootstrap
arithmetic against a brute-force double loop and takes a second.

## What each stage costs

| stage | unit of work | wall clock (24 CPU + GPU) | wall clock (2 CPU) | peak RAM | disk |
|---|---|---|---|---|---|
| `manifest.py` | parses both big DepMap CSVs once, caches `.npz` | 5 min (40 s cached) | 5 min | 1.5 GB | 36 MB cache |
| `arms.py` | rejection sampling | 15 s | 15 s | <100 MB | 300 KB |
| `build_arms.py` | 4 SEN sets x 134 + 4 diff-exp files | 15 min | 25 min | 1.5 GB | ~9 GB |
| `run_arms.py` | 828 graph builds + forward passes | 30-60 min | ~2.5 h | 4.5 GB | batch_size x 180 MB transient |
| `native_repro.py` | 75 graph builds + forward passes | 3-5 min | 14 min (measured) | 4.5 GB | as above |
| `metrics.py` | 2 analyses x (6 arms + 4 baselines), 2x2000 bootstrap | ~10 min | ~20 min | 3 GB | ~10 MB JSON |

Measured on the two-CPU, 7 GB machine the pipeline was first developed on, with `--cores 1 --batch_size 3`:
SEN build 0.4-1.5 s per cell after the first (the CPG feature vector of a node is memoised, so
only the expression test is repeated); graph build plus forward pass 11 s per cell averaged over
the 75 authors' test lines (814 s in total), which is 7-8 s for a small tissue network
(Lung/Bowel/Skin, ~1400 nodes) and 15-30 s for a large one (brain, kidney, bone marrow, adrenal
gland: ~2700 nodes); ~500 MB of PyG scratch per batch of 3; the 2000+2000 bootstrap costs
seconds at 6 lines and about 3 minutes per analysis at 134 lines. Scale the run_arms row from
the measured 11 s per cell: 828 cell-arms is ~2.5 h single-worker, and roughly 6-8x less with
`--cores 8` and a GPU for the forward pass.

Disk: the SEN feature file is ~5-7 MB per cell (nodes x 3247 CPG features), so the four SEN
sets are ~9 GB for 134 recipients. The tissue edge list is written once per SEN directory and
symlinked per cell, which saves ~40 GB. The processed PyG graphs are the big transient item
(60-180 MB per cell); `run_arms.py` deletes each batch's directory before starting the next,
so peak transient disk is `batch_size x 180 MB`. Budget 15 GB for a comfortable full run.

Memory: keep `--cores` at or below `--batch_size`; each worker holds one graph
(nodes x 3247 features plus nodes x 7993 expression) while building it. With `--cores 8` expect
~4.5 GB. On a 7 GB machine use `--cores 1 --batch_size 3`.

## Restarting

Every stage is idempotent and resumable.

* `build_arms.py` skips a cell whose `<cell>_feat.txt` and `<cell>.txt` already exist, and only
  appends missing rows to the diff-exp files.
* `run_arms.py` reads `preds_<arm>.csv`, skips the cells already in it, and appends after every
  batch. Kill it and rerun the same command.
* `metrics.py` and `native_repro.py` are pure post-processing; rerun freely.
* To force a rebuild, delete the artefact (`arms/seed11/sen/own/ACH-000350_feat.txt`,
  `preds/seed11/preds_own.csv`, `cache/*.npz`) and rerun.

Use `--tag` to run a subset without disturbing the full run's arm specification, e.g.
`build_arms.py --tag _pilot --cells ACH-000350 ACH-000935` then
`run_arms.py --tag _pilot --out preds/pilot`.

## Outputs

    manifest.json          eligible samples, held-out set (134 lines), panel, families, input md5s
    manifest_138.json      the superseded first-run manifest (138 lines; PROVENANCE.md section 4)
    schedules.json         donor schedules for seeds 11 / 22 / 33, donor-distinct within lineage
    schedules_138.json     the first run's schedules (138 lines, no donor-patient constraint)
    schedules_unconstrained.json   the 134-line schedules before the donor-patient constraint
    recompute_cells.json   the 134 lines rebuilt and rescored after the correction
    arms/seed11/           SEN directories, diff-exp files, per-arm specifications
    preds/seed11/preds_<arm>.csv    cell, gene_name, preds_raw (seed22 and seed33 likewise)
    native/preds_native.csv, native_repro.json
    audit_results.json     every statistic, with counts and bootstrap ranges (seed 11)
    audit_results_seed22.json, audit_results_seed33.json   the same under the other schedules

## Reading `audit_results.json`

Three concordance domains are reported side by side under `analyses`, differing only in which
gene-line pairs are scored:

* `primary_template_domain`, the fair domain. A gene-line pair counts only when **both** lines'
  lineage tissue templates (the assigned GSM enzyme graphs) contain that gene's node at all.
  Within the domain, a node the template carries but the sample's SEN drops because it is
  unexpressed keeps its zero-filled prediction (the model's own statement that an unexpressed
  enzyme is not a dependency); a line whose template lacks the node is not scored on that gene.
  The per-line AUROC uses the same rule. `counts.template_domain_size_per_gene` and the
  `template_domain_size_per_gene` map inside this block give, per gene, how many scored lines'
  templates carry it.
* `secondary_explicit_only`, only pairs where every arm emitted an explicit prediction.
* `tertiary_zero_fill_all`, the whole (held-out line x panel gene) grid, absent nodes
  zero-filled. Widest and most diluted: it scores DeepMeta on genes its tissue network does not
  even carry, which is why its per-line AUROC collapses toward 0.5.

Report all three; the template domain is the one to lead with.

`verdicts_vs_sesi_0.01` labels each donor effect `C_own - C_arm` by where its 95% interval
falls: `advantage` (interval entirely above 0.01), `no sample-specific advantage larger than
0.01` (entirely below 0.01 and not below 0), `reversed` (entirely below 0), `inconclusive`
(spanning 0.01). Labels are given for both the line bootstrap and the family bootstrap; they
have to agree before the conclusion is worth stating.
