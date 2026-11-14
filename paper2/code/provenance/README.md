# The launchers as they ran

These are the scripts that issued the study's collections on the machine they ran on, kept
unedited so that the provenance of every archive under `../../results/` is on the record. They are
not the reproduction path and they will not run anywhere else: each one changes into a directory of
that host, writes to its log tree, hardcodes the endpoint the main backbone was served on, and in
two cases stops and restarts a training queue that shared the accelerator. One of them reads a
patient list from a working directory that is not part of the release; `results/repr/design.json`
holds the identical list and is what `collect.sh` passes instead.

To re-issue the calls, use `../../collect.sh`, which runs every collection these scripts ran, with
the same flags, seeds and donor seeds, against endpoints you supply. The only thing it does not
reproduce is the serving: these scripts also start and stop a local vLLM server between backbones,
and the serving flags they use are recorded in the manuscript's serving section.

| script | what it collected |
|---|---|
| `run_endpoint_il.sh` | the primary interleaved Design A collections on the main backbone |
| `run_endpoint_e_il.sh` | the primary interleaved Design E collection on the main backbone |
| `run_msi_interleaved.sh` | the interleaved microsatellite collections |
| `run_backbone3.sh` | serving and collection for the third backbone |
| `run_b2il.sh`, `run_b3il.sh` | the interleaved collections on the second and third backbones |
| `run_msi2_qwen.sh` | the frozen percentile-only interface on the main backbone, three cohorts by three configurations |
| `run_msi2_local.sh` | the same on the two locally served backbones |
| `run_replication_backbones.sh` | the replication of the own-versus-donor contrast on the second and third backbones, on a second host (`REPLICATION_BACKBONES_PLAN_2026-11-13.md`); committed with that plan, before it ran |
| `tokenization_check.py` | not a launcher: the check that the two hosts turn the first prompts of the replication sample into the same token identifiers for the second and third backbones; its outputs are in `results/replication/backbones_course.json` |
