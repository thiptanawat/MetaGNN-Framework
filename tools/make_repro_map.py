#!/usr/bin/env python3
"""Regenerate the table-and-figure map of docs/REPRODUCTION_PAPER<N>.md from the built documents.

The numbers a manuscript gives its tables and figures change whenever a float moves between the
article and the supplement, and a hand-typed map goes stale the same day. This reads each label's
number from the LaTeX .aux file the last build wrote, its caption lead from the .tex source that
declares it, and the inputs behind it from the SOURCES table below, and rewrites the section of the
reproduction guide between the map markers. A label the sources table does not know is printed
with a warning and mapped to "see the section that reads it", never silently dropped.

Runs after the manuscripts are typeset (reproduce.sh, paper mode). The manuscript sources are not
part of the public checkout; the guide carries the last generated map, dated, and says so.

    python3 tools/make_repro_map.py paper1
    python3 tools/make_repro_map.py paper2
"""
import os, re, sys, glob, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
P = sys.argv[1] if len(sys.argv) > 1 else "paper1"
MS = os.path.join(REPO, P, "manuscript")
DOC = os.path.join(REPO, "docs", f"REPRODUCTION_{P.upper()}.md")
START, END = "<!-- map:start -->", "<!-- map:end -->"

# what each table or figure is computed from: the result files one level above the results JSON,
# and the script that produced those files. Paths are relative to <paper>/.
SOURCES = {
    "paper1": {
        "tab:practice": ("results/cohort.json, results/param_counts.json", "cohort_stats.py, count_params.py"),
        "tab:settings": ("none: a conceptual table written into the methods", "not generated"),
        "tab:main": ("results/<model>_mb<lambda>_p<pf>_r<rf>.json (the main grid)", "double_holdout.py"),
        "fig:memorization": ("the main grid, as tab:main", "double_holdout.py; drawn by make_figures_p1.py"),
        "tab:naive": ("results/naive_baselines.json, results/naive_baselines2.json", "naive_baselines.py, naive_baselines2.py"),
        "tab:ladder": ("results/*_mean.json, results/*_zero.json, results/ivr/*.json, results/naive_allfolds.json, results/naive_pooled.json", "double_holdout_ctrl.py, naive_baselines_allfolds.py, pooled_baselines.py"),
        "fig:ladder": ("as tab:ladder", "drawn by make_figures_p1.py"),
        "tab:collapse": ("results/collapse.json, results/collapse_noise.json, results/arm_noise.json", "compare_ctrl.py, collapse_noise.py, arm_noise.py"),
        "tab:synth": ("results/synth/*.json", "double_holdout_ctrl.py with the synthetic target options"),
        "fig:family": ("data/families/families_union.json, results/fam/*.json, results/fam_clean_scores.json", "build_families.py, double_holdout_ctrl.py, fam_clean_scores.py; drawn by make_figures_p1.py"),
        "tab:family": ("results/fam/*.json, results/naive_allfolds_fam.json, results/naive_pooled_fam.json, results/split_audit_fam.json, results/fam_matched.json, results/fam_clean_scores.json", "double_holdout_ctrl.py, naive_baselines_allfolds.py, pooled_baselines.py, split_audit.py, fam_clean_scores.py"),
        "tab:famstab": ("results/fam/*.json, data/families/*.json", "build_families.py, double_holdout_ctrl.py"),
        "tab:label": ("data/labels_ht29.json", "build_labels_ht29.py"),
        "fig:provenance": ("data/labels_ht29.json, results/cohort.json", "build_labels_ht29.py, cohort_stats.py; drawn by make_figures_p1.py"),
        "tab:strata": ("results/strata.json", "strata_eval.py"),
        "tab:phenotypes": ("results/phenotype_probe.json, results/phenotype_probe_multi.json, results/phenotype_null_*.json", "phenotype_probe.py, phenotype_probe_multi.py, phenotype_null.py"),
        "tab:robust": ("results/ht29/*.json, results/seedrep/*.json, results/rankgnn/*.json, results/permute/*.json, results/naive_allfolds_ht29.json, results/naive_allfolds_rank.json", "double_holdout_ctrl.py under each change, naive_baselines_allfolds.py"),
        "tab:deepmeta": ("results/deepmeta/audit_results.json, audit_results_seed22.json, audit_results_seed33.json, schedules.json, manifest.json", "code/deepmeta_audit/metrics.py on results/deepmeta/preds/seed*/*.csv (run_donor_arms.sh is the launcher that ran)"),
        "tab:domains": ("results/deepmeta/audit_results.json", "code/deepmeta_audit/metrics.py"),
        "tab:masks": ("results/mask_scores.json", "mask_scores_p1.py, on the released prediction arrays"),
        "fig:personalization": ("the substitution ladder (results/ivr/*.json, results/fam/*.json), results/naive_allfolds*.json, results/naive_pooled*.json, results/inference.json and the robustness cells", "double_holdout_ctrl.py, naive_baselines_allfolds.py, pooled_baselines.py, inference_p1.py; drawn by make_figures_p1.py"),
        "fig:phenotype": ("results/phenotype_probe.json, results/phenotype_null_*.json", "phenotype_probe.py, phenotype_null.py; drawn by make_figures_p1.py"),
        "tab:dsim": ("results/diagnostic_sim.json", "diagnostic_sim.py"),
    },
    "paper2": {
        "tab:main": ("results/repr_il/ (Design A), results/pilot/ (Design B), results/repr40_il/ (Design E)", "llm_probe.py (collection), stats.py (statistics)"),
        "tab:msi": ("results/msi_il/, clinical_metadata_msi.tsv", "msi_probe.py, msi_stats.py"),
        "tab:external": ("results/msi2/*/ (21 collections), results/external/, results/msi2_frozen.json", "msi_probe2.py (collection), msi2_stats.py (statistics)"),
        "tab:label": ("../paper1/data/labels_ht29.json", "../paper1/code/build_labels_ht29.py"),
        "tab:format": ("results/fmt_cat/, results/fmt_pct/", "llm_probe.py with the format flags, stats.py"),
        "tab:htlabel": ("the archived responses of tab:main and tab:backbone2, rescored against ../paper1/data/activity_labels_ht29.pt", "stats.py"),
        "fig:reliability": ("results/msi_il/, results/msi_b2il/, results/msi_b3il/", "msi_stats.py; drawn by make_figures.py"),
        "tab:deployment": ("results/msi2/*/design.json and raw.json, results/model_records.json", "deployment_records.py, model_records.py"),
        "fig:design": ("none: a schematic", "drawn by make_figures.py"),
        "fig:msi": ("results/msi_il/", "msi_stats.py; drawn by make_figures.py"),
        "fig:ladder": ("as tab:main", "stats.py; drawn by make_figures.py"),
        "tab:resample": ("results/repr_il/, results/crossed_boot_10k.json", "stats.py, crossed_boot_10k.py"),
        "fig:responsive": ("results/repr_il/, results/repr40_il/", "stats.py; drawn by make_figures.py"),
        "fig:presence": ("results/repr_il/", "stats.py; drawn by make_figures.py"),
        "tab:pc": ("results/pc/, results/pc_echoval/", "positive_control.py, stats.py"),
        "tab:backbone2": ("results/repr_b2il/, results/repr_b3il/", "llm_probe.py on the second and third backbones, stats.py"),
        "tab:backbone2desc": ("results/repr_b2il/, results/repr_b3il/", "stats.py"),
        "tab:backbone2pc": ("results/pc_b2/, results/pc_b3il/", "positive_control.py, stats.py"),
        "tab:backbone2msi": ("results/msi_b2il/, results/msi_b3il/", "msi_probe.py, msi_stats.py"),
        "tab:protocol": ("results/repr/ against results/repr_il/, and the same pairs for the other backbones and Design E", "stats.py"),
        "tab:donor2": ("results/repr_il_d2/, results/repr40_il_d2/", "llm_probe.py under the second donor schedule, stats.py"),
        "tab:think": ("results/think/", "llm_probe.py with reasoning enabled, stats.py"),
    },
}


def tex_sources():
    """Every .tex body the combined document inputs, read in one string."""
    main = open(os.path.join(MS, f"{P}.tex"), encoding="utf-8").read()
    seen, out = set(), []

    def expand(text, depth=0):
        def sub(m):
            name = m.group(1)
            path = os.path.join(MS, name if name.endswith(".tex") else name + ".tex")
            if name == "numbers" or not os.path.exists(path) or path in seen or depth > 6:
                return ""
            seen.add(path)
            return expand(open(path, encoding="utf-8").read(), depth + 1)
        return re.sub(r"\\input\{([^}]+)\}", sub, text)
    return expand(main)


def captions(text):
    """label -> (kind, caption lead) for every float environment that carries a label."""
    out = {}
    for env in ("table", "figure"):
        for m in re.finditer(r"\\begin\{" + env + r"\*?\}(?P<body>(?:(?!\\begin\{" + env + r"\}).)*?)\\end\{" + env + r"\*?\}", text, re.S):
            body = m.group("body")
            lab = re.search(r"\\label\{((?:tab|fig):[^}]+)\}", body)
            cap = re.search(r"\\caption\{", body)
            if not lab:
                continue
            lead = ""
            if cap:
                i = cap.end(); depth = 1; j = i
                while j < len(body) and depth:
                    depth += {"{": 1, "}": -1}.get(body[j], 0); j += 1
                raw = body[i:j - 1]
                # a macro at the head of a caption is a generated word or count; print it as such
                raw = re.sub(r"\\([A-Za-z]+)\{\}", lambda m: "<" + m.group(1) + ">", raw)
                raw = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", raw)
                raw = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", raw)
                raw = re.sub(r"\\[a-zA-Z]+\*?", "", raw)
                raw = re.sub(r"[{}$~]", "", raw)
                raw = re.sub(r"\s+", " ", raw).strip()
                lead = re.split(r"(?<=[.!?])\s", raw, 1)[0][:140]
            out[lab.group(1)] = ("Table" if env == "table" else "Figure", lead)
    return out


def numbers():
    aux = os.path.join(MS, f"{P}.aux")
    out = {}
    for m in re.finditer(r"\\newlabel\{((?:tab|fig):[^}]+)\}\{\{([^}]*)\}\{([^}]*)\}", open(aux, encoding="utf-8", errors="replace").read()):
        out[m.group(1)] = (m.group(2), m.group(3))
    return out


def main():
    text = tex_sources()
    caps = captions(text)
    nums = numbers()
    src = SOURCES[P]
    rows_main, rows_supp = [], []
    for lab, (num, page) in sorted(nums.items(), key=lambda kv: (kv[1][0].startswith("S"), int(re.sub(r"\D", "", kv[1][0]) or 0), kv[0])):
        kind, lead = caps.get(lab, ("Table" if lab.startswith("tab") else "Figure", ""))
        if lab not in src:
            print(f"warning: no source entry for {lab}; mapped to the section that reads it", file=sys.stderr)
        inputs, produced = src.get(lab, ("see the section that reads it", "see the section that reads it"))
        row = f"| `{lab}` ({kind} {num}) | {lead} | {inputs} | {produced} |"
        (rows_supp if num.startswith("S") else rows_main).append(row)
    unbuilt = sorted(set(caps) - set(nums))
    hdr = "| Label (number in the built document) | Caption lead | Input result files (relative to `" + P + "/`) | Produced by |\n|---|---|---|---|"
    date = datetime.date.today().isoformat()
    body = [START,
            f"_Generated by `tools/make_repro_map.py` from the documents built on {date}; the manuscript sources are not in the public checkout, so this is the last built numbering._",
            "", "### Main text", "", hdr, *rows_main, "", "### Supplementary material", "", hdr, *rows_supp]
    if unbuilt:
        body += ["", "Labels declared in the sources but not built (guarded by a macro that was undefined): " + ", ".join(f"`{u}`" for u in unbuilt) + "."]
    body += [END]
    doc = open(DOC, encoding="utf-8").read()
    if START in doc and END in doc:
        doc = doc[:doc.index(START)] + "\n".join(body) + doc[doc.index(END) + len(END):]
    else:
        doc = doc.rstrip() + "\n\n## Table and figure map\n\n" + "\n".join(body) + "\n"
    open(DOC, "w", encoding="utf-8").write(doc)
    print(f"{P}: {len(rows_main)} main-text and {len(rows_supp)} supplementary floats mapped into {os.path.relpath(DOC, REPO)}")


if __name__ == "__main__":
    main()
