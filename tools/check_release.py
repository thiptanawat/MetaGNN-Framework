#!/usr/bin/env python3
"""Check the release: the versioned required-artifact manifest, then every file path the manuscripts name.

The first check reads tools/release_manifest.json and fails on any required file or file set that
is absent or short, whether or not the manuscript sources are present. The second walks the .tex
sources when they are in the checkout; a checkout without them reports that the path check was
skipped rather than passing it.

A manuscript that names a file the reader cannot find is a broken promise, and
the promise is easy to break silently: a script gets renamed, a result moves,
and the sentence that names it stays as it was. This walks the .tex sources
under paper1/manuscript/ and paper2/manuscript/, pulls the file-path-like
strings out of \\texttt{...} and \\path{...} spans (a slash, or a recognized
extension), undoes the LaTeX escaping inside them, and looks for each one from
the repository root, from paper1/, from paper2/, and finally by basename
anywhere in the tree. It prints what it found and exits nonzero on a miss.

This checks existence only. It says nothing about whether a path is
reachable from a committed script, whether the file is current, or
whether it is tracked by git; a path can pass this check and still be,
for instance, an output that only a GPU run regenerates.

Usage:
    python3 tools/check_release.py
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

MANUSCRIPT_DIRS = [
    os.path.join(REPO, "paper1", "manuscript"),
    os.path.join(REPO, "paper2", "manuscript"),
]

# Extensions that mark a bare word (no slash) as a file path worth checking.
# The task names .py .json .pt .tsv .npz .sh .md .gz; the rest are the same
# kind of thing and show up in these two manuscripts' code and data
# availability text.
PATH_EXTENSIONS = (
    ".py", ".json", ".pt", ".tsv", ".npz", ".sh", ".md", ".gz",
    ".mat", ".npy", ".h5", ".csv", ".yaml", ".yml", ".txt", ".cff",
    ".png", ".pdf", ".bib", ".ipynb", ".toml", ".cfg", ".lock",
)

# Files the manuscripts name that live outside this repository: in the data deposit (Zenodo record
# 10.5281/zenodo.21217578) or, for the eleven reconstructions, in the upstream Human1 publication
# data (Zenodo record 3583004). Reported with their location, not as missing. Keep this list in step
# with the deposit's manifest.
DEPOSITED = {
    "11models.mat": "upstream: the eleven NCI-60 tINIT models in Human1_Publication_Data_Scripts.zip, Zenodo record 3583004 (Robinson et al. 2020), not in the data deposit",
    "activity_pseudolabels.pt": "the label vector, in the deposit's crc_624/ directory",
    "clinical_metadata.tsv": "the clinical annotation, in the deposit's crc_624/ directory",
    "recon3d_stoich.h5": "the stoichiometric matrix, in the deposit's crc_624/ directory",
}
# Hugging Face model identifiers look like paths (owner/name) but are not files anywhere.
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MODEL_ID_OWNERS = {"Qwen", "mistralai", "google", "meta-llama"}

TEXTTT_RE = re.compile(r"\\texttt\{([^{}]*)\}")
PATH_RE = re.compile(r"\\path\{([^{}]*)\}")

# LaTeX escapes that turn up inside \texttt{...}/\path{...} spans naming a
# file path. Applied in order; only the ones actually needed inside a
# filename are here, not general LaTeX unescaping.
LATEX_ESCAPES = (
    ("\\_", "_"),
    ("\\%", "%"),
    ("\\&", "&"),
    ("\\#", "#"),
    ("\\{", "{"),
    ("\\}", "}"),
    ("\\~{}", "~"),
    ("\\textasciitilde{}", "~"),
)


def normalize(raw):
    s = raw.strip()
    for escaped, plain in LATEX_ESCAPES:
        s = s.replace(escaped, plain)
    return s


def looks_like_path(s):
    if not s or " " in s or "\n" in s or "\t" in s:
        return False
    if "/" in s:
        return True
    return s.lower().endswith(PATH_EXTENSIONS)


def find_candidates(tex_path):
    text = open(tex_path, encoding="utf-8", errors="replace").read()
    found = []
    for pattern in (TEXTTT_RE, PATH_RE):
        for m in pattern.finditer(text):
            norm = normalize(m.group(1))
            if looks_like_path(norm):
                found.append(norm)
    return found


_BASENAME_INDEX = None


def basename_index():
    """Map every filename in the repository to the relative paths it appears at."""
    global _BASENAME_INDEX
    if _BASENAME_INDEX is not None:
        return _BASENAME_INDEX
    idx = {}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            idx.setdefault(f, []).append(os.path.relpath(os.path.join(root, f), REPO))
    _BASENAME_INDEX = idx
    return idx


def resolve(path):
    """Return (found, how, where) for one normalized candidate path."""
    tried = (
        (os.path.join(REPO, path), "repo root"),
        (os.path.join(REPO, "paper1", path), "paper1/"),
        (os.path.join(REPO, "paper2", path), "paper2/"),
    )
    for full, how in tried:
        if os.path.exists(full):
            return True, how, os.path.relpath(full, REPO)
    hits = basename_index().get(os.path.basename(path))
    if hits:
        where = hits[0]
        if len(hits) > 1:
            where += f"  (+{len(hits) - 1} more location{'s' if len(hits) > 2 else ''})"
        return True, "basename match", where
    return False, None, None


def collect():
    by_path = {}
    for d in MANUSCRIPT_DIRS:
        for tex_path in sorted(glob.glob(os.path.join(d, "**", "*.tex"), recursive=True)):
            rel_source = os.path.relpath(tex_path, REPO)
            for candidate in find_candidates(tex_path):
                by_path.setdefault(candidate, set()).add(rel_source)
    return by_path


def check_manifest():
    """The versioned required-artifact manifest: every listed file must exist, every glob must reach
    its count, and the ledger must record the counts the manifest fixes. This part runs whether or
    not the manuscript sources are present, so a checkout without them cannot pass by default."""
    import json
    mp = os.path.join(HERE, "release_manifest.json")
    M = json.load(open(mp))
    problems = []
    for rel in M["required_files"]:
        if not os.path.exists(os.path.join(REPO, rel)):
            problems.append(f"required file missing: {rel}")
    for pat, n in M["required_globs"].items():
        got = len(glob.glob(os.path.join(REPO, pat)))
        if got < n:
            problems.append(f"required set short: {pat} has {got} files, {n} required")
    lp = os.path.join(REPO, "paper1", "results", "LEDGER.json")
    if os.path.exists(lp):
        L = json.load(open(lp)); want = M.get("ledger_counts", {})
        got_pa = (L.get("prediction_arrays") or {}).get("n_arrays", 0)
        if got_pa != want.get("prediction_arrays_listed", got_pa):
            problems.append(f"ledger lists {got_pa} prediction arrays; manifest requires {want['prediction_arrays_listed']}")
        got_dm = (L.get("external_audit_predictions") or {}).get("n_files", 0)
        if got_dm != want.get("external_audit_prediction_files", got_dm):
            problems.append(f"ledger lists {got_dm} external-audit prediction files; manifest requires {want['external_audit_prediction_files']}")
    print(f"release manifest v{M['version']} ({M['release']}): {len(M['required_files'])} required files, "
          f"{len(M['required_globs'])} required sets, {len(problems)} problem{'s' if len(problems) != 1 else ''}")
    for p in problems:
        print("  " + p)
    return problems


def main():
    manifest_problems = check_manifest()

    present = [d for d in MANUSCRIPT_DIRS if os.path.isdir(d)]
    for d in MANUSCRIPT_DIRS:
        if not os.path.isdir(d):
            print(f"note: manuscript directory not in this checkout: {os.path.relpath(d, REPO)} "
                  f"(the manuscript-path check needs the .tex sources and is skipped for it)")

    by_path = collect()
    if not by_path:
        if not present:
            print("Manuscript sources are not in this checkout, so no manuscript-named path was checked; "
                  "only the release manifest above applies.")
        else:
            print("No \\texttt{...} or \\path{...} file-path-like strings found in the manuscripts.")
        return 1 if manifest_problems else 0

    name_w = max(len(p) for p in by_path)
    header = f"{'PATH'.ljust(name_w)}  STATUS    RESOLVED AS / SOURCE"
    rule = "-" * max(len(header), name_w + 40)
    print(header)
    print(rule)

    missing = []
    found_n = deposited_n = model_n = 0
    for path in sorted(by_path):
        ok, how, where = resolve(path)
        if ok:
            found_n += 1
            print(f"{path.ljust(name_w)}  found     {how}: {where}")
        elif os.path.basename(path) in DEPOSITED:
            deposited_n += 1
            _where = "upstream " if DEPOSITED[os.path.basename(path)].startswith("upstream") else "deposit  "
            print(f"{path.ljust(name_w)}  {_where} {DEPOSITED[os.path.basename(path)]}")
        elif MODEL_ID_RE.match(path) and path.split("/")[0] in MODEL_ID_OWNERS:
            model_n += 1
            print(f"{path.ljust(name_w)}  model id  a Hugging Face model identifier, not a file")
        else:
            missing.append(path)
            sources = ", ".join(sorted(by_path[path]))
            print(f"{path.ljust(name_w)}  MISSING   referenced in {sources}")

    n_refs = sum(len(v) for v in by_path.values())
    print(rule)
    print(f"{found_n} found in the repository, {deposited_n} in the data deposit or upstream, {model_n} model identifiers, "
          f"{len(missing)} missing, {len(by_path)} distinct paths "
          f"({n_refs} references) checked across {sum(1 for d in MANUSCRIPT_DIRS if os.path.isdir(d))} manuscript directories.")

    if missing:
        print()
        print("Missing paths:")
        for path in missing:
            sources = ", ".join(sorted(by_path[path]))
            print(f"  {path}  (referenced in {sources})")
        return 1
    return 1 if manifest_problems else 0


if __name__ == "__main__":
    sys.exit(main())
