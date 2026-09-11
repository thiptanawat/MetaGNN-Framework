#!/usr/bin/env python3
"""Check every SHA-256 the ledger records against the file it names.

The ledger (results/LEDGER.json) is regenerated at the end of the analysis chain, so a mismatch here
means a hashed file changed after the ledger was written, or a ledger from an earlier run was
committed beside newer results. Also checks that every result file on disk is listed, and that the
prediction-array checksum list (results/prediction_arrays.sha256) names the arrays the ledger says
were released. Exits nonzero on any problem.

    python3 paper1/code/verify_ledger.py
"""
import os, sys, json, glob, hashlib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
L = json.load(open(os.path.join(RES, "LEDGER.json")))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


problems = []
listed = set()
entries = list(L["data_records"]) + [f for fam in L["families"] for f in fam["files"]]
entries += list((L.get("external_audit_predictions") or {}).get("files", []))
for e in entries:
    p = os.path.join(HERE, e["file"]); listed.add(os.path.normpath(p))
    if not os.path.exists(p):
        problems.append(f"missing: {e['file']}"); continue
    if os.path.getsize(p) != e["bytes"] or sha256(p) != e["sha256"]:
        problems.append(f"changed since the ledger was written: {e['file']}")
on_disk = {os.path.normpath(p) for p in glob.glob(os.path.join(RES, "**", "*.json"), recursive=True)
           if os.path.basename(p) not in ("LEDGER.json", "prediction_arrays.json")}
for p in sorted(on_disk - listed):
    problems.append(f"result file not in the ledger: {os.path.relpath(p, HERE)}")

# the released prediction arrays: the checksum list must carry the count the ledger records
pa = L.get("prediction_arrays")
if pa:
    lst = os.path.join(HERE, pa["checksums"])
    if not os.path.exists(lst):
        problems.append(f"missing: {pa['checksums']}")
    else:
        n = sum(1 for line in open(lst) if line.strip())
        if n != pa["n_arrays"]:
            problems.append(f"{pa['checksums']} lists {n} arrays; the ledger says {pa['n_arrays']}")
        # arrays present locally (after downloading the release assets) are checked too
        root = os.path.join(HERE, pa["local_root"])
        checked = 0
        for line in open(lst):
            if not line.strip(): continue
            h, rel = line.split()
            p = os.path.join(root, rel)
            if os.path.exists(p):
                checked += 1
                if sha256(p) != h: problems.append(f"prediction array differs from its checksum: {rel}")
        print(f"prediction arrays: {n} listed, {checked} present locally and checked")

print(f"ledger: {len(entries)} hashed files checked, {len(on_disk)} result files on disk")
if problems:
    print("\n".join(problems)); sys.exit(1)
print("ledger verified: every hash matches and every result file is listed")
