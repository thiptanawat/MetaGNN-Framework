#!/usr/bin/env python3
"""What each collection's own design.json records, read from the archives rather than asserted.

The manuscript says what provenance a collection carries. Different collectors, written at
different points in the study, record different fields, and the earliest ones record the fewest.
Rather than describe them in prose that can drift, this walks every design.json under results/ and
reports, per collection, which provenance fields are present and what they say. The manuscript's
methods quote the coverage from here, so a claim about provenance cannot outrun the archives.

Where a field is absent from the archive the default that the collector would have used is recorded
separately, under `defaults_not_recorded`, and marked as a property of the code rather than of the
run: llm_probe.py and msi_probe.py both default --seed to 2024 and derive the donor seed as
seed + 1 unless --donor_seed is given, so a collection that records neither was drawn with those
values unless its launcher passed others. That is an inference from the code, not a record, and the
manifest labels it as such.

Writes results/collection_manifest.json. Offline, no arguments.
"""
import json, os, glob

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIELDS = ("url", "model", "max_tokens", "temperature", "seed", "donor_seed", "started_at",
          "interleave", "full_archive", "plain_chat", "minutes")
DEFAULTS = dict(seed=2024, donor_seed="seed + 1 unless --donor_seed was passed", temperature=0.0)

rows = {}
for d in sorted(glob.glob(os.path.join(RES, "*", "design.json")) +
                glob.glob(os.path.join(RES, "*", "*", "design.json"))):
    run = os.path.relpath(os.path.dirname(d), RES)
    try:
        j = json.load(open(d))
    except Exception as e:
        rows[run] = dict(error=str(e)); continue
    present = {k: j[k] for k in FIELDS if k in j and not isinstance(j[k], (list, dict))}
    rows[run] = dict(records=present,
                     absent=[k for k in FIELDS if k not in j],
                     defaults_not_recorded={k: v for k, v in DEFAULTS.items() if k not in j})

have = {k: sum(1 for r in rows.values() if k in (r.get("records") or {})) for k in FIELDS}
M = dict(_generated_by="code/collection_manifest.py",
         note=("Per collection, the provenance fields its own design.json carries. `absent` names "
               "the fields it does not carry; `defaults_not_recorded` gives the collector's default "
               "for an absent field, which is a property of the code and not a record of the run."),
         n_collections=len(rows), fields=list(FIELDS), n_recording_field=have, collections=rows)
json.dump(M, open(os.path.join(RES, "collection_manifest.json"), "w"), indent=1)
print(f"wrote results/collection_manifest.json: {len(rows)} collections")
for k in FIELDS:
    print(f"  {k:14s} recorded by {have[k]:3d} of {len(rows)}")
