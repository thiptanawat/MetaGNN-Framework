#!/usr/bin/env python3
"""Serving record of a locally served backbone of the replication, written before its first call.

Plan: paper2/REPLICATION_BACKBONES_PLAN_2026-11-13.md, Section 3.

    python3 code/replication_serving_record.py --tag b3 --served-name mistral-small-3.2-24b-instruct \
        --snapshot /path/to/hub/models--.../snapshots/<revision> --out results/replication_b3/serving_record.json \
        [--command "vllm serve ..."] [--models-json answer_of_v1_models.json]

Every configuration, tokenizer and weight file of the snapshot (the file types code/model_records.py
records) is hashed and compared, by name, size and SHA-256, with the record of the same served name in
results/model_records.json, which holds what the earlier collections of that backbone loaded. The
digest over the per-file digests is computed as model_records.py computes it, so it equals the recorded
one exactly when every file does. Files of other types (the chat template among them) are hashed too and
compared with results/replication/first_host_snapshot_files.json where that file names them. The exit
status is 1 when a file is missing, differs, or is of a recorded type and present without a record; the
launcher does not serve a backbone in that case.

The record also holds a label for the host with its system, the accelerator the server may use and its
driver, the interpreter and library versions, the commit of the study checkout, the command that started the server and the answer
of its model list when they are given. Run a second time for the same snapshot, it reuses the file
hashes it stored (the files are compared again by name and size), so the model list can be added
once the server is up without hashing the weights twice.
"""
import argparse, hashlib, json, os, platform, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
INTERESTING = (".json", ".model", ".txt", ".safetensors", ".bin", ".py")      # as code/model_records.py
LIBS = ("vllm", "torch", "transformers", "tokenizers", "mistral_common", "huggingface_hub", "safetensors", "numpy")


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def clean(x):
    """Local paths in a recorded string are written relative to $HF_HOME and to the home directory."""
    roots = [(os.path.realpath(os.environ["HF_HOME"]), "$HF_HOME")] if os.environ.get("HF_HOME") else []
    roots.append((os.path.expanduser("~"), "~"))
    if isinstance(x, str):
        for full, short in roots: x = x.replace(full, short)
        return x
    if isinstance(x, list): return [clean(v) for v in x]
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    return x


def versions():
    from importlib import metadata
    out = {}
    for name in LIBS:
        try: out[name] = metadata.version(name)
        except metadata.PackageNotFoundError: out[name] = None
    try:
        import torch
        out["torch_cuda"] = torch.version.cuda; out["torch_cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--served-name", required=True)
    ap.add_argument("--snapshot", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--command", default=None); ap.add_argument("--models-json", default=None)
    ap.add_argument("--host-label", default="second host", help="how the record names the machine; its network name is not stored")
    ap.add_argument("--records", default=os.path.join(ROOT, "results", "model_records.json"))
    ap.add_argument("--other-reference", default=os.path.join(ROOT, "results", "replication", "first_host_snapshot_files.json"))
    a = ap.parse_args()
    snap = os.path.realpath(a.snapshot); rev = os.path.basename(snap.rstrip(os.sep))
    want = next((m for m in json.load(open(a.records))["models"] if m["served_name"] == a.served_name), None)
    if want is None or not want.get("files"): sys.exit(f"no file record for {a.served_name} in {a.records}")
    prefix = f"snapshots/{want['revision']}/"
    wanted = {f["path"][len(prefix):]: f for f in want["files"] if f["path"].startswith(prefix)}
    if len(wanted) != len(want["files"]): sys.exit("the record holds files outside the recorded revision")
    old = json.load(open(a.out)) if os.path.exists(a.out) else {}
    cache = {f["name"]: f for f in old.get("weights", {}).get("files", [])} if old.get("snapshot") == clean(snap) else {}
    files, other, diffs = [], [], []
    for dirpath, _dirs, names in os.walk(snap):
        for n in sorted(names):
            p = os.path.join(dirpath, n)
            if not os.path.exists(p) or os.path.getsize(p) == 0: continue
            rel = os.path.relpath(p, snap); size = os.path.getsize(p)
            if not n.endswith(INTERESTING):
                other.append(dict(name=rel, bytes=size, sha256=sha256(p))); continue
            c = cache.get(rel)
            digest = c["sha256"] if c and c["bytes"] == size else sha256(p)
            files.append(dict(name=rel, bytes=size, sha256=digest))
    files.sort(key=lambda d: d["name"]); other.sort(key=lambda d: d["name"])
    # files of other types (the chat template among them): compared with what the first host held, where that is on record
    oref = {}
    if os.path.exists(a.other_reference):
        oref = {f["name"]: f for f in json.load(open(a.other_reference))["models"].get(a.served_name, {}).get("files", [])}
    for name, w in sorted(oref.items()):
        h = next((f for f in other if f["name"] == name), None)
        if h is None: diffs.append(dict(name=name, problem="missing in the snapshot"))
        elif h["bytes"] != w["bytes"] or h["sha256"] != w["sha256"]: diffs.append(dict(name=name, problem="differs from the first host"))
    for f in other: f["first_host"] = ("identical" if f["name"] in oref and not any(d["name"] == f["name"] for d in diffs) else
                                       "differs" if f["name"] in oref else "not on record")
    have = {f["name"]: f for f in files}
    for name, w in sorted(wanted.items()):
        h = have.get(name)
        if h is None: diffs.append(dict(name=name, problem="missing in the snapshot"))
        elif h["bytes"] != w["bytes"] or h["sha256"] != w["sha256"]: diffs.append(dict(name=name, problem="differs from the record"))
    for name in sorted(set(have) - set(wanted)): diffs.append(dict(name=name, problem="present in the snapshot, absent from the record"))
    agg = hashlib.sha256("\n".join(f"snapshots/{rev}/{f['name']} {f['sha256']}" for f in files).encode()).hexdigest()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    rec = dict(
        tag=a.tag, served_name=a.served_name, plan="REPLICATION_BACKBONES_PLAN_2026-11-13.md",
        written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), snapshot=snap, revision=rev,
        weights=dict(identical_to_record=(not diffs and rev == want["revision"]), recorded_revision=want["revision"],
                     n_files=len(files), total_bytes=sum(f["bytes"] for f in files), artifact_digest=agg,
                     recorded_artifact_digest=want["artifact_digest"], differences=diffs, files=files, other_files=other),
        host=dict(label=a.host_label, system=platform.platform(), python=platform.python_version(),
                  cuda_visible_devices=gpu,
                  accelerator=sh(f"nvidia-smi -i {gpu} --query-gpu=index,name,memory.total,driver_version --format=csv,noheader") if gpu else None),
        libraries=versions(), study_commit=sh(f"git -C {ROOT} rev-parse HEAD") or None,
        command=a.command if a.command is not None else old.get("command"),
        models=json.load(open(a.models_json)) if a.models_json else old.get("models"))
    rec = clean(rec)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    w = rec["weights"]
    print(f"{a.tag} ({a.served_name}): {w['n_files']} files, {w['total_bytes']} bytes, digest {agg[:16]}, "
          f"{'identical to the record' if w['identical_to_record'] else 'DIFFERENT from the record: ' + json.dumps(diffs)[:400]}")
    sys.exit(0 if w["identical_to_record"] else 1)


if __name__ == "__main__":
    main()
