#!/usr/bin/env python3
"""Immutable records of the served model artifacts (P2-5).

A served name is not a version. This walks each local model directory and records what was actually
loaded: the served name, the directory name, the revision recorded by the download tool when one is
present, and a SHA-256 of every configuration, tokenizer and weight file, with its size. The output
(results/model_records.json) is what the manuscript's release table is generated from, so a model
swapped under the same served name would change the table rather than pass unnoticed.

The endpoint used for the first backbone was not under our control and has no such directory; it is
recorded as unavailable rather than guessed.

  python3 model_records.py --model mistral-small-3.2-24b-instruct=/path/to/dir \
                           --model gemma-4-31b-it=/path/to/dir --out results/model_records.json
"""
import argparse, hashlib, json, os, time

INTERESTING = (".json", ".model", ".txt", ".safetensors", ".bin", ".py")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def record(name, root):
    files = []
    for dirpath, _dirs, names in os.walk(root):
        for n in sorted(names):
            if not n.endswith(INTERESTING):
                continue
            p = os.path.join(dirpath, n)
            if os.path.islink(p) and not os.path.exists(p):
                continue
            files.append(dict(path=os.path.relpath(p, root), bytes=os.path.getsize(p), sha256=sha256(p)))
    files.sort(key=lambda d: d["path"])
    rev = None
    for cand in ("refs/main", ".git/HEAD"):
        cp = os.path.join(root, cand)
        if os.path.exists(cp):
            rev = open(cp).read().strip()
            break
    # one digest over the per-file digests, so a single string identifies the whole artifact set
    agg = hashlib.sha256("\n".join(f"{f['path']} {f['sha256']}" for f in files).encode()).hexdigest()
    return dict(served_name=name, directory=os.path.basename(os.path.realpath(root)), revision=rev,
                n_files=len(files), total_bytes=sum(f["bytes"] for f in files),
                artifact_digest=agg, files=files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--unavailable", action="append", default=[], metavar="NAME",
                    help="a served name with no local artifact directory; recorded as unavailable")
    ap.add_argument("--out", default="results/model_records.json")
    a = ap.parse_args()
    out = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), models=[])
    for spec in a.model:
        name, _, root = spec.partition("=")
        if not root or not os.path.isdir(root):
            raise SystemExit(f"not a directory: {spec}")
        out["models"].append(record(name, root))
        print(f"{name}: {out['models'][-1]['n_files']} files, digest {out['models'][-1]['artifact_digest'][:16]}")
    for name in a.unavailable:
        out["models"].append(dict(served_name=name, directory=None, revision=None, n_files=0,
                                  total_bytes=0, artifact_digest=None, files=[],
                                  unavailable="served through an endpoint not under our control; no "
                                              "artifact directory exists to hash"))
        print(f"{name}: unavailable (no local artifacts)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
