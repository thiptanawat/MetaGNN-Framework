#!/usr/bin/env python3
"""Mint the Zenodo version DOIs for a tagged release: a new version of the code record and a new
version of the data record, both under the concept identifiers the manuscripts cite.

What it does, per record:

  code  a new version of the code record (concept 10.5281/zenodo.21217568) holding one file, the
        archive of the tagged tree (`git archive` of the tag named in release.json, or the same
        archive downloaded from the repository host when this checkout does not have the tag).
        The inherited file of the previous version is removed; the description names the tag, the
        commit and the SHA-256 of the archive.
  data  a new version of the data record (concept 10.5281/zenodo.21217578) that keeps every file of
        the previous version (the curated cohort, the reaction features, the labels, the splits)
        and adds the per-cell prediction arrays of the first study: the ten tarballs, one per
        result family, and the two committed checksum lists. Each tarball is checked against
        paper1/results/prediction_array_tarballs.sha256 before it is sent, and the version is not
        published unless every file of the previous version is present with an unchanged MD5.

Both drafts are populated, checked and, with --publish, published; the minted DOIs are written to
release.json (version_doi_code, version_doi_data) and a record of the deposit to
tools/zenodo_deposit_record.json. Without --publish the drafts are left for review in the Zenodo
web interface and nothing is written to release.json.

The run is resumable. A draft that already exists for the concept is reused, a file already in the
draft with the same MD5 is skipped, and a file of the same name with a different MD5 is replaced,
so a run interrupted during a long upload can simply be started again.

Authentication is a Zenodo personal access token with the scopes deposit:write and
deposit:actions (https://zenodo.org/account/settings/applications/tokens/new/), read from the
environment variable ZENODO_TOKEN or, failing that, from the file named by --token-file
(default ~/.zenodo_token). The token is never placed on a command line.

Usage:
    python3 tools/zenodo_deposit.py plan --assets DIR         # no token needed: what would be sent
    python3 tools/zenodo_deposit.py status                    # the records and any open drafts
    python3 tools/zenodo_deposit.py code --publish
    python3 tools/zenodo_deposit.py data --assets DIR --publish
    python3 tools/zenodo_deposit.py all  --assets DIR --publish

DIR is the directory holding preds_<family>.tar (the release assets); the checksum lists are read
from the checkout. --sandbox targets sandbox.zenodo.org (with a sandbox token and the sandbox
concept identifiers given by --code-concept and --data-concept).

Requires Python 3.8 or later and curl (used for the file uploads).
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

OWNER = "thiptanawat"
REPO_NAME = "MetaGNN-Framework"
GITHUB = "https://github.com/%s/%s" % (OWNER, REPO_NAME)

CODE_CONCEPT = "21217568"   # concept record id of 10.5281/zenodo.21217568
DATA_CONCEPT = "21217578"   # concept record id of 10.5281/zenodo.21217578

FAMILIES = ["dh", "ctrl", "rewire", "fam", "ivr", "synth", "ht29", "seedrep", "rankgnn", "permute"]
ARRAY_LIST = os.path.join("paper1", "results", "prediction_arrays.sha256")
TARBALL_LIST = os.path.join("paper1", "results", "prediction_array_tarballs.sha256")

RECORD_FILE = os.path.join(HERE, "zenodo_deposit_record.json")
STATE_FILE = os.path.join(HERE, ".zenodo_state.json")
RELEASE_FILE = os.path.join(REPO, "release.json")

# The metadata fields the deposit API accepts back, and the keys of their list entries. Anything
# else a draft reports is dropped before the metadata is sent, so an inherited field the loader
# does not know cannot fail the update.
METADATA_KEYS = (
    "upload_type", "publication_type", "image_type", "publication_date", "title", "creators",
    "description", "access_right", "license", "embargo_date", "access_conditions", "keywords",
    "notes", "related_identifiers", "contributors", "references", "communities", "grants",
    "subjects", "version", "language", "locations", "dates", "method",
)
ENTRY_KEYS = {
    "creators": ("name", "affiliation", "orcid", "gnd"),
    "contributors": ("name", "type", "affiliation", "orcid", "gnd"),
    "related_identifiers": ("identifier", "relation", "resource_type", "scheme"),
    "communities": ("identifier",),
    "grants": ("id",),
    "subjects": ("term", "identifier", "scheme"),
    "locations": ("lat", "lon", "place", "description"),
    "dates": ("start", "end", "type", "description"),
}


# ---------------------------------------------------------------- helpers

def log(msg):
    print("[%s] %s" % (_dt.datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def die(msg):
    print("error: " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def file_digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sha_list(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, name = line.split(None, 1)
            out[name.strip().lstrip("*")] = digest.lower()
    return out


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as fh:
        return json.load(fh)


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


def human(n):
    """Decimal units (1 GB = 10^9 bytes), the convention of the manuscripts and guides."""
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return ("%d B" % n) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1000.0


def arrays_total_gb():
    """The released arrays' total, from the committed companion of the checksum lists."""
    meta = load_json(os.path.join(REPO, "paper1", "results", "prediction_arrays.json"), {})
    total = meta.get("tarballs_total_bytes") or meta.get("arrays_total_bytes")
    return "%.1f" % (total / 1e9) if total else "2.9"


def md5_of_entry(entry):
    c = entry.get("checksum") or ""
    return c.split(":", 1)[1] if ":" in c else c


def is_open_draft(d):
    if not d or not d.get("id"):
        return False
    if d.get("submitted") is True:
        return False
    return d.get("state") in (None, "unsubmitted", "inprogress")


# ---------------------------------------------------------------- HTTP

class Zenodo:
    def __init__(self, token, sandbox=False):
        self.base = "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"
        self.site = "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"
        self.token = token

    def request(self, method, url, body=None, auth=True, expect=(200, 201, 202, 204), tries=6):
        if url.startswith("/"):
            url = self.base + url
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if auth:
            if not self.token:
                die("a token is needed for this step (ZENODO_TOKEN or --token-file)")
            headers["Authorization"] = "Bearer " + self.token
        delay = 5
        last = None
        for attempt in range(1, tries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw = resp.read()
                    code = resp.getcode()
                    if code not in expect:
                        die("%s %s returned %d: %s" % (method, url, code, raw[:500]))
                    return json.loads(raw) if raw.strip() else {}
            except urllib.error.HTTPError as e:
                raw = e.read()
                if e.code in expect:
                    return {}
                if e.code in (429, 500, 502, 503, 504) and attempt < tries:
                    last = "%d %s" % (e.code, raw[:200])
                else:
                    die("%s %s returned %d: %s" % (method, url, e.code, raw[:1000]))
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt >= tries:
                    die("%s %s failed: %s" % (method, url, e))
                last = str(e)
            log("  transient failure (%s), retrying in %ds (%d/%d)" % (last, delay, attempt, tries))
            time.sleep(delay)
            delay = min(delay * 2, 120)

    # public records API (no token) ------------------------------------
    def latest_record(self, concept_id, tries=6):
        return self.request("GET", "/records/%s" % concept_id, auth=False, tries=tries)

    def record(self, record_id, tries=6):
        return self.request("GET", "/records/%s" % record_id, auth=False, tries=tries)

    # deposit API ------------------------------------------------------
    def deposition(self, dep_id, tries=6):
        return self.request("GET", "/deposit/depositions/%s" % dep_id, tries=tries)

    def new_version(self, dep_id):
        return self.request("POST", "/deposit/depositions/%s/actions/newversion" % dep_id)

    def files(self, dep_id):
        return self.request("GET", "/deposit/depositions/%s/files" % dep_id)

    def delete_file(self, dep_id, file_id):
        return self.request("DELETE", "/deposit/depositions/%s/files/%s" % (dep_id, file_id))

    def import_previous_files(self, dep_id):
        return self.request("POST", "/records/%s/draft/actions/files-import" % dep_id, expect=(200, 201))

    def set_metadata(self, dep_id, metadata):
        return self.request("PUT", "/deposit/depositions/%s" % dep_id, body={"metadata": metadata})

    def publish(self, dep_id):
        return self.request("POST", "/deposit/depositions/%s/actions/publish" % dep_id)

    def _curl(self, args, tries, what):
        """Run curl with the token in a config read from stdin (not on the command line); returns
        the parsed JSON body on HTTP 200/201."""
        if shutil.which("curl") is None:
            die("curl is required for uploads")
        for attempt in range(1, tries + 1):
            fd, out_path = tempfile.mkstemp(prefix="zenodo_", suffix=".json")
            os.close(fd)
            cfg = 'header = "Authorization: Bearer %s"\n' % self.token
            cmd = ["curl", "-sS", "-K", "-", "-o", out_path, "-w", "%{http_code}"] + args
            proc = subprocess.run(cmd, input=cfg, capture_output=True, text=True)
            code = proc.stdout.strip()[-3:]
            body = open(out_path).read()
            os.unlink(out_path)
            if proc.returncode == 0 and code in ("200", "201"):
                return json.loads(body)
            log("  %s failed (curl exit %d, HTTP %s): %s" % (
                what, proc.returncode, code or "none", (proc.stderr or body)[:300]))
            if attempt < tries:
                time.sleep(15 * attempt)
        die("%s did not succeed after %d attempts" % (what, tries))

    def upload(self, draft, path, name, tries=3):
        """PUT one file into the draft's bucket; falls back to the multipart files endpoint when
        the draft reports no bucket link. Returns the file entry Zenodo reports."""
        bucket = (draft.get("links") or {}).get("bucket")
        if bucket:
            url = "%s/%s" % (bucket.rstrip("/"), name)
            return self._curl(["-X", "PUT", "--upload-file", path,
                               "-H", "Content-Type: application/octet-stream", url], tries, "upload of " + name)
        url = "%s/deposit/depositions/%s/files" % (self.base, draft["id"])
        return self._curl(["-X", "POST", "-F", "name=%s" % name, "-F", "file=@%s" % path, url], tries, "upload of " + name)


# ---------------------------------------------------------------- local inputs

def release_info():
    rel = load_json(RELEASE_FILE)
    if not rel or not rel.get("tag"):
        die("release.json is missing or has no tag")
    return rel


def make_snapshot(tag, commit, workdir):
    """The archive of the tagged tree: git archive when the tag is here, the host's archive of the
    same tag otherwise. Returns (path, sha256, route)."""
    os.makedirs(workdir, exist_ok=True)
    name = "%s-%s.zip" % (REPO_NAME, tag)
    path = os.path.join(workdir, name)
    have_tag = subprocess.run(["git", "-C", REPO, "rev-parse", "--verify", "--quiet", tag + "^{commit}"],
                              capture_output=True, text=True)
    if have_tag.returncode == 0:
        resolved = have_tag.stdout.strip()
        if commit and resolved != commit:
            die("tag %s resolves to %s here but release.json records %s" % (tag, resolved, commit))
        if os.path.exists(path):
            os.unlink(path)
        subprocess.run(["git", "-C", REPO, "archive", "--format=zip",
                        "--prefix=%s-%s/" % (REPO_NAME, tag), "-o", path, tag], check=True)
        route = "git archive of %s (%s) in this checkout" % (tag, resolved)
    else:
        url = "%s/archive/refs/tags/%s.zip" % (GITHUB, tag)
        log("tag %s is not in this checkout; downloading %s" % (tag, url))
        subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "-o", path, url], check=True)
        route = "archive of %s downloaded from %s" % (tag, url)
    sha = file_digest(path, "sha256")
    comment = zipfile.ZipFile(path).comment.decode("ascii", "replace").strip()
    resolved = comment if len(comment) == 40 and all(c in "0123456789abcdef" for c in comment) else ""
    if commit and resolved and resolved != commit:
        die("the archive is of commit %s but release.json records %s" % (resolved, commit))
    return path, sha, route, (commit or resolved)


def resolve_commit(rel, workdir):
    """The commit the tag points at: release.json when it is filled there, otherwise the
    identifier every archive of the tag carries in its comment."""
    if rel.get("commit"):
        return rel["commit"]
    _, _, _, commit = make_snapshot(rel["tag"], "", workdir)
    if not commit:
        die("could not resolve the commit of tag %s" % rel["tag"])
    return commit


def data_assets(assets_dir):
    """The tarballs and the two checksum lists, each verified before it is offered."""
    if not assets_dir:
        die("--assets DIR is required for the data record")
    tarball_list = os.path.join(REPO, TARBALL_LIST)
    array_list = os.path.join(REPO, ARRAY_LIST)
    for p in (tarball_list, array_list):
        if not os.path.exists(p):
            die("missing committed checksum list %s" % p)
    expected = read_sha_list(tarball_list)
    items = []
    for fam in FAMILIES:
        name = "preds_%s.tar" % fam
        path = os.path.join(assets_dir, name)
        if not os.path.exists(path):
            die("missing %s in %s" % (name, assets_dir))
        if name not in expected:
            die("%s is not listed in %s" % (name, TARBALL_LIST))
        items.append({"name": name, "path": path, "sha256_expected": expected[name]})
    items.append({"name": os.path.basename(array_list), "path": array_list, "sha256_expected": None})
    items.append({"name": os.path.basename(tarball_list), "path": tarball_list, "sha256_expected": None})
    return items


def verify_assets(items):
    for it in items:
        it["size"] = os.path.getsize(it["path"])
        it["sha256"] = file_digest(it["path"], "sha256")
        it["md5"] = file_digest(it["path"], "md5")
        if it["sha256_expected"] and it["sha256"] != it["sha256_expected"]:
            die("%s: SHA-256 %s does not match the committed %s" % (it["name"], it["sha256"], it["sha256_expected"]))
    return items


# ---------------------------------------------------------------- metadata

def clean_metadata(md):
    """The inherited metadata of a draft reduced to the fields the deposit API accepts back."""
    out = {}
    for k in METADATA_KEYS:
        if k not in md or md[k] in (None, "", [], {}):
            continue
        v = md[k]
        if k in ENTRY_KEYS and isinstance(v, list):
            v = [{kk: e[kk] for kk in ENTRY_KEYS[k] if kk in e and e[kk] not in (None, "")}
                 for e in v if isinstance(e, dict)]
            v = [e for e in v if e]
        out[k] = v
    lic = out.get("license")
    if isinstance(lic, dict):
        out["license"] = lic.get("id")
    return out


def merge_related(existing, additions):
    seen = {(r.get("identifier"), r.get("relation")) for r in existing or []}
    out = list(existing or [])
    for r in additions:
        if (r["identifier"], r["relation"]) not in seen:
            out.append(r)
    return out


def code_metadata(inherited, rel, commit, snapshot_sha, snapshot_name, today, data_concept_doi):
    md = clean_metadata(inherited)
    tag = rel["tag"]
    md["upload_type"] = "software"
    md["title"] = ("MetaGNN-Framework: metabench and the benchmark-validity studies for metabolic "
                   "reaction activity scoring (%s)" % tag)
    md["version"] = tag
    md["publication_date"] = today
    md["access_right"] = "open"
    md.setdefault("license", "mit-license")
    keywords = list(md.get("keywords") or [])
    for k in ("benchmark validity", "genome-scale metabolic models", "graph neural networks",
              "data leakage", "cross-validation", "language models", "reproducibility"):
        if k not in keywords:
            keywords.append(k)
    md["keywords"] = keywords
    md["description"] = (
        "<p>The tagged tree <code>%s</code> (commit <code>%s</code>) of "
        "<a href=\"%s\">%s</a>: the code and committed results of two benchmark-validity studies of "
        "transcriptome-based metabolic reaction activity scoring on Recon3D, the <code>metabench</code> "
        "checks they share, the scorer they audit, the reproduction guides, and a single driver "
        "(<code>reproduce.sh</code>) that regenerates every reported number from the committed result "
        "files with no GPU and no network.</p>"
        "<p>Contents: <code>paper1/</code>, the double hold-out study with its input-substitution arms, "
        "the family-disjoint split, the synthetic-world check, the external audit of a published "
        "predictor and the per-cell result files; <code>paper2/</code>, the language-model study with "
        "its prompt builders, archived model responses, external-cohort panels, analysis plan, "
        "departure log, deployment and model records; <code>metabench/</code>, <code>docs/</code>, "
        "<code>tools/</code>. The manuscripts under review are not part of this archive.</p>"
        "<p>The per-cell prediction arrays of the first study (ten tarballs, %s GB) are not in this "
        "archive: they are attached to the release <code>%s</code> on the repository host and "
        "deposited with the curated cohort in the data record "
        "<a href=\"https://doi.org/%s\">%s</a>; their SHA-256 lists are committed under "
        "<code>paper1/results/</code> and <code>paper1/code/verify_ledger.py</code> checks a copy.</p>"
        "<p>File: <code>%s</code>, SHA-256 <code>%s</code>. The previous version of this record held "
        "the scorer alone, as it stood in July 2026.</p>"
        % (tag, commit, GITHUB, GITHUB, arrays_total_gb(), tag, data_concept_doi, data_concept_doi, snapshot_name, snapshot_sha))
    md["related_identifiers"] = merge_related(md.get("related_identifiers"), [
        {"identifier": "%s/tree/%s" % (GITHUB, tag), "relation": "isSupplementTo", "resource_type": "software"},
        {"identifier": data_concept_doi, "relation": "requires", "resource_type": "dataset"},
    ])
    return md


def data_metadata(inherited, rel, commit, items, today, code_concept_doi):
    md = clean_metadata(inherited)
    tag = rel["tag"]
    md["upload_type"] = "dataset"
    md["version"] = tag
    md["publication_date"] = today
    md["access_right"] = "open"
    md.setdefault("license", "cc-by-4.0")
    md.setdefault("title", "MetaGNN-Framework data: the curated TCGA-COAD/READ cohort and the prediction arrays of the benchmark-validity study")
    keywords = list(md.get("keywords") or [])
    for k in ("benchmark validity", "prediction arrays", "double hold-out"):
        if k not in keywords:
            keywords.append(k)
    md["keywords"] = keywords
    rows = "".join("<li><code>%s</code> (%s, SHA-256 <code>%s</code>)</li>" % (it["name"], human(it["size"]), it["sha256"])
                   for it in items)
    old = md.get("description") or ""
    marker = "<p><strong>Version %s</strong>" % tag
    if marker not in old:
        md["description"] = old + (
            "%s adds the per-cell prediction arrays of the benchmark-validity study built on this "
            "cohort, as ten tarballs, one per result family, together with the two committed checksum "
            "lists (<code>prediction_arrays.sha256</code> for every array, "
            "<code>prediction_array_tarballs.sha256</code> for every tarball). Unpack the tarballs under "
            "<code>paper1/results/prediction_arrays/</code> of the tagged tree <code>%s</code> (commit "
            "<code>%s</code>) of <a href=\"%s\">%s</a> and run "
            "<code>python3 paper1/code/verify_ledger.py</code> to check every array. Every file of the "
            "previous version is kept unchanged.</p><ul>%s</ul>"
            % (marker, tag, commit, GITHUB, GITHUB, rows))
    md["related_identifiers"] = merge_related(md.get("related_identifiers"), [
        {"identifier": "%s/tree/%s" % (GITHUB, tag), "relation": "isSupplementTo", "resource_type": "software"},
        {"identifier": code_concept_doi, "relation": "isRequiredBy", "resource_type": "software"},
    ])
    return md


# ---------------------------------------------------------------- deposit

def open_draft(z, concept_id, state, key):
    """The draft of the next version of the concept: the one recorded in the state file if it is
    still open, otherwise the one Zenodo returns for a new-version request (Zenodo returns the
    existing open draft when there is one). Returns (draft, previous published record)."""
    previous = z.latest_record(concept_id)
    log("latest published version of concept %s is record %s (%s)" % (concept_id, previous["id"], previous.get("doi")))
    dep_id = (state.get(key) or {}).get("draft_id")
    if dep_id:
        d = z.request("GET", "/deposit/depositions/%s" % dep_id, expect=(200, 403, 404, 410), tries=3)
        if is_open_draft(d):
            log("reusing open draft %s" % dep_id)
            return d, previous
        log("recorded draft %s is no longer open; asking for a new version" % dep_id)
    resp = z.new_version(previous["id"])
    draft_url = (resp.get("links") or {}).get("latest_draft")
    if draft_url:
        draft = z.request("GET", draft_url)
    elif is_open_draft(resp) and str(resp.get("id")) != str(previous["id"]):
        draft = resp
    else:
        die("new-version response carries no draft: %s" % json.dumps(resp)[:500])
    if not is_open_draft(draft):
        die("the new-version draft %s is not open for editing" % draft.get("id"))
    state.setdefault(key, {})["draft_id"] = draft["id"]
    save_json(STATE_FILE, state)
    log("draft %s opened (%s)" % (draft["id"], (draft.get("links") or {}).get("html", "")))
    return draft, previous


def sync_files(z, draft, wanted, keep_others):
    """Bring the draft's file set to `wanted` ({name: {path, md5}}): skip what matches, replace what
    differs, remove inherited files not wanted unless keep_others. Returns {name: md5} after."""
    dep_id = draft["id"]
    present = {f["filename"]: f for f in z.files(dep_id)}
    for name, entry in present.items():
        if name in wanted:
            if md5_of_entry(entry) == wanted[name]["md5"]:
                log("  %s already in the draft with the same MD5, kept" % name)
                wanted[name]["done"] = True
            else:
                log("  %s in the draft differs, removing it" % name)
                z.delete_file(dep_id, entry["id"])
        elif not keep_others:
            log("  removing inherited %s" % name)
            z.delete_file(dep_id, entry["id"])
        else:
            log("  keeping inherited %s" % name)
    for name, w in wanted.items():
        if w.get("done"):
            continue
        log("  uploading %s (%s)" % (name, human(os.path.getsize(w["path"]))))
        t0 = time.time()
        result = z.upload(draft, w["path"], name)
        got = md5_of_entry(result)
        if got and got != w["md5"]:
            die("%s: Zenodo reports MD5 %s, local MD5 is %s" % (name, got, w["md5"]))
        log("  done in %.0fs%s" % (time.time() - t0, ", MD5 verified" if got else ""))
    final = {f["filename"]: md5_of_entry(f) for f in z.files(dep_id)}
    for name, w in wanted.items():
        if final.get(name) != w["md5"]:
            die("after upload, %s is not in the draft with the expected MD5" % name)
    return final


def ensure_inherited_files(z, draft, previous):
    """A new version must carry the previous version's files. When the draft opened empty, import
    them; when they are present, they are checked by name and MD5 before anything is published."""
    prev_files = {f["key"]: md5_of_entry(f) for f in (previous.get("files") or [])}
    if not prev_files:
        return {}
    present = {f["filename"]: md5_of_entry(f) for f in z.files(draft["id"])}
    if not any(k in present for k in prev_files):
        log("draft has none of the previous version's %d files; importing them" % len(prev_files))
        z.import_previous_files(draft["id"])
        present = {f["filename"]: md5_of_entry(f) for f in z.files(draft["id"])}
    missing = [k for k, m in prev_files.items() if present.get(k) != m]
    if missing:
        die("the draft lacks these files of the previous version, or they differ: %s" % ", ".join(missing))
    log("all %d files of the previous version are present and unchanged" % len(prev_files))
    return prev_files


def finish(z, draft, publish, record, key, doi_field):
    dep_id = draft["id"]
    if not publish:
        log("draft %s populated and not published; review it at %s" % (dep_id, (draft.get("links") or {}).get("html", z.site)))
        record[key]["draft_id"] = dep_id
        record[key]["published"] = False
        return None
    pub = z.publish(dep_id)
    doi = pub.get("doi") or (pub.get("metadata") or {}).get("doi")
    if not doi:
        die("publish returned no DOI: %s" % json.dumps(pub)[:500])
    links = pub.get("links") or {}
    html = links.get("record_html") or links.get("html") or "%s/records/%s" % (z.site, pub.get("id"))
    log("published: %s  (%s)" % (doi, html))
    rel = load_json(RELEASE_FILE)
    rel[doi_field] = doi
    save_json(RELEASE_FILE, rel)
    record[key].update({"record_id": pub.get("id"), "doi": doi, "conceptdoi": pub.get("conceptdoi"),
                        "url": html, "published": True, "published_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    return doi


def deposit_code(z, rel, args, record, state, today):
    log("== code record: new version of concept %s" % args.code_concept)
    path, sha, route, commit = make_snapshot(rel["tag"], rel.get("commit", ""), args.workdir)
    name = os.path.basename(path)
    log("snapshot %s (%s), SHA-256 %s, commit %s, %s" % (name, human(os.path.getsize(path)), sha, commit, route))
    draft, previous = open_draft(z, args.code_concept, state, "code")
    md = code_metadata(draft.get("metadata") or {}, rel, commit, sha, name, today, "10.5281/zenodo.%s" % args.data_concept)
    wanted = {name: {"path": path, "md5": file_digest(path, "md5")}}
    final = sync_files(z, draft, wanted, keep_others=False)
    if sorted(final) != [name]:
        die("the code draft should hold exactly %s, it holds %s" % (name, sorted(final)))
    z.set_metadata(draft["id"], md)
    log("metadata set: %s" % md["title"])
    record["code"] = {"concept": args.code_concept, "previous_record": previous.get("id"),
                      "snapshot": name, "snapshot_sha256": sha, "snapshot_size": os.path.getsize(path),
                      "snapshot_route": route, "tag": rel["tag"], "commit": commit}
    finish(z, z.deposition(draft["id"]), args.publish, record, "code", "version_doi_code")


def deposit_data(z, rel, args, record, state, today):
    log("== data record: new version of concept %s" % args.data_concept)
    items = verify_assets(data_assets(args.assets))
    log("assets verified against %s: %d files, %s" % (
        TARBALL_LIST, len(items), human(sum(it["size"] for it in items))))
    commit = resolve_commit(rel, args.workdir)
    draft, previous = open_draft(z, args.data_concept, state, "data")
    prev_files = ensure_inherited_files(z, draft, previous)
    clash = [it["name"] for it in items if it["name"] in prev_files]
    if clash:
        die("these names already exist in the previous version and would be replaced: %s" % ", ".join(clash))
    md = data_metadata(draft.get("metadata") or {}, rel, commit, items, today, "10.5281/zenodo.%s" % args.code_concept)
    wanted = {it["name"]: {"path": it["path"], "md5": it["md5"]} for it in items}
    final = sync_files(z, draft, wanted, keep_others=True)
    ensure_inherited_files(z, draft, previous)
    z.set_metadata(draft["id"], md)
    log("metadata set: %s" % md.get("title"))
    record["data"] = {"concept": args.data_concept, "previous_record": previous.get("id"),
                      "tag": rel["tag"], "commit": commit,
                      "files_added": [{"name": it["name"], "size": it["size"], "sha256": it["sha256"], "md5": it["md5"]} for it in items],
                      "files_kept": sorted(prev_files), "files_in_version": sorted(final)}
    finish(z, z.deposition(draft["id"]), args.publish, record, "data", "version_doi_data")


# ---------------------------------------------------------------- commands

def cmd_plan(z, rel, args):
    path, sha, route, commit = make_snapshot(rel["tag"], rel.get("commit", ""), args.workdir)
    print("release.json: tag %s, commit %s%s" % (rel["tag"], commit, "" if rel.get("commit") else " (resolved from the archive)"))
    print("code concept %s -> new version with one file:" % args.code_concept)
    print("  %s  %s  sha256 %s\n  (%s)" % (os.path.basename(path), human(os.path.getsize(path)), sha, route))
    if args.assets:
        items = verify_assets(data_assets(args.assets))
        print("data concept %s -> new version keeping the previous files and adding:" % args.data_concept)
        for it in items:
            print("  %-36s %10s  sha256 %s" % (it["name"], human(it["size"]), it["sha256"]))
        print("  total %s; every tarball matches %s" % (human(sum(it["size"] for it in items)), TARBALL_LIST))
    else:
        print("data concept %s: pass --assets DIR to list the tarballs that would be added" % args.data_concept)
    for cid, label in ((args.code_concept, "code"), (args.data_concept, "data")):
        try:
            r = z.latest_record(cid, tries=2)
            print("%s record now: %s, '%s', %d file(s)" % (
                label, r.get("doi"), (r.get("metadata") or {}).get("title"), len(r.get("files") or [])))
        except SystemExit:
            print("%s record: could not be read right now (Zenodo unreachable); the deposit step reads it again" % label)
    print("nothing was sent.")


def cmd_status(z, rel, args):
    state = load_json(STATE_FILE, {})
    for cid, key in ((args.code_concept, "code"), (args.data_concept, "data")):
        r = z.latest_record(cid, tries=3)
        md = r.get("metadata") or {}
        print("%s: latest %s (record %s), version %s, %d file(s)" % (
            key, r.get("doi"), r.get("id"), md.get("version"), len(r.get("files") or [])))
        dep_id = (state.get(key) or {}).get("draft_id")
        if dep_id and z.token:
            d = z.request("GET", "/deposit/depositions/%s" % dep_id, expect=(200, 403, 404, 410), tries=2)
            if d and d.get("id"):
                print("  draft %s: %s, %d file(s), %s" % (
                    dep_id, "open" if is_open_draft(d) else "published", len(d.get("files") or []),
                    (d.get("links") or {}).get("html", "")))
    print("release.json: version_doi_code=%r version_doi_data=%r" % (rel.get("version_doi_code"), rel.get("version_doi_data")))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["plan", "status", "code", "data", "all"])
    ap.add_argument("--assets", help="directory holding preds_<family>.tar")
    ap.add_argument("--publish", action="store_true", help="publish the populated drafts and write the DOIs to release.json")
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    ap.add_argument("--code-concept", default=CODE_CONCEPT)
    ap.add_argument("--data-concept", default=DATA_CONCEPT)
    ap.add_argument("--token-file", default=os.path.expanduser("~/.zenodo_token"))
    ap.add_argument("--workdir", default=os.path.join(REPO, ".zenodo_work"))
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token and os.path.exists(args.token_file):
        token = open(args.token_file).read().strip()
    z = Zenodo(token, sandbox=args.sandbox)
    rel = release_info()

    if args.command == "plan":
        return cmd_plan(z, rel, args)
    if args.command == "status":
        return cmd_status(z, rel, args)
    if not token:
        die("no token: set ZENODO_TOKEN or write it to %s" % args.token_file)
    if args.command in ("data", "all") and not args.assets:
        die("--assets DIR is required for the data record")

    today = _dt.date.today().isoformat()
    state = load_json(STATE_FILE, {})
    record = load_json(RECORD_FILE, {})
    record["generated"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record["site"] = z.site
    try:
        if args.command in ("code", "all"):
            deposit_code(z, rel, args, record, state, today)
        if args.command in ("data", "all"):
            deposit_data(z, rel, args, record, state, today)
    finally:
        save_json(RECORD_FILE, record)
    if args.publish:
        rel_now = load_json(RELEASE_FILE)
        log("release.json now: version_doi_code=%r version_doi_data=%r" % (
            rel_now.get("version_doi_code"), rel_now.get("version_doi_data")))
        log("record written to %s" % os.path.relpath(RECORD_FILE, REPO))
    else:
        log("drafts left unpublished; rerun with --publish to publish them")


if __name__ == "__main__":
    main()
