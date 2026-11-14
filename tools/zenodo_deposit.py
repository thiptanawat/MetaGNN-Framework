#!/usr/bin/env python3
"""Mint the Zenodo version DOIs for a tagged release: a new version of the code record and a new
version of the data record, both under the concept identifiers the manuscripts cite.

What it does, per record:

  code  a new version of the code record (concept 10.5281/zenodo.21217568) holding one file, the
        archive of the tagged tree (`git archive` of the tag named in release.json, or the same
        archive downloaded from the repository host when this checkout does not have the tag).
        The inherited file of the previous version is removed; the description names the commit,
        the tag and the SHA-256 of the archive, and the title carries no version (the version
        field does).
  data  a new version of the data record (concept 10.5281/zenodo.21217578) that carries the
        per-cell prediction arrays of the reaction-scoring audit, the ten tarballs, one per result family, and
        the two committed checksum lists, beside the files of the previous version. Each tarball is
        checked against paper1/results/prediction_array_tarballs.sha256 before it is sent; a file
        the previous version already holds with the same MD5 is kept and not sent again. With
        --cohort-archive FILE the version also carries that cohort archive, and --retire NAME drops
        a file of the previous version that the archive replaces; the description of the record
        is then written from tools/zenodo_data_description.html. The version is not published
        unless every file of the previous version, other than the retired ones, is present with an
        unchanged MD5.

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
    python3 tools/zenodo_deposit.py status                    # the records and, with a token, what any open draft holds
    python3 tools/zenodo_deposit.py code --publish
    python3 tools/zenodo_deposit.py data --assets DIR --publish
    python3 tools/zenodo_deposit.py all  --assets DIR --publish
    python3 tools/zenodo_deposit.py data --assets DIR --cohort-archive metagnn-data-v2.zip \
                                         --retire metagnn-data.zip --publish
    python3 tools/zenodo_deposit.py describe                  # title and description of the open code draft; no file is touched
    python3 tools/zenodo_deposit.py describe --record ID --publish   # the same for a published version of the code record, edited in place
    python3 tools/zenodo_deposit.py describe-data --record ID --publish   # the text of a published version of the data record, for this release

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
import re
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

PREPRINT_DOI = "10.64898/2026.09.02.748315"
DATA_DESCRIPTION = os.path.join(HERE, "zenodo_data_description.html")

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


def data_assets(assets_dir, cohort_archive=None):
    """The tarballs and the two checksum lists, each verified before it is offered, and the cohort
    archive when this version carries one."""
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
    if cohort_archive:
        if not os.path.isfile(cohort_archive):
            die("the cohort archive %s is not a file" % cohort_archive)
        items.insert(0, {"name": os.path.basename(cohort_archive), "path": cohort_archive,
                         "sha256_expected": None, "role": "cohort"})
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


def code_description(tag, commit, snapshot_sha, snapshot_name, data_concept_doi):
    """The description of a version of the code record. It names the studies by what they do and the
    version by its commit, so that it reads the same whatever the tag is called. The two analyses
    added in benchmark-validity-2026-11c and their extensions in benchmark-validity-2026-11d are named
    from those releases on, so that the text of an earlier version, when it is set again, does not list
    what that version does not hold."""
    since_11c = not (tag.startswith("journal-") or tag == "benchmark-validity-2026-11b")
    since_11d = since_11c and tag != "benchmark-validity-2026-11c"
    cohort2 = ("the linear rows and the graph scorer repeated on a second cohort with their plan, " if since_11d else
               "the linear rows repeated on a second cohort with their plan, " if since_11c else "")
    replication = ("a replication of the own-versus-donor contrast on the three language models with its plans "
                   "and the records of its collections, " if since_11d else
                   "a replication of the own-versus-donor contrast with its plan and the record of its "
                   "collection, " if since_11c else "")
    return (
        "<p>This record archives the MetaGNN-Framework repository: a graph neural network scorer of "
        "metabolic reaction activity, the <code>metabench</code> library of benchmark-validity checks, "
        "and the code and committed results of two studies of transcriptome-based reaction scoring on "
        "the Recon3D human metabolic reconstruction. The first study audits patient-specific reaction "
        "scoring with joint holdouts of patients and reactions and with input-substitution controls. "
        "The second tests when language models use sample-specific molecular evidence, with controlled "
        "interventions and external cohorts.</p>"
        "<p><strong>Reproduction.</strong> One driver, <code>reproduce.sh</code>, regenerates every "
        "reported number, table and figure from the committed result files on a CPU in about half an "
        "hour, with no GPU and no network access. The software versions are pinned in "
        "<code>requirements.txt</code>, and the guides in <code>docs/</code> describe each level of "
        "reproduction.</p>"
        "<p><strong>Contents.</strong> For the reaction-scoring audit: the joint-holdout driver, the "
        "input-substitution arms, the family-disjoint reaction split, a synthetic check with known "
        "answers, the audit of an independently published predictor, " + cohort2 + "and the per-cell "
        "result files. "
        "For the language-model study: the prompt builders, the archived model responses, the "
        "external-cohort panels, the analysis plan with its departure log, " + replication + "and the "
        "deployment and model records. Shared by both: the scorer, the <code>metabench</code> checks, the reproduction guides "
        "and the release tools.</p>"
        "<p><strong>Related data.</strong> The per-cell prediction arrays of the reaction-scoring audit "
        "(ten tarballs, %s GB) are not part of this archive. They are deposited with the curated cohort "
        "in the data record <a href=\"https://doi.org/%s\">%s</a> and attached to the matching release "
        "on GitHub. Their SHA-256 lists are committed in this archive, and a script in the repository "
        "checks a downloaded copy against them.</p>"
        "<p><strong>This version.</strong> Source: <a href=\"%s\">%s</a>, commit <code>%s</code> "
        "(Git tag <code>%s</code>). File: <code>%s</code>, SHA-256 <code>%s</code>. Earlier versions of "
        "this record hold earlier states of the repository; the first version held the scorer alone, as "
        "it stood in July 2026.</p>"
        % (arrays_total_gb(), data_concept_doi, data_concept_doi, GITHUB, GITHUB, commit, tag, snapshot_name, snapshot_sha))


CODE_TITLE = "MetaGNN-Framework: metabench and the benchmark-validity studies for metabolic reaction activity scoring"


def code_metadata(inherited, rel, commit, snapshot_sha, snapshot_name, today, data_concept_doi):
    md = clean_metadata(inherited)
    tag = rel["tag"]
    md["upload_type"] = "software"
    md["title"] = CODE_TITLE          # the version is in the version field, not in the title
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
    md["description"] = code_description(tag, commit, snapshot_sha, snapshot_name, data_concept_doi)
    # the tree link of an earlier tag belongs to the version it was minted for
    inherited_related = [r for r in (md.get("related_identifiers") or [])
                         if not str(r.get("identifier", "")).startswith(GITHUB + "/tree/")]
    md["related_identifiers"] = merge_related(inherited_related, [
        {"identifier": "%s/tree/%s" % (GITHUB, tag), "relation": "isSupplementTo", "resource_type": "software"},
        {"identifier": data_concept_doi, "relation": "requires", "resource_type": "dataset"},
    ])
    return md


def data_description(rel, commit, items, retired, previous_doi, revision_note=""):
    """The description of a version that carries a cohort archive, written from the committed
    template so that what the record says can be read in the repository."""
    cohort = [it for it in items if it.get("role") == "cohort"][0]
    rows = "".join("<li><code>%s</code> (%s, SHA-256 <code>%s</code>)</li>" % (it["name"], human(it["size"]), it["sha256"])
                   for it in items if it.get("role") != "cohort")
    template = open(DATA_DESCRIPTION).read()
    return template.format(tag=rel["tag"], commit=commit, github=GITHUB, preprint=PREPRINT_DOI,
                           cohort_name=cohort["name"], cohort_size=human(cohort["size"]),
                           cohort_sha256=cohort["sha256"], array_rows=rows,
                           retired=", ".join("<code>%s</code>" % n for n in retired) or "none",
                           previous_doi=previous_doi or "", revision_note=revision_note)


def data_metadata(inherited, rel, commit, items, today, code_concept_doi, retired=(), previous_doi=None):
    md = clean_metadata(inherited)
    tag = rel["tag"]
    if any(it.get("role") == "cohort" for it in items):
        md["upload_type"] = "dataset"
        md["version"] = tag
        md["publication_date"] = today
        md["access_right"] = "open"
        md.setdefault("license", "cc-by-4.0")
        md["title"] = ("MetaGNN-Framework data: the curated TCGA-COAD/READ cohort, its fixed splits and "
                       "the prediction arrays of the benchmark-validity studies")
        keywords = list(md.get("keywords") or [])
        for k in ("benchmark validity", "prediction arrays", "double hold-out", "TCGA-COAD", "TCGA-READ", "Recon3D"):
            if k not in keywords:
                keywords.append(k)
        md["keywords"] = keywords
        md["description"] = data_description(rel, commit, items, list(retired), previous_doi)
        # the tree link of an earlier tag belongs to the version it was minted for
        inherited_related = [r for r in (md.get("related_identifiers") or [])
                             if not str(r.get("identifier", "")).startswith(GITHUB + "/tree/")]
        md["related_identifiers"] = merge_related(inherited_related, [
            {"identifier": "%s/tree/%s" % (GITHUB, tag), "relation": "isSupplementTo", "resource_type": "software"},
            {"identifier": code_concept_doi, "relation": "isRequiredBy", "resource_type": "software"},
            {"identifier": PREPRINT_DOI, "relation": "isSupplementTo", "resource_type": "publication-preprint"},
        ])
        return md
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
            "<code>prediction_array_tarballs.sha256</code> for every tarball). Unpack the tarballs into the "
            "directory <code>results/prediction_arrays/</code> of the reaction-scoring audit at commit "
            "<code>%s</code> of <a href=\"%s\">%s</a> (Git tag <code>%s</code>) and run that study's "
            "<code>verify_ledger.py</code> to check every array. Every file of the "
            "previous version is kept unchanged.</p><ul>%s</ul>"
            % (marker, commit, GITHUB, GITHUB, tag, rows))
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


def previous_files(previous):
    return {f["key"]: md5_of_entry(f) for f in (previous.get("files") or [])}


def import_inherited_files(z, draft, previous):
    """A draft that opened without any file of the previous version gets them imported."""
    prev_files = previous_files(previous)
    if not prev_files:
        return
    present = {f["filename"] for f in z.files(draft["id"])}
    if not any(k in present for k in prev_files):
        log("draft has none of the previous version's %d files; importing them" % len(prev_files))
        z.import_previous_files(draft["id"])


def ensure_inherited_files(z, draft, previous, retired=()):
    """A new version must carry the previous version's files, other than those it retires: they
    are checked by name and MD5 before anything is published, and a retired file must be gone."""
    prev_files = previous_files(previous)
    if not prev_files:
        return {}
    present = {f["filename"]: md5_of_entry(f) for f in z.files(draft["id"])}
    missing = [k for k, m in prev_files.items() if k not in retired and present.get(k) != m]
    if missing:
        die("the draft lacks these files of the previous version, or they differ: %s" % ", ".join(missing))
    still = [k for k in retired if k in present]
    if still:
        die("these retired files are still in the draft: %s" % ", ".join(still))
    log("all %d files kept from the previous version are present and unchanged%s" % (
        len(prev_files) - len([k for k in retired if k in prev_files]),
        (", %d retired" % len(retired)) if retired else ""))
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


def describe_code(z, rel, args, state, today):
    """Set the title and description of a version of the code record without touching its files.

    Without --record: the open draft of the next version (the one `code` works on), from this
    release. With --record ID: a published version, edited in place and published again when
    --publish is given; its tag, commit, file name and SHA-256 are read from the description it
    carries, so the text changes and the facts do not. Its DOI and its publication date stay."""
    data_doi = "10.5281/zenodo.%s" % args.data_concept
    if not args.record:
        dep_id = (state.get("code") or {}).get("draft_id")
        d = z.deposition(dep_id) if dep_id else None
        if not is_open_draft(d):
            die("no open draft of the code record is recorded; run `code` first")
        path, sha, route, commit = make_snapshot(rel["tag"], rel.get("commit", ""), args.workdir)
        md = code_metadata(d.get("metadata") or {}, rel, commit, sha, os.path.basename(path), today, data_doi)
        z.set_metadata(d["id"], md)
        log("draft %s: metadata set: %s (version %s)" % (d["id"], md["title"], md["version"]))
        return
    d = z.deposition(args.record)
    old = d.get("metadata") or {}
    text = old.get("description") or ""
    tag = old.get("version")
    m_commit = re.search(r"commit <code>([0-9a-f]{40})</code>", text)
    m_file = re.search(r"File: <code>([^<]+)</code>, SHA-256 <code>([0-9a-f]{64})</code>", text)
    if not (tag and m_commit and m_file):
        die("record %s: tag, commit or file are not in its metadata; nothing was changed" % args.record)
    md = clean_metadata(old)
    md["title"] = CODE_TITLE
    md["description"] = code_description(tag, m_commit.group(1), m_file.group(2), m_file.group(1), data_doi)
    log("record %s (%s, version %s): title and description rebuilt for commit %s, file %s" % (
        args.record, d.get("doi"), tag, m_commit.group(1)[:12], m_file.group(1)))
    if not args.publish:
        log("nothing was sent; rerun with --publish to edit the published record")
        return
    if d.get("submitted") and d.get("state") == "done":
        z.request("POST", "/deposit/depositions/%s/actions/edit" % args.record)
    z.set_metadata(args.record, md)
    pub = z.publish(args.record)
    log("record %s published again with the new title and description (DOI %s)" % (args.record, pub.get("doi")))


def long_date(iso):
    """2026-11-11 as 11 November 2026."""
    d = _dt.date.fromisoformat(iso)
    return "%d %s %d" % (d.day, d.strftime("%B"), d.year)


def describe_data(z, rel, args, today):
    """Rewrite the text of a published version of the data record for this release, in place.

    The version keeps its files, its DOI and its publication date. What its description says about
    its own files (the cohort archive, the tarballs with their sizes and SHA-256, the files it
    retired) is read back from that description, and the committed template is applied to those
    facts with the tag and the commit of this release, so the record names the release its files
    accompany now and says that it was revised. The version field and the tree link follow. Nothing
    is sent without --publish, and nothing is sent at all if any checksum would change."""
    if not args.record:
        die("describe-data needs --record ID, a published version of the data record")
    d = z.deposition(args.record)
    old = d.get("metadata") or {}
    text = old.get("description") or ""
    m_cohort = re.search(r"<li>\s*<code>([^<]+)</code>\s*\(([^()]+?),\s*SHA-256\s*<code>([0-9a-f]{64})</code>\)\.\s*<code>crc_624/</code>", text)
    # the list of the other files: every entry is a name, a size and a checksum and nothing more
    m_rows = re.search(r"<ul>\s*((?:<li>\s*<code>[^<]+</code>\s*\([^()]+?,\s*SHA-256\s*<code>[0-9a-f]{64}</code>\)\s*</li>\s*)+)</ul>", text)
    m_retired = re.search(r"earlier versions of this record \((.+?); July 2026", text)
    published = old.get("publication_date")
    if not (m_cohort and m_rows and m_retired and published):
        die("record %s: its description does not carry the file facts in the form of the template; nothing was changed" % args.record)
    commit = resolve_commit(rel, args.workdir)
    tag = rel["tag"]
    note = (" The files of this version have not changed since it was published on %s with an earlier "
            "release of the repository; the release named here uses the same files, and this text was "
            "revised on %s to name it." % (long_date(published), long_date(today)))
    new = open(DATA_DESCRIPTION).read().format(
        tag=tag, commit=commit, github=GITHUB, preprint=PREPRINT_DOI,
        cohort_name=m_cohort.group(1), cohort_size=m_cohort.group(2).strip(), cohort_sha256=m_cohort.group(3),
        array_rows=m_rows.group(1).strip(), retired=m_retired.group(1), previous_doi="", revision_note=note)
    sums = lambda s: sorted(re.findall(r"\b[0-9a-f]{64}\b", s))
    names = lambda s: sorted(set(re.findall(r"<code>([^<]+\.(?:zip|tar|sha256))</code>", s)))
    if sums(text) != sums(new) or names(text) != names(new):
        die("record %s: the rewritten text would not carry the same files and checksums; nothing was changed" % args.record)
    md = clean_metadata(old)
    md["version"] = tag
    md["description"] = new
    kept = [r for r in (md.get("related_identifiers") or [])
            if not str(r.get("identifier", "")).startswith(GITHUB + "/tree/")]
    md["related_identifiers"] = merge_related(kept, [
        {"identifier": "%s/tree/%s" % (GITHUB, tag), "relation": "isSupplementTo", "resource_type": "software"}])
    log("record %s (%s): version %s -> %s; %d checksums and %d file names carried over unchanged" % (
        args.record, d.get("doi"), old.get("version"), tag, len(sums(new)), len(names(new))))
    split = lambda s: [x for x in re.split(r"(?<=[.;:>])\s+|(?=<li>)|(?=<p>)", s) if x.strip()]
    import difflib
    for line in difflib.unified_diff(split(text), split(new), "published", "rewritten", lineterm="", n=0):
        print("    " + line)
    for r in md["related_identifiers"]:
        print("    related: %s %s" % (r.get("relation"), r.get("identifier")))
    if not args.publish:
        log("nothing was sent; rerun with --publish to edit the published record")
        return
    if d.get("submitted") and d.get("state") == "done":
        z.request("POST", "/deposit/depositions/%s/actions/edit" % args.record)
    z.set_metadata(args.record, md)
    pub = z.publish(args.record)
    log("record %s published again with the rewritten text (DOI %s, version %s)" % (
        args.record, pub.get("doi"), (pub.get("metadata") or {}).get("version")))
    record = load_json(RECORD_FILE, {})
    if str((record.get("data") or {}).get("record_id")) == str(args.record):
        # the record of the deposit keeps the tag the version was deposited for and gains what it says now
        record["data"]["version_label"] = tag
        record["data"]["text_revised_at"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        save_json(RECORD_FILE, record)
        log("record of the deposit updated: %s" % os.path.relpath(RECORD_FILE, REPO))


def deposit_data(z, rel, args, record, state, today):
    log("== data record: new version of concept %s" % args.data_concept)
    items = verify_assets(data_assets(args.assets, args.cohort_archive))
    log("assets verified against %s: %d files, %s" % (
        TARBALL_LIST, len(items), human(sum(it["size"] for it in items))))
    commit = resolve_commit(rel, args.workdir)
    draft, previous = open_draft(z, args.data_concept, state, "data")
    prev_files = previous_files(previous)
    retired = list(args.retire or [])
    unknown = [n for n in retired if n not in prev_files]
    if unknown:
        die("--retire names files the previous version does not hold: %s" % ", ".join(unknown))
    # a file the previous version already holds under the same name must be the same file
    clash = [it["name"] for it in items if it["name"] in prev_files and prev_files[it["name"]] != it["md5"]]
    if clash:
        die("these names exist in the previous version with a different MD5 and would be replaced: %s" % ", ".join(clash))
    if any(it["name"] in retired for it in items):
        die("a file cannot be both retired and deposited")
    import_inherited_files(z, draft, previous)
    for entry in z.files(draft["id"]):
        if entry["filename"] in retired:
            log("  retiring %s" % entry["filename"])
            z.delete_file(draft["id"], entry["id"])
    ensure_inherited_files(z, draft, previous, retired=retired)
    md = data_metadata(draft.get("metadata") or {}, rel, commit, items, today, "10.5281/zenodo.%s" % args.code_concept,
                       retired=retired, previous_doi=previous.get("doi"))
    wanted = {it["name"]: {"path": it["path"], "md5": it["md5"]} for it in items}
    final = sync_files(z, draft, wanted, keep_others=True)
    ensure_inherited_files(z, draft, previous, retired=retired)
    z.set_metadata(draft["id"], md)
    log("metadata set: %s" % md.get("title"))
    record["data"] = {"concept": args.data_concept, "previous_record": previous.get("id"),
                      "tag": rel["tag"], "commit": commit,
                      "files_added": [{"name": it["name"], "size": it["size"], "sha256": it["sha256"], "md5": it["md5"]}
                                      for it in items if it["name"] not in prev_files],
                      "files_kept": sorted(k for k in prev_files if k not in retired),
                      "files_retired": sorted(retired), "files_in_version": sorted(final)}
    finish(z, z.deposition(draft["id"]), args.publish, record, "data", "version_doi_data")


# ---------------------------------------------------------------- commands

def cmd_plan(z, rel, args):
    path, sha, route, commit = make_snapshot(rel["tag"], rel.get("commit", ""), args.workdir)
    print("release.json: tag %s, commit %s%s" % (rel["tag"], commit, "" if rel.get("commit") else " (resolved from the archive)"))
    print("code concept %s -> new version with one file:" % args.code_concept)
    print("  %s  %s  sha256 %s\n  (%s)" % (os.path.basename(path), human(os.path.getsize(path)), sha, route))
    if args.assets:
        items = verify_assets(data_assets(args.assets, args.cohort_archive))
        if args.retire:
            print("data concept %s -> new version retiring %s, keeping the other previous files, and carrying:" % (
                args.data_concept, ", ".join(args.retire)))
        else:
            print("data concept %s -> new version keeping the previous files and carrying:" % args.data_concept)
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
                if is_open_draft(d):
                    dm = d.get("metadata") or {}
                    print("    title:   %s" % dm.get("title"))
                    print("    version: %s, publication date %s, license %s" % (
                        dm.get("version"), dm.get("publication_date"), dm.get("license")))
                    for rid in dm.get("related_identifiers") or []:
                        print("    related: %s %s" % (rid.get("relation"), rid.get("identifier")))
                    prev = previous_files(r)
                    for f in sorted(z.files(dep_id), key=lambda e: e["filename"]):
                        name, md5 = f["filename"], md5_of_entry(f)
                        mark = "kept" if prev.get(name) == md5 else ("replaces a file of that name" if name in prev else "new")
                        print("    file:    %-36s %12d bytes  md5 %s  (%s)" % (name, int(f.get("filesize") or 0), md5, mark))
                    gone = sorted(k for k in prev if k not in {f["filename"] for f in z.files(dep_id)})
                    if gone:
                        print("    not carried from the previous version: %s" % ", ".join(gone))
    print("release.json: version_doi_code=%r version_doi_data=%r" % (rel.get("version_doi_code"), rel.get("version_doi_data")))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["plan", "status", "code", "data", "all", "describe", "describe-data"])
    ap.add_argument("--record", help="describe, describe-data: a published version of the record to edit in place")
    ap.add_argument("--assets", help="directory holding preds_<family>.tar")
    ap.add_argument("--cohort-archive", help="a cohort archive this version of the data record carries")
    ap.add_argument("--retire", action="append", default=[],
                    help="a file of the previous data version that this version drops (repeatable)")
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
    if args.command == "describe":
        return describe_code(z, rel, args, state, today)
    if args.command == "describe-data":
        return describe_data(z, rel, args, today)
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
