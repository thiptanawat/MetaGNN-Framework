#!/usr/bin/env python3
"""
build_families.py -- label-blind reaction families for the 10,600 Recon3D
benchmark reactions.

Input (read-only):
    paper1/data/recon3d_aligned.json.gz, or --input
Only the fields `metabolites` (signed stoichiometry, ids carry a compartment
suffix) and `gene_reaction_rule` are used to build the relations.  `id`,
`subsystem` and `is_exchange` are used for reporting only.  No label or
result file is read, so the families are label-blind by construction.

Three equivalence relations are defined (functions exact_key, stripped_key,
gene_set_key below).  Each relation is an equality of a canonical key, so it
is automatically reflexive, symmetric and transitive; classes are the sets of
reactions sharing a key.  Reactions whose key is `None` are "ineligible" and
form singleton classes (they are never joined by that relation).

  (i)   EXACT stoichiometric equivalence.
        key = the sorted tuple of (metabolite_id_with_compartment, coeff),
        taken as min(v, -v) so that a reaction and its reversed writing
        (whole vector negated) get the same key.  Bounds are ignored.
        Scalar multiples other than -1 are NOT merged.  An empty metabolite
        dict (none occur in this dataset) would be ineligible.

  (ii)  COMPARTMENT-STRIPPED equivalence with sides preserved.
        The compartment suffix (text after the last underscore, e.g. '_c')
        is removed from every metabolite id.  Substrates (coeff < 0) and
        products (coeff > 0) are kept as two separate multisets
        {stripped_id -> total |coeff| on that side}, so a transport reaction
        'a_c -> a_m' has substrates {a:1}, products {a:1} -- a non-empty
        signature.  key = min((S, P), (P, S)) so that mirror images
        (swapped sides) are equivalent.  Empty metabolite dicts would be
        ineligible.  Ids without any underscore (none occur) would be kept
        whole.

  (iii) NON-EMPTY gene-rule equivalence.
        key = sorted tuple of the set of tokens that remain after replacing
        '(' and ')' by spaces in gene_reaction_rule, splitting on the words
        'and' / 'or', and trimming whitespace.  Recon3D writes transcript-level
        ids such as '314_AT1'; by default a trailing '_AT<digits>' is stripped,
        so the key is a set of genes and two rules naming the same genes through
        different transcripts are equivalent.  Pass --transcript_level to keep
        the ids verbatim, which makes the relation finer.  An empty or missing
        (None) rule gives key None -> ineligible, never joined.

Schemes:
    union : (i) + (ii) + (iii)
    S2    : (i) + (ii)
    S3    : (i) + (ii) + (iii) where (iii) is dropped for every gene set
            whose own class (relation (iii) alone) has more than
            GPR_CAP = 50 reactions.

Components are computed with an explicit union-find and cross-checked with
scipy.sparse.csgraph.connected_components.

Outputs (in the directory of this script):
    families_union.json, families_S2.json, families_S3.json,
    overlap.json, audit.json, audit_summary.txt
"""
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

import argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # the repository root
_ap = argparse.ArgumentParser()
_ap.add_argument("--input", default=os.path.join(_HERE, "..", "data", "recon3d_aligned.json.gz"))
_ap.add_argument("--out", default=os.path.join(_HERE, "..", "data", "families"))
_ap.add_argument("--transcript_level", action="store_true",
                 help="keep Recon3D transcript suffixes (314_AT1, 314_AT2) as distinct tokens in relation (iii); "
                      "the default collapses them to the gene, which is the level at which expression is mapped")
_A = _ap.parse_args()
INPUT = _A.input
OUT_DIR = _A.out
os.makedirs(OUT_DIR, exist_ok=True)
GPR_CAP = 50           # S3: drop relation (iii) for gene sets joining > 50 reactions
COEF_DECIMALS = 9      # coefficients are rounded to this many decimals before comparison
SIZE_BINS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 50), (51, 200), (201, 10 ** 9)]
N_LARGEST = 5
TOP_GENE_SETS = 10


# ----------------------------------------------------------------------------
# Relation keys
# ----------------------------------------------------------------------------
def _r(x):
    """Round a coefficient for stable comparison (float noise guard)."""
    v = round(float(x), COEF_DECIMALS)
    return 0.0 if v == 0 else v


def exact_key(mets):
    """Relation (i): identical signed vector incl. compartments, up to global negation."""
    if not mets:
        return None
    v = tuple(sorted((m, _r(c)) for m, c in mets.items()))
    neg = tuple(sorted((m, _r(-c)) for m, c in mets.items()))
    return min(v, neg)


def strip_compartment(met_id):
    """Drop the compartment suffix = text after the LAST underscore ('glc_D_c' -> 'glc_D')."""
    return met_id.rsplit("_", 1)[0]


def stripped_key(mets):
    """Relation (ii): compartment-stripped, sides kept as two multisets, mirror allowed."""
    if not mets:
        return None
    sub, prod = defaultdict(float), defaultdict(float)
    for m, c in mets.items():
        c = float(c)
        if c < 0:
            sub[strip_compartment(m)] += -c
        elif c > 0:
            prod[strip_compartment(m)] += c
    S = tuple(sorted((k, _r(v)) for k, v in sub.items()))
    P = tuple(sorted((k, _r(v)) for k, v in prod.items()))
    return min((S, P), (P, S))


_SPLIT_AND_OR = re.compile(r"\b(?:and|or)\b")


def gene_set_key(rule):
    """Relation (iii): sorted tuple of distinct gene tokens; None for empty/missing rule."""
    if rule is None:
        return None
    s = rule.replace("(", " ").replace(")", " ")
    toks = {t.strip() for t in _SPLIT_AND_OR.split(s)}
    toks.discard("")
    if not _A.transcript_level:
        toks = {re.sub(r"_AT\d+$", "", t) for t in toks}
    if not toks:
        return None
    return tuple(sorted(toks))


# ----------------------------------------------------------------------------
# Union-find
# ----------------------------------------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:          # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def labels(self):
        """Component ids numbered 0.. in order of the smallest member index."""
        n = len(self.parent)
        roots = [self.find(i) for i in range(n)]
        first = {}
        lab = [0] * n
        for i, r in enumerate(roots):
            if r not in first:
                first[r] = len(first)
            lab[i] = first[r]
        return lab


def classes_from_keys(keys):
    """Map key -> sorted list of reaction indices (ineligible None keys excluded)."""
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        if k is not None:
            groups[k].append(i)
    return groups


def components(n, class_maps):
    """Union-find over the union of the given relations; cross-checked with scipy."""
    uf = UnionFind(n)
    rows, cols = [], []
    for groups in class_maps:
        for members in groups.values():
            for j in members[1:]:
                uf.union(members[0], j)
                rows.append(members[0])
                cols.append(j)
    lab = uf.labels()
    # cross-check with scipy.sparse.csgraph
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    ncc, sci = connected_components(A, directed=False)
    assert ncc == len(set(lab)), "union-find / scipy disagree on component count"
    # same partition?
    pair_map = {}
    for a, b in zip(lab, sci):
        assert pair_map.setdefault(a, b) == b, "union-find / scipy partitions differ"
    return lab


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------
def size_hist(sizes):
    out = {}
    for lo, hi in SIZE_BINS:
        name = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 8 else f">{lo - 1}")
        out[name] = int(sum(1 for s in sizes if lo <= s <= hi))
    return out


def relation_stats(name, groups, rx, n):
    nontrivial = {k: v for k, v in groups.items() if len(v) > 1}
    n_in_nontrivial = sum(len(v) for v in nontrivial.values())
    n_eligible = sum(len(v) for v in groups.values())
    n_classes_total = len(groups) + (n - n_eligible)   # ineligible -> singleton classes
    if nontrivial:
        # largest class; ties broken by the smallest member index (deterministic)
        largest_key, largest = max(nontrivial.items(), key=lambda kv: (len(kv[1]), -kv[1][0]))
    else:
        largest_key, largest = None, []
    sizes = sorted((len(v) for v in nontrivial.values()), reverse=True)
    return {
        "relation": name,
        "n_reactions_eligible": int(n_eligible),
        "n_reactions_ineligible": int(n - n_eligible),
        "n_reactions_in_nontrivial_class": int(n_in_nontrivial),
        "n_classes_total_including_singletons": int(n_classes_total),
        "n_classes_among_eligible": int(len(groups)),
        "n_nontrivial_classes": int(len(nontrivial)),
        "n_pairs_joined": int(sum(len(v) * (len(v) - 1) // 2 for v in nontrivial.values())),
        "nontrivial_class_size_hist": size_hist(sizes),
        "top_class_sizes": sizes[:10],
        "largest_class": None if not nontrivial else {
            "size": int(len(largest)),
            "key": _key_to_str(largest_key),
            "member_ids_first20": [rx[i]["id"] for i in largest[:20]],
            "member_idx_first20": largest[:20],
            "subsystems": dict(Counter(str(rx[i]["subsystem"]) for i in largest).most_common()),
        },
    }


def _key_to_str(k):
    if k is None:
        return None
    return json.dumps(k, default=list)


def component_stats(name, lab, rx, class_maps_named, relations_used):
    """Component-level statistics for one scheme."""
    n = len(lab)
    members = defaultdict(list)
    for i, c in enumerate(lab):
        members[c].append(i)
    sizes = {c: len(v) for c, v in members.items()}
    size_list = sorted(sizes.values(), reverse=True)
    largest = sorted(members.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:N_LARGEST]

    # global pair (edge) accounting over the whole graph, per relation
    pair_sets = {}
    for rname, groups in class_maps_named.items():
        s = set()
        for mem in groups.values():
            if len(mem) > 1:
                s.update(combinations(mem, 2))
        pair_sets[rname] = s
    all_pairs = set().union(*pair_sets.values()) if pair_sets else set()
    global_edges = {"n_edges_union": len(all_pairs)}
    for rname, s in pair_sets.items():
        others = set().union(*(v for k, v in pair_sets.items() if k != rname)) if len(pair_sets) > 1 else set()
        global_edges[f"n_edges_{rname}"] = len(s)
        global_edges[f"n_edges_{rname}_only"] = len(s - others)

    giant = []
    for c, mem in largest:
        memset = set(mem)
        comp_pairs = {r: {p for p in s if p[0] in memset} for r, s in pair_sets.items()}
        # (both endpoints are in the same component whenever one is, since
        #  every relation edge lies inside a component)
        union_pairs = set().union(*comp_pairs.values()) if comp_pairs else set()
        edge_info = {"n_edges_union": len(union_pairs)}
        for r, s in comp_pairs.items():
            others = set().union(*(v for k, v in comp_pairs.items() if k != r)) if len(comp_pairs) > 1 else set()
            edge_info[f"n_edges_{r}"] = len(s)
            edge_info[f"n_edges_{r}_only"] = len(s - others)
        # fragmentation when a relation is removed: number of sub-components
        frag = {}
        for r in comp_pairs:
            uf = UnionFind(len(mem))
            pos = {i: j for j, i in enumerate(mem)}
            for r2, s in comp_pairs.items():
                if r2 == r:
                    continue
                for a, b in s:
                    uf.union(pos[a], pos[b])
            frag[f"n_subcomponents_without_{r}"] = len(set(uf.labels()))
        # dominant relation = the one that appears in the most edges of the component
        dominant = max(comp_pairs, key=lambda r: len(comp_pairs[r])) if comp_pairs else None
        # gene sets present in this component
        gsets = Counter(gene_set_key(rx[i]["gene_reaction_rule"]) for i in mem)
        gsets.pop(None, None)
        giant.append({
            "component_id": int(c),
            "size": int(len(mem)),
            "member_ids_first20": [rx[i]["id"] for i in mem[:20]],
            "subsystems": dict(Counter(str(rx[i]["subsystem"]) for i in mem).most_common()),
            "n_exchange_flagged": int(sum(1 for i in mem if rx[i]["is_exchange"])),
            "n_with_nonempty_gpr": int(sum(1 for i in mem if gene_set_key(rx[i]["gene_reaction_rule"]) is not None)),
            "n_distinct_gene_sets": int(len(gsets)),
            "top_gene_sets": [{"gene_set": list(k), "n_reactions_in_component": v} for k, v in gsets.most_common(5)],
            "n_distinct_stripped_signatures": int(len({stripped_key(rx[i]["metabolites"]) for i in mem})),
            "n_distinct_exact_signatures": int(len({exact_key(rx[i]["metabolites"]) for i in mem})),
            "edges": edge_info,
            "dominant_relation_by_edges": dominant,
            "fragmentation": frag,
        })

    return {
        "scheme": name,
        "relations_used": relations_used,
        "n_reactions": int(n),
        "n_components": int(len(members)),
        "n_singletons": int(sum(1 for s in size_list if s == 1)),
        "n_reactions_in_nonsingleton_components": int(sum(s for s in size_list if s > 1)),
        "size_distribution": size_hist(size_list),
        "largest_component_sizes": size_list[:20],
        "global_edges": global_edges,
        "largest_components": giant,
    }


# ----------------------------------------------------------------------------
def main():
    with gzip.open(INPUT, "rt") as f:
        data = json.load(f)
    rx = data["reactions"]
    n = len(rx)
    assert n == 10600, n
    assert all(r["idx"] == i for i, r in enumerate(rx)), "reactions not in canonical order"
    ids = [r["id"] for r in rx]
    assert len(set(ids)) == n, "reaction ids are not unique"

    # ---- data audit (edge cases) ----
    all_mets = {m for r in rx for m in r["metabolites"]}
    suffixes = Counter(m.rsplit("_", 1)[1] if "_" in m else "<no underscore>" for m in all_mets)
    data_audit = {
        "n_reactions": n,
        "n_distinct_metabolite_ids": len(all_mets),
        "compartment_suffix_counts": dict(sorted(suffixes.items())),   # sorted, so the file is byte-stable across runs
        "n_metabolite_ids_without_underscore": int(sum(1 for m in all_mets if "_" not in m)),
        "n_metabolite_ids_with_non_single_letter_suffix": int(
            sum(1 for m in all_mets if "_" in m and not (len(m.rsplit("_", 1)[1]) == 1 and m.rsplit("_", 1)[1].isalpha()))),
        "n_reactions_with_empty_metabolite_dict": int(sum(1 for r in rx if not r["metabolites"])),
        "n_zero_coefficients": int(sum(1 for r in rx for v in r["metabolites"].values() if float(v) == 0)),
        "n_reactions_with_noninteger_coefficients": int(
            sum(1 for r in rx if any(abs(float(v) - round(float(v))) > 1e-9 for v in r["metabolites"].values()))),
        "n_is_exchange": int(sum(1 for r in rx if r["is_exchange"])),
        "n_single_metabolite_reactions": int(sum(1 for r in rx if len(r["metabolites"]) == 1)),
        "n_single_metabolite_not_flagged_exchange": int(sum(1 for r in rx if len(r["metabolites"]) == 1 and not r["is_exchange"])),
        "single_metabolite_id_prefixes": dict(Counter(
            r["id"].split("_")[0] if r["id"].startswith(("EX_", "DM_", "sink_")) else "<other>"
            for r in rx if len(r["metabolites"]) == 1)),
        "n_gene_reaction_rule_None": int(sum(1 for r in rx if r["gene_reaction_rule"] is None)),
        "n_gene_reaction_rule_empty_string": int(sum(1 for r in rx if r["gene_reaction_rule"] == "")),
        "n_gene_reaction_rule_nonempty": int(sum(1 for r in rx if gene_set_key(r["gene_reaction_rule"]) is not None)),
        "n_unaligned_reactions_(aligned=False)": int(sum(1 for r in rx if not r.get("aligned", True))),
        "n_subsystem_None": int(sum(1 for r in rx if r["subsystem"] is None)),
        "gene_token_pattern": {
            "n_distinct_tokens": None, "n_tokens_matching_<entrez>_AT<n>": None, "tokens_not_matching": None},
    }
    toks = Counter(t for r in rx for t in (gene_set_key(r["gene_reaction_rule"]) or ()))
    pat = re.compile(r"\d+_AT\d+")
    data_audit["gene_token_pattern"] = {
        "n_distinct_tokens": len(toks),
        "n_tokens_matching_<entrez>_AT<n>": int(sum(1 for t in toks if pat.fullmatch(t))),
        "tokens_not_matching": {t: [ids[i] for i, r in enumerate(rx) if t in (gene_set_key(r["gene_reaction_rule"]) or ())]
                                for t in toks if not pat.fullmatch(t)},
    }

    # ---- keys and classes ----
    k_exact = [exact_key(r["metabolites"]) for r in rx]
    k_strip = [stripped_key(r["metabolites"]) for r in rx]
    k_gpr = [gene_set_key(r["gene_reaction_rule"]) for r in rx]
    G_exact = classes_from_keys(k_exact)
    G_strip = classes_from_keys(k_strip)
    G_gpr = classes_from_keys(k_gpr)

    rel_stats = {
        "exact": relation_stats("(i) exact stoichiometric (incl. compartments, up to negation)", G_exact, rx, n),
        "stripped": relation_stats("(ii) compartment-stripped, sides preserved, mirror allowed", G_strip, rx, n),
        "gpr": relation_stats("(iii) identical non-empty gene set from gene_reaction_rule", G_gpr, rx, n),
    }
    # sanity: relation (i) refines relation (ii): same exact key => same stripped key
    for mem in G_exact.values():
        assert len({k_strip[i] for i in mem}) == 1, "exact class spans several stripped classes?!"
    # how much of (i) is a reversed writing?
    n_rev = 0
    for mem in G_exact.values():
        if len(mem) > 1:
            raw = {tuple(sorted((m, _r(c)) for m, c in rx[i]["metabolites"].items())) for i in mem}
            if len(raw) > 1:
                n_rev += 1
    rel_stats["exact"]["n_nontrivial_classes_containing_a_reversed_writing"] = int(n_rev)

    # per-reaction flags
    has_exact = [k is not None and len(G_exact[k]) > 1 for k in k_exact]
    has_strip = [k is not None and len(G_strip[k]) > 1 for k in k_strip]
    has_gpr = [k is not None and len(G_gpr[k]) > 1 for k in k_gpr]

    # ---- schemes ----
    big_gene_sets = {k: len(v) for k, v in G_gpr.items() if len(v) > GPR_CAP}
    G_gpr_capped = {k: v for k, v in G_gpr.items() if len(v) <= GPR_CAP}
    has_gpr_capped = [k is not None and k in G_gpr_capped and len(G_gpr_capped[k]) > 1 for k in k_gpr]

    lab_union = components(n, [G_exact, G_strip, G_gpr])
    lab_s2 = components(n, [G_exact, G_strip])
    lab_s3 = components(n, [G_exact, G_strip, G_gpr_capped])

    st_union = component_stats("union", lab_union, rx,
                               {"exact": G_exact, "stripped": G_strip, "gpr": G_gpr}, ["(i)", "(ii)", "(iii)"])
    st_s2 = component_stats("S2", lab_s2, rx, {"exact": G_exact, "stripped": G_strip}, ["(i)", "(ii)"])
    st_s3 = component_stats("S3", lab_s3, rx,
                            {"exact": G_exact, "stripped": G_strip, "gpr_capped": G_gpr_capped},
                            ["(i)", "(ii)", f"(iii) restricted to gene sets joining <= {GPR_CAP} reactions"])
    st_s3["gene_sets_dropped_by_cap"] = [
        {"gene_set": list(k), "n_reactions": v} for k, v in sorted(big_gene_sets.items(), key=lambda kv: -kv[1])]
    st_s3["n_reactions_losing_gpr_edges_by_cap"] = int(sum(big_gene_sets.values()))

    # ---- transporter GPR diagnosis ----
    top_gene_sets = []
    for k, mem in sorted(G_gpr.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:TOP_GENE_SETS]:
        top_gene_sets.append({
            "gene_set": list(k),
            "n_reactions": len(mem),
            "n_distinct_stripped_signatures": len({k_strip[i] for i in mem}),
            "n_distinct_exact_signatures": len({k_exact[i] for i in mem}),
            "subsystems": dict(Counter(str(rx[i]["subsystem"]) for i in mem).most_common()),
            "n_transport_like_(a stripped metabolite on both sides)": int(sum(
                1 for i in mem if set(strip_compartment(m) for m, c in rx[i]["metabolites"].items() if c < 0)
                & set(strip_compartment(m) for m, c in rx[i]["metabolites"].items() if c > 0))),
            "member_ids_first20": [ids[i] for i in mem[:20]],
            "n_union_components_spanned_without_this_gene_set":
                None,  # filled below
        })
    # how many S2 components does each top gene set stitch together?
    for entry in top_gene_sets:
        k = tuple(entry["gene_set"])
        entry["n_union_components_spanned_without_this_gene_set"] = len({lab_s2[i] for i in G_gpr[k]})
        entry["n_S2_components_joined"] = entry.pop("n_union_components_spanned_without_this_gene_set")

    # ---- sensitivity checks on documented ambiguities (not used for the saved families) ----
    def strip_at(t):
        return re.sub(r"_AT\d+$", "", t)
    k_gpr_gene = [tuple(sorted({strip_at(t) for t in k})) if k is not None else None for k in k_gpr]
    G_gpr_gene = classes_from_keys(k_gpr_gene)
    k_table = [gene_set_key(r.get("gene_rule_table")) for r in rx]
    G_table = classes_from_keys(k_table)
    lab_gene = components(n, [G_exact, G_strip, G_gpr_gene])
    lab_table = components(n, [G_exact, G_strip, G_table])

    def scalar_key(mets):
        if not mets:
            return None
        m0 = min(mets)
        s = float(mets[m0])
        v = tuple(sorted((m, _r(float(c) / abs(s))) for m, c in mets.items()))
        neg = tuple(sorted((m, _r(-float(c) / abs(s))) for m, c in mets.items()))
        return min(v, neg)
    G_scalar = classes_from_keys([scalar_key(r["metabolites"]) for r in rx])
    scalar_pairs = []
    for mem in G_scalar.values():
        if len(mem) > 1:
            scalar_pairs.append({
                "ids": [ids[i] for i in mem],
                "metabolites": [rx[i]["metabolites"] for i in mem],
                "gene_sets": [list(k_gpr[i]) if k_gpr[i] else None for i in mem],
                "same_union_component": len({lab_union[i] for i in mem}) == 1,
                "same_S2_component": len({lab_s2[i] for i in mem}) == 1,
                "same_S3_component": len({lab_s3[i] for i in mem}) == 1,
            })
    # reactions with the same metabolite-id SET but a different coefficient vector
    same_set = defaultdict(list)
    for i, r in enumerate(rx):
        same_set[tuple(sorted(r["metabolites"]))].append(i)
    same_set_groups = [v for v in same_set.values() if len(v) > 1]

    cnt_gene, cnt_union = Counter(lab_gene), Counter(lab_union)
    sensitivity = {
        "gpr_gene_level_(strip _AT transcript suffix from gene_reaction_rule tokens)": {
            "n_reactions_in_nontrivial_class": int(sum(len(v) for v in G_gpr_gene.values() if len(v) > 1)),
            "n_nontrivial_classes": int(sum(1 for v in G_gpr_gene.values() if len(v) > 1)),
            "largest_class_size": int(max(len(v) for v in G_gpr_gene.values())),
            "union_components": int(len(set(lab_gene))),
            "union_size_distribution": size_hist(list(Counter(lab_gene).values())),
            "union_largest_sizes": sorted(Counter(lab_gene).values(), reverse=True)[:5],
            "n_reactions_whose_union_component_size_changes_vs_primary": int(sum(
                1 for i in range(n) if cnt_gene[lab_gene[i]] != cnt_union[lab_union[i]])),
        },
        "gpr_from_gene_rule_table_field_(gene-level ids, present for all 10600 incl. 289 unaligned)": {
            "n_nonempty": int(sum(1 for k in k_table if k is not None)),
            "n_reactions_in_nontrivial_class": int(sum(len(v) for v in G_table.values() if len(v) > 1)),
            "n_nontrivial_classes": int(sum(1 for v in G_table.values() if len(v) > 1)),
            "largest_class_size": int(max(len(v) for v in G_table.values())),
            "union_components": int(len(set(lab_table))),
            "union_size_distribution": size_hist(list(Counter(lab_table).values())),
            "union_largest_sizes": sorted(Counter(lab_table).values(), reverse=True)[:5],
        },
        "exact_allowing_any_scalar_multiple_(not used)": {
            "n_reactions_in_nontrivial_class": int(sum(len(v) for v in G_scalar.values() if len(v) > 1)),
            "n_nontrivial_classes": int(sum(1 for v in G_scalar.values() if len(v) > 1)),
            "extra_reactions_vs_relation_(i)": int(sum(len(v) for v in G_scalar.values() if len(v) > 1)
                                                   - rel_stats["exact"]["n_reactions_in_nontrivial_class"]),
            "scalar_multiple_groups": scalar_pairs,
        },
        "same_metabolite_id_set_but_different_coefficient_vector_(not a relation)": {
            "n_groups": len(same_set_groups),
            "n_reactions": int(sum(len(v) for v in same_set_groups)),
            "n_groups_fully_inside_one_union_component": int(sum(1 for v in same_set_groups if len({lab_union[i] for i in v}) == 1)),
            "n_groups_fully_inside_one_S2_component": int(sum(1 for v in same_set_groups if len({lab_s2[i] for i in v}) == 1)),
            "example_ids": [[ids[i] for i in v] for v in same_set_groups[:5]],
        },
    }

    # ---- exchange / boundary reactions under relation (ii) ----
    boundary_idx = [i for i, r in enumerate(rx) if len(r["metabolites"]) == 1]
    b_classes = Counter(k_strip[i] for i in boundary_idx)
    mixed = 0
    for i in boundary_idx:
        if any(len(rx[j]["metabolites"]) > 1 for j in G_strip[k_strip[i]]):
            mixed += 1
    boundary_audit = {
        "n_single_metabolite_reactions": len(boundary_idx),
        "n_single_metabolite_reactions_in_nontrivial_stripped_class": int(sum(1 for i in boundary_idx if has_strip[i])),
        "n_stripped_classes_containing_>=2_single_metabolite_reactions": int(sum(1 for v in b_classes.values() if v > 1)),
        "n_single_metabolite_reactions_sharing_a_stripped_class_with_a_multi_metabolite_reaction": int(mixed),
        "explanation": ("EX_x[e] (x_e ->), DM_x_c_ (x_c ->) and sink_x_c_ (x_c <=>) all reduce to "
                        "(substrates {x}, products {}) after compartment stripping, so boundary reactions "
                        "of the same species are joined by relation (ii)."),
    }

    # ---- overlap report ----
    def overlap(lab):
        cnt = Counter(lab)
        return [int(cnt[c] - 1) for c in lab]

    # ---- save ----
    definitions = {
        "(i)": "identical signed stoichiometric vector over compartment-suffixed metabolite ids, up to global negation "
               "(reversed writing); coefficients rounded to %d decimals; bounds ignored; scalar multiples other than -1 not merged; "
               "empty metabolite dict -> never joined" % COEF_DECIMALS,
        "(ii)": "compartment suffix (after last underscore) stripped from every metabolite id; substrates (coef<0) and products (coef>0) "
                "kept as separate multisets {stripped_id: sum |coef|}; key = min((S,P),(P,S)) so mirror images are equivalent; "
                "empty metabolite dict -> never joined",
        "(iii)": "set of tokens of gene_reaction_rule after replacing parentheses by spaces, splitting on the words and/or, trimming; "
                 + ("tokens used verbatim, so transcript-level ids like 314_AT1 stay distinct"
                    if _A.transcript_level else
                    "a trailing transcript suffix (_AT followed by digits) is stripped, so 314_AT1 and 314_AT2 are the same token")
                 + "; empty or None rule -> never joined",
        "S2": "(i)+(ii)",
        "S3": "(i)+(ii)+(iii), with (iii) dropped for gene sets whose class alone has more than %d reactions" % GPR_CAP,
        "component_id": "components numbered from 0 in order of their smallest member reaction index",
    }

    def save_families(path, name, lab, extra_flags=None):
        cnt = Counter(lab)
        obj = {
            "scheme": name,
            "definitions": definitions,
            "n_reactions": n,
            "n_components": len(cnt),
            "reaction_ids": ids,
            "component_id": [int(c) for c in lab],
            "component_size": [int(cnt[c]) for c in lab],
            "has_exact_twin": has_exact,
            "has_stripped_twin": has_strip,
            "has_gpr_twin": has_gpr,
        }
        if extra_flags:
            obj.update(extra_flags)
        with open(path, "w") as f:
            json.dump(obj, f)

    save_families(os.path.join(OUT_DIR, "families_union.json"), "union", lab_union)
    save_families(os.path.join(OUT_DIR, "families_S2.json"), "S2", lab_s2,
                  {"note": "has_gpr_twin is reported for information only; relation (iii) is NOT used in S2"})
    save_families(os.path.join(OUT_DIR, "families_S3.json"), "S3", lab_s3,
                  {"has_gpr_twin_after_cap": has_gpr_capped,
                   "note": "has_gpr_twin is the uncapped flag; has_gpr_twin_after_cap is the flag actually used in S3"})

    with open(os.path.join(OUT_DIR, "overlap.json"), "w") as f:
        json.dump({
            "description": "for each reaction (canonical index order), the number of OTHER reactions in the same component",
            "reaction_ids": ids,
            "union": overlap(lab_union),
            "S2": overlap(lab_s2),
            "S3": overlap(lab_s3),
        }, f)

    audit = {
        "input": os.path.relpath(INPUT, _ROOT) if os.path.abspath(INPUT).startswith(_ROOT) else INPUT,
        "definitions": definitions,
        "parameters": {"GPR_CAP": GPR_CAP, "COEF_DECIMALS": COEF_DECIMALS},
        "data_audit": data_audit,
        "boundary_reactions_under_relation_ii": boundary_audit,
        "relations": rel_stats,
        "per_reaction_flag_counts": {
            "has_exact_twin": int(sum(has_exact)),
            "has_stripped_twin": int(sum(has_strip)),
            "has_gpr_twin": int(sum(has_gpr)),
            "has_gpr_twin_after_cap": int(sum(has_gpr_capped)),
            "has_any_twin": int(sum(a or b or c for a, b, c in zip(has_exact, has_strip, has_gpr))),
            "has_stripped_but_not_exact": int(sum(b and not a for a, b in zip(has_exact, has_strip))),
            "has_gpr_only": int(sum(c and not a and not b for a, b, c in zip(has_exact, has_strip, has_gpr))),
        },
        "schemes": {"union": st_union, "S2": st_s2, "S3": st_s3},
        "top_gene_sets": top_gene_sets,
        "sensitivity_checks_not_used_for_saved_families": sensitivity,
    }
    with open(os.path.join(OUT_DIR, "audit.json"), "w") as f:
        json.dump(audit, f, indent=1, default=list)

    # ---- human-readable summary ----
    lines = []
    P = lines.append
    P("DATA AUDIT")
    for k, v in data_audit.items():
        P(f"  {k}: {v}")
    P("")
    for key in ("exact", "stripped", "gpr"):
        s = rel_stats[key]
        P(f"RELATION {s['relation']}")
        for k, v in s.items():
            if k not in ("relation", "largest_class"):
                P(f"  {k}: {v}")
        lc = s["largest_class"]
        if lc is None:
            P("  largest class: none (every class is a singleton)")
        else:
            P(f"  largest class: size={lc['size']} key={lc['key'][:200]}")
            P(f"    members(first20)={lc['member_ids_first20']}")
            P(f"    subsystems={lc['subsystems']}")
        P("")
    P("PER-REACTION FLAGS: " + str(audit["per_reaction_flag_counts"]))
    P("BOUNDARY (single-metabolite) REACTIONS UNDER (ii): " + str(boundary_audit))
    P("")
    for name, st in (("union", st_union), ("S2", st_s2), ("S3", st_s3)):
        P(f"SCHEME {name}: relations {st['relations_used']}")
        P(f"  n_components={st['n_components']} singletons={st['n_singletons']} "
          f"reactions_in_nonsingleton={st['n_reactions_in_nonsingleton_components']}")
        P(f"  size_distribution={st['size_distribution']}")
        P(f"  largest sizes={st['largest_component_sizes']}")
        P(f"  global edges={st['global_edges']}")
        if name == "S3":
            P(f"  gene sets dropped by cap (> {GPR_CAP}): {len(st['gene_sets_dropped_by_cap'])}, "
              f"reactions affected={st['n_reactions_losing_gpr_edges_by_cap']}")
            for g in st["gene_sets_dropped_by_cap"]:
                P(f"    {g['n_reactions']:4d}  {g['gene_set']}")
        for g in st["largest_components"]:
            P(f"  component {g['component_id']} size={g['size']} exchange_flagged={g['n_exchange_flagged']} "
              f"with_gpr={g['n_with_nonempty_gpr']} distinct_gene_sets={g['n_distinct_gene_sets']} "
              f"distinct_stripped={g['n_distinct_stripped_signatures']} distinct_exact={g['n_distinct_exact_signatures']}")
            P(f"    subsystems={g['subsystems']}")
            P(f"    edges={g['edges']}")
            P(f"    dominant={g['dominant_relation_by_edges']} fragmentation={g['fragmentation']}")
            P(f"    top gene sets={g['top_gene_sets']}")
            P(f"    members(first20)={g['member_ids_first20']}")
        P("")
    P("TOP GENE SETS (relation iii)")
    for g in top_gene_sets:
        P(f"  n={g['n_reactions']} stripped_sigs={g['n_distinct_stripped_signatures']} exact_sigs={g['n_distinct_exact_signatures']} "
          f"S2_components_joined={g['n_S2_components_joined']} transport_like={g['n_transport_like_(a stripped metabolite on both sides)']} "
          f"genes={g['gene_set']}")
        P(f"    subsystems={g['subsystems']}")
        P(f"    members(first20)={g['member_ids_first20']}")
    P("")
    P("SENSITIVITY (not used for saved families)")
    for k, v in sensitivity.items():
        P(f"  {k}: {v}")
    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "audit_summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
