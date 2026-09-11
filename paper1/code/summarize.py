#!/usr/bin/env python3
"""Roll up the double hold-out cells written so far."""
import json, glob, collections, os, statistics as st, sys
# the cell files are written by double_holdout.py on the GPU host; point DH_OUT at them, or run
# this from a checkout where they have been pulled into paper1/results/
_OUT = os.environ.get("DH_OUT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
rows = [json.load(open(f)) for f in glob.glob(os.path.join(_OUT, "*.json"))]
if not rows:
    sys.exit("no cells yet")
g = collections.defaultdict(list)
for r in rows:
    g[(r["model"], r["lambda_mb"])].append(r)
hdr = ("model", "n", "TRAINrxn", "HELDOUTrxn", "gap", "rawexpr", "indic", "rho", "interpat r")
print("%-12s %3s %9s %9s %8s %8s %7s %7s %10s" % hdr)
print("-" * 78)
for k in sorted(g, key=lambda z: (z[0], z[1])):
    v = g[k]
    tr = [x["trainrxn_AUROC_perpatient"] for x in v]
    ho = [x["heldout_AUROC_perpatient"] for x in v]
    sd = st.stdev(ho) if len(ho) > 1 else 0.0
    name = "%s:%.1f" % (k[0], k[1])
    print("%-12s %3d %9.4f %9.4f %8.4f %8.4f %7.4f %7.2f %10.4f" % (
        name, len(v), st.mean(tr), st.mean(ho), st.mean(tr) - st.mean(ho),
        st.mean([x["baseline_rawexpr"] for x in v]),
        st.mean([x["baseline_indicator"] for x in v]),
        st.mean([x["rho"] for x in v]),
        st.mean([x["interpatient_r_median"] for x in v])))
print("-" * 78)
print("held-out sd is across cells; %d cells total on disk" % len(rows))
