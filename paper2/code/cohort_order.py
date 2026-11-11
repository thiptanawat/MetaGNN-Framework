"""The order of the cohort at each reaction, as the collection scripts computed it.

Every patient prompt prints the value beside its percentile within the cohort, and that percentile
is the rank from NumPy's default argsort over the 624 patients, with no tie correction. The default
argsort is not stable: the order it gives tied values is left to the sorting implementation, which
differs with the processor's vector instruction set. The collections ran on an Intel x86-64
processor with AVX-512. The order that machine gave is archived in cohort_order.npz for every
reaction at which it differs from a stable sort, so the ranks, the prompts and every statistic
computed from them are the same on any machine.

The archive also holds the top-k selections that the same unstable sort decided where candidates
tie: the 40-reaction panel of the patient-level probe, and the five fold panels of the
fold-inductive reference of msi_stats.py.

    order = argsort_as_collected(X)          # what np.argsort(X, axis=0) returned at collection
    pct   = percentile_as_collected(X)       # 100 * rank / (n - 1), as the prompts print it
    panel = top_k_as_collected('panel', scores, k, candidates)

export_cohort_order.py writes the archive and checks it against the percentiles that the
positive-control records stored at collection time.
"""
import os
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _archive():
    for p in (os.path.join(_ROOT, 'cohort_order.npz'), 'cohort_order.npz'):
        if os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError(os.path.join(_ROOT, 'cohort_order.npz'))


def argsort_as_collected(X):
    """np.argsort(X, axis=0) as the collection host computed it, on any machine."""
    A = _archive()
    if list(A['shape']) != list(X.shape):
        raise ValueError('cohort_order.npz was written for a matrix of shape %s, not %s' % (tuple(A['shape']), X.shape))
    n = X.shape[0]
    order = np.argsort(X, axis=0, kind='stable')
    cols = A['cols']
    archived = A['order'].astype(order.dtype)
    # the archived order has to be an ascending order of this very matrix: a permutation of the
    # patients along which no value decreases
    if not (np.sort(archived, axis=0) == np.arange(n)[:, None]).all():
        raise ValueError('cohort_order.npz: an archived column is not a permutation of the patients')
    if not (np.diff(np.take_along_axis(X[:, cols], archived, axis=0), axis=0) >= 0).all():
        raise ValueError('cohort_order.npz does not order this expression matrix')
    order[:, cols] = archived
    return order


def ranks_as_collected(X):
    """Rank of each patient within the cohort at each reaction (0 .. n-1), ties as collected."""
    order = argsort_as_collected(X)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(X.shape[0], dtype=order.dtype)[:, None], axis=0)
    return ranks


def percentile_as_collected(X):
    """The cohort percentile the prompts print: 100 * rank / (n - 1), no tie correction."""
    return 100.0 * ranks_as_collected(X) / max(X.shape[0] - 1, 1)


def top_k_as_collected(name, scores, k, candidates):
    """The k highest-scoring candidates as selected at collection, where ties left the choice open.

    `scores` are the scores of `candidates`, in the same order. The archived selection is returned
    only if it is a valid top-k selection for these scores: k distinct candidates, none scoring
    below a candidate left out (to within floating-point rounding of the score).
    """
    A = _archive()
    sel = np.asarray(A[name], dtype=int)
    candidates = np.asarray(candidates)
    scores = np.asarray(scores, dtype=float)
    pos = {int(c): i for i, c in enumerate(candidates.tolist())}
    if len(sel) != k or len(set(sel.tolist())) != k or any(int(s) not in pos for s in sel):
        raise ValueError('cohort_order.npz: %s is not a selection of %d of the candidates' % (name, k))
    inside = np.zeros(len(candidates), dtype=bool)
    inside[[pos[int(s)] for s in sel]] = True
    tol = 1e-6 * max(1.0, float(np.abs(scores).max()))
    if scores[inside].min() < scores[~inside].max() - tol:
        raise ValueError('cohort_order.npz: %s is not a top-%d selection for these scores' % (name, k))
    return sel
