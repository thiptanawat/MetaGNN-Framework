"""Small shared helpers."""
import json

import numpy as np


def clean(o):
    """JSON-safe: NaN/Inf -> None, numpy scalars/arrays -> python, so the output parses strictly."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def dump(obj, path, indent=1):
    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=indent, allow_nan=False, default=str)
    return path
