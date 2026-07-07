"""
Per-experiment validators.

Each module exposes a `validate(verbose: bool = False) -> ValidationReport`
function that loads the shipped result/checkpoint, recomputes the
headline metric, and compares it against the manuscript-anchored value.

No retraining. Runs in seconds on CPU. Reviewer workflow:

    git clone <repo>
    cd Journals
    bash experiments/scripts/wire_shared_data.sh
    python experiments/validate.py
"""
