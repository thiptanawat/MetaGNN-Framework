#!/usr/bin/env python3
"""Compare the installed NumPy, SciPy and scikit-learn with the versions pinned in requirements.txt.

The committed result files were written in the pinned environment. The scripts run with other
versions and most values do not change, but the results of the language-model study are stored
with ten decimals and those of the first study at full precision, so a value that comes from a
logistic-regression fit can move in its fifth decimal with the library versions. This prints a
note when that can happen. It never stops the run.

The processor does not enter: the order the collection host gave tied expression values is
archived in paper2/cohort_order.npz and every script of the language-model study reads it from
there (see the Environment section of docs/REPRODUCTION_PAPER2.md).

    python3 tools/check_environment.py
"""
import os
import re
from importlib import metadata

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pinned():
    pins = {}
    for line in open(os.path.join(HERE, "requirements.txt")):
        m = re.match(r"^(numpy|scipy|scikit-learn)==([0-9][0-9A-Za-z.]*)", line.strip())
        if m:
            pins[m.group(1)] = m.group(2)
    return pins


def main():
    differ = []
    for name, want in pinned().items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = "not installed"
        if have != want:
            differ.append((name, want, have))
    if not differ:
        print("environment: NumPy, SciPy and scikit-learn match requirements.txt")
        return
    print("note: the installed numerical libraries differ from requirements.txt")
    for name, want, have in differ:
        print("  %-13s pinned %s, installed %s" % (name, want, have))
    print("  The analysis runs. Regenerated result files can differ from the committed ones in late")
    print("  decimals (values from logistic-regression fits in the fifth); see the Environment sections")
    print("  of docs/REPRODUCTION_PAPER1.md and docs/REPRODUCTION_PAPER2.md.")


if __name__ == "__main__":
    main()
