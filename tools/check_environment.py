#!/usr/bin/env python3
"""Compare the installed NumPy, SciPy and scikit-learn with the versions pinned in requirements.txt.

The committed result files were written in the pinned environment on Intel x86-64 processors with
AVX-512. The scripts run with other versions and on other processors and most values do not change, but the results of the
language-model study are stored with ten decimals and those of the first study at full precision,
so two things can show in the files: a value that comes from a logistic-regression fit can move
in its fifth decimal with the library versions, and the quantities computed from the cohort
percentile can move by up to 0.002 with the processor, because NumPy's default argsort leaves
the order of tied values to an implementation that differs with the vector instruction set
(AVX-512, AVX2, NEON). This prints a note for either. It never stops the run.

    python3 tools/check_environment.py
"""
import os
import platform
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


def has_avx512():
    """Whether NumPy runs with AVX-512 here: True, False, or None when it does not say."""
    try:
        try:
            from numpy._core._multiarray_umath import __cpu_features__ as feats
        except ImportError:
            from numpy.core._multiarray_umath import __cpu_features__ as feats
        return bool(feats.get("AVX512F"))
    except Exception:
        return None


def main():
    machine = platform.machine().lower()
    x86 = machine in ("x86_64", "amd64")
    if not x86 or has_avx512() is False:
        what = "this x86-64 processor runs NumPy without AVX-512" if x86 else "this machine is %s" % (machine or "of an unknown architecture")
        print("note: %s; the committed results were computed on Intel x86-64 processors with AVX-512" % what)
        print("  The analysis runs. The quantities of the language-model study that use the cohort percentile")
        print("  can differ by up to 0.002, because the order NumPy's argsort gives tied values depends")
        print("  on the processor's vector instruction set; see the Environment section of")
        print("  docs/REPRODUCTION_PAPER2.md.")
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
