#!/usr/bin/env python3
"""Fail the build if the manuscript uses a macro the results pipeline does not define.

This is the guard that makes the reproducibility claim enforceable: a number can only
appear in the text through a macro, and a macro only exists if a run produced it.
"""
import re, os, sys, glob

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(HERE, "ms" if os.path.isdir(os.path.join(HERE, "ms")) else "manuscript")
# a checkout that carries the analysis without the manuscript sources has nothing to check
# against; say so rather than pass silently on an empty set of files
if not glob.glob(os.path.join(MS, "body_*.tex")):
    print("no manuscript sources in this checkout: the macro file is written, but there is\nno text to check it against. This is the expected state of a code-and-results release.")
    raise SystemExit(0)
defs = set(re.findall(r"\\newcommand\{\\(\w+)\}", open(os.path.join(MS, "numbers.tex")).read()))

def strip_undefined_blocks(txt):
    """Return the text LaTeX would typeset given the macros the pipeline defines: every
    \\ifdefined\\X ... [\\else ...] \\fi block keeps its first branch when X is defined and its
    \\else branch otherwise, with nested blocks resolved the same way, so that text guarded on a
    macro that a later run will supply is not counted against the build before that run exists."""
    tok = re.compile(r"\\ifdefined\\(\w+)|\\ifnum\b|\\else\b|\\fi\b")
    out = []; pos = 0; stack = []          # stack entries: [keep_now, keep_after_else]
    for m in tok.finditer(txt):
        if all(st[0] for st in stack): out.append(txt[pos:m.start()])
        pos = m.end()
        if m.group(1) is not None:
            on = m.group(1) in defs; stack.append([on, not on])
        elif m.group(0) == "\\ifnum":
            stack.append([True, True])     # a numeric test (singular/plural): both branches use the same macros
        elif m.group(0) == "\\else":
            if stack: stack[-1][0] = stack[-1][1]
        else:
            if stack: stack.pop()
    if all(st[0] for st in stack): out.append(txt[pos:])
    return "".join(out)

def guarded_bodies(txt):
    """Every \\ifdefined body, with the name it is guarded on and its line number.

    These are the sentences a future run switches on, so they are the ones no reader will
    proof-read before they appear. They have to survive whatever value that run produces."""
    out = []
    for m in re.finditer(r"\\ifdefined\\(\w+)", txt):
        j = txt.index("\\fi", m.end())
        body = txt[m.end():j].partition("\\else")[0]
        out.append((m.group(1), txt[:m.start()].count("\n") + 1, body))
    return out

# A guarded sentence may not read a direction off an absolute value: \fooAbs strips the sign, so
# "the cost is \fooAbs{}" stays in the document unchanged if the sign flips. Use the signed macro,
# or the *Word / *Dir macro the pipeline derives from the sign.
DIRECTIONAL = re.compile(r"\b(cost|costs|benefit|benefits|gain|gains|higher|lower|above|below|"
                         r"favors|favor|penalizes|worse|better|negative|positive|drops?|rises?)\b")
used = set()
sign_violations = []
for f in sorted(glob.glob(os.path.join(MS, "*.tex"))):
    if os.path.basename(f) == "numbers.tex":
        continue
    raw = open(f).read()
    for name, line, body in guarded_bodies(raw):
        for sent in re.split(r"(?<=[.;])\s", body):
            abs_macros = re.findall(r"\\(\w+Abs)\b", sent)
            hit = DIRECTIONAL.search(sent)
            if abs_macros and hit and not re.search(r"\\\w+(Word|Dir)\b", sent):
                sign_violations.append(
                    f"{os.path.basename(f)}:{line} guarded on \\{name}: "
                    f"\\{abs_macros[0]} asserts \"{hit.group(0)}\" with no signed or *Word macro")
    for m in re.findall(r"\\([A-Za-z]+)", strip_undefined_blocks(raw)):
        if m in defs:
            used.add(m)
        else:
            used.add(("?", m))

# Generated macro names are camel case; LaTeX and package commands are not. Anything
# undefined that carries an internal capital is a macro the pipeline stopped emitting.
# package commands that happen to carry an internal capital are not pipeline macros
PACKAGE_CMDS = {"FloatBarrier"}
missing = sorted({m for _, m in (u for u in used if isinstance(u, tuple))
                  if re.search(r"[a-z][A-Z]", m) and m not in PACKAGE_CMDS})
real_used = sorted(u for u in used if isinstance(u, str))
unused = sorted(d for d in defs if d not in real_used)

print(f"macros defined : {len(defs)}")
print(f"macros used    : {len(real_used)}")
print(f"defined, unused: {len(unused)}")
if missing:
    print("USED BUT UNDEFINED:", ", ".join(missing))
    sys.exit(1)
print("USED BUT UNDEFINED: none")
if sign_violations:
    print("DIRECTION ASSERTED ON AN ABSOLUTE VALUE INSIDE A GUARD:")
    for v in sign_violations:
        print("  " + v)
    sys.exit(1)
print("guarded sentences asserting a direction: none")
