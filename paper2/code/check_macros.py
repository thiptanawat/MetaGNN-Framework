#!/usr/bin/env python3
"""Guard: every macro used in the manuscript must be defined in numbers.tex, and every macro
defined there should be traceable. Run as part of the build so a missing result cannot slip through
as blank output.

The manuscript switches optional passages on \\ifdefined\\Macro ... \\else ... \\fi. The check
resolves those guards against the macros numbers.tex currently defines and then checks only the
branch that LaTeX would typeset, so a macro that is used only inside a guard that is off is not
reported, while a macro missing from an active branch is."""
import re, glob, os, sys
MS = 'ms' if os.path.isdir('ms') else 'manuscript'
# a checkout that carries the analysis without the manuscript sources has nothing to check
# against; say so rather than pass silently on an empty set of files
if not glob.glob(f'{MS}/body_*.tex'):
    print('no manuscript sources in this checkout: the macro file is written, but there is '
          'no text to check it against. This is the expected state of a code-and-results release.')
    raise SystemExit(0)
defined = set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', open(f'{MS}/numbers.tex').read()))

TOK = re.compile(r'\\ifdefined\\([A-Za-z]+)|\\else\b|\\fi\b')

def active_text(txt):
    """Return the text LaTeX would typeset given the macros currently defined."""
    out = []; pos = 0; stack = []          # stack entries: [keep_now, keep_after_else, depth]
    for m in TOK.finditer(txt):
        piece = txt[pos:m.start()]
        if all(s[0] for s in stack): out.append(piece)
        pos = m.end()
        if m.group(1) is not None:
            on = m.group(1) in defined
            stack.append([on, not on])
        elif m.group(0) == '\\else':
            if stack: stack[-1][0] = stack[-1][1]
        else:
            if stack: stack.pop()
    if all(s[0] for s in stack): out.append(txt[pos:])
    return ''.join(out)

used = set()
for f in glob.glob(f'{MS}/paper2.tex') + glob.glob(f'{MS}/body_*.tex'):
    txt = active_text(open(f).read())
    for m in re.findall(r'\\([A-Za-z]+)\{\}', txt): used.add(m)
    for m in re.findall(r'\\([A-Za-z]+)\\,\\%', txt): used.add(m)
    for m in re.findall(r'\\([A-Za-z]+)\{\}--\\([A-Za-z]+)\{\}', txt): used.update(m)
    # bare uses inside table cells and math, e.g. "& \rAInd{}" is caught above; "\rTPatient r" is a guard
latex = {'textbf', 'emph', 'texttt', 'textit', 'ref', 'cite', 'label', 'url', 'item', 'small',
         'noindent', 'centering', 'toprule', 'midrule', 'bottomrule', 'maketitle', 'today',
         'begin', 'end', 'caption', 'includegraphics', 'input', 'newcommand', 'thanks', 'date',
         'right', 'left', 'quad', 'mathrm', 'log', 'sloppy', 'itemsep', 'affil', 'author', 'title',
         'hfill', 'bigskip', 'medskip', 'smallskip', 'clearpage', 'newpage', 'linewidth', 'textwidth',
         'arraystretch', 'textsc', 'footnotesize', 'scriptsize', 'normalsize', 'par', 'hline', 'fi', 'else'}
# Anything used that is neither a generated macro nor a known LaTeX or package command is a
# macro the pipeline stopped emitting. Do not filter by name prefix: that silently exempts
# whole macro families and lets the guard pass on a missing result.
used = {u for u in used if u not in latex}
missing = sorted(u for u in used - defined if u not in latex)
unused = sorted(defined - used)
if missing:
    print('MISSING (used in the active text, not defined):'); [print('  \\' + m) for m in missing]
print(f'{len(defined)} macros defined, {len(used & defined)} used in the active text, {len(unused)} defined but unused')
sys.exit(1 if missing else 0)
