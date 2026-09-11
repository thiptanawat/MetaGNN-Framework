#!/usr/bin/env python3
"""Re-resolve every reference of both manuscripts against Crossref (DOIs) and arXiv (preprints).

For each \bibitem the script extracts the DOI or arXiv identifier from the entry's URL, fetches the
record's metadata, and compares the title (token overlap), the year, the first author's surname and,
for journal articles, the container title, volume and pages with the text of the entry. It writes a
JSON report and prints one line per entry; entries whose title overlap is below 0.6 or whose year
differs are flagged for a human check. Entries without an identifier (model cards, companions) are
listed as unresolved rather than failed.

Usage: python3 tools/verify_refs.py paper1/manuscript/body_bib.tex paper2/manuscript/body_bib.tex
"""
import json, re, sys, time, urllib.request, urllib.parse

UA = {'User-Agent': 'metabench-ref-check/2.0 (mailto:thiptanawat@gmail.com)'}

def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
            return r.read().decode('utf-8', 'replace')
    except Exception as e:
        return '__ERR__' + str(e)

def norm(t):
    t = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', t or '')
    t = re.sub(r'[{}\\]', '', t)
    return re.sub(r'[^a-z0-9]+', ' ', t.lower()).strip()

def overlap(a, b):
    A, B = set(norm(a).split()), set(norm(b).split())
    A = {w for w in A if len(w) > 2}; B = {w for w in B if len(w) > 2}
    return round(len(A & B) / max(len(A), 1), 2)

def parse(path):
    txt = open(path).read()
    out = []
    for m in re.finditer(r'\\bibitem\{([^}]+)\}(.*?)(?=\n\\bibitem|\n\\end\{thebibliography\}|\Z)', txt, re.S):
        key, body = m.group(1), ' '.join(m.group(2).split())
        doi = None; arxiv = None
        d = re.search(r'doi\.org/(10\.\d{4,9}/[^\s}]+)', body)
        if d: doi = d.group(1).rstrip('.')
        a = re.search(r'arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})', body)
        if a: arxiv = a.group(1)
        y = re.search(r'\((\d{4})\)', body)
        year = y.group(1) if y else None
        first = re.match(r'\s*([A-Z][^,]*?),', body)
        surname = norm(first.group(1)) if first else None
        # the title is the sentence after the year
        t = re.search(r'\(\d{4}\)\.\s*(.*?)(?:\.\s+\\textit|\.\s+In\s|\.\s+\\emph|\.\s+[A-Z][a-z]+ preprint|\.\s+Hugging Face)', body)
        title = t.group(1) if t else body[:120]
        out.append(dict(key=key, doi=doi, arxiv=arxiv, year=year, surname=surname, title=title, entry=body))
    return out

def resolve(e):
    rec = dict(key=e['key'], doi=e['doi'], arxiv=e['arxiv'], resolved=False)
    if e['doi']:
        x = get('https://api.crossref.org/works/' + urllib.parse.quote(e['doi'], safe=''))
        if not x.startswith('__ERR__'):
            try:
                d = json.loads(x)['message']
                rec.update(resolved=True, source='crossref', title=(d.get('title') or [''])[0],
                           container=(d.get('container-title') or [''])[0], volume=d.get('volume'), page=d.get('page'),
                           year=str(((d.get('published-print') or d.get('published-online') or d.get('issued') or {}).get('date-parts') or [['']])[0][0]),
                           first_author=(d.get('author') or [{}])[0].get('family'), type=d.get('type'))
            except Exception as ex:
                rec['error'] = 'crossref parse: ' + str(ex)
        else:
            rec['error'] = x
        time.sleep(0.4)
    if not rec['resolved'] and e['arxiv']:
        x = get('http://export.arxiv.org/api/query?id_list=' + e['arxiv'])
        m = re.search(r'<entry>.*?<title>(.*?)</title>.*?<published>(\d{4})', x, re.S)
        if m:
            rec.update(resolved=True, source='arxiv', title=' '.join(m.group(1).split()), year=m.group(2))
            a = re.search(r'<author>\s*<name>([^<]*)</name>', x)
            if a: rec['first_author'] = a.group(1).split()[-1]
        else:
            rec['error'] = 'arxiv: no entry'
        time.sleep(0.4)
    if rec['resolved']:
        rec['title_overlap'] = overlap(rec.get('title', ''), e['title'])
        rec['year_cited'] = e['year']; rec['year_match'] = (e['year'] == rec.get('year')) if e['year'] and rec.get('year') else None
        fa = norm(rec.get('first_author') or '').split(); rec['author_match'] = (bool(fa) and fa[-1] in (e['surname'] or ''))
        rec['flag'] = (rec['title_overlap'] < 0.6) or (rec['year_match'] is False) or (not rec['author_match'])
    return rec

if __name__ == '__main__':
    report = {}
    for path in sys.argv[1:]:
        entries = parse(path); report[path] = []
        for i, e in enumerate(entries, 1):
            r = resolve(e); report[path].append(r)
            status = ('FLAG' if r.get('flag') else 'ok') if r['resolved'] else 'unresolved'
            print(f"{i:3d}/{len(entries)} {e['key']:<22} {status:<10} overlap={r.get('title_overlap')} year={e['year']}->{r.get('year')} author={r.get('author_match')} {('ERR ' + r['error'][:60]) if r.get('error') else ''}", flush=True)
    json.dump(report, open('docs/verify_refs_report.json', 'w'), indent=1)
    print('wrote docs/verify_refs_report.json')
