import json, re, time, urllib.request, urllib.parse, sys
E = json.load(open('/tmp/refs.json'))
def get(url, hdr=None):
    try:
        req = urllib.request.Request(url, headers=hdr or {'User-Agent':'ref-check/1.0 (mailto:thiptanawat@gmail.com)'})
        with urllib.request.urlopen(req, timeout=30) as r: return r.read().decode('utf-8', 'replace')
    except Exception as e: return f'__ERR__{e}'
def norm(t): return re.sub(r'[^a-z0-9]+',' ', (t or '').lower()).strip()
def overlap(a, b):
    A, B = set(norm(a).split()), set(norm(b).split())
    return len(A & B) / max(len(A), 1)
out=[]
for i, e in enumerate(E, 1):
    rec = dict(key=e['key'], pmid_ok=None, doi_ok=None, arxiv_ok=None, title_pubmed=None, title_crossref=None, title_arxiv=None)
    if e['pmid']:
        x = get(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={e['pmid']}&retmode=json")
        try:
            d = json.loads(x)['result'][e['pmid']]
            rec['title_pubmed'] = d.get('title'); rec['pmid_ok'] = bool(d.get('title'))
            rec['journal_pubmed'] = d.get('fulljournalname'); rec['year_pubmed'] = (d.get('pubdate') or '')[:4]
        except Exception: rec['pmid_ok'] = False
        time.sleep(0.35)
    if e['doi']:
        x = get(f"https://api.crossref.org/works/{urllib.parse.quote(e['doi'], safe='')}")
        try:
            d = json.loads(x)['message']
            rec['title_crossref'] = (d.get('title') or [''])[0]; rec['doi_ok'] = True
            rec['journal_crossref'] = (d.get('container-title') or [''])[0]
            dp = d.get('published-print') or d.get('published-online') or {}
            rec['year_crossref'] = str((dp.get('date-parts') or [['']])[0][0])
        except Exception: rec['doi_ok'] = False
        time.sleep(0.35)
    if e['arxiv']:
        x = get(f"http://export.arxiv.org/api/query?id_list={e['arxiv']}")
        m = re.search(r'<entry>.*?<title>(.*?)</title>', x, re.S)
        rec['title_arxiv'] = ' '.join(m.group(1).split()) if m else None; rec['arxiv_ok'] = bool(m)
        time.sleep(0.35)
    # title agreement between the cited text and whatever resolved
    resolved = rec['title_pubmed'] or rec['title_crossref'] or rec['title_arxiv']
    rec['title_match'] = round(overlap(resolved, e['cite']), 2) if resolved else None
    out.append(rec); print(f"{i:2d}/{len(E)} {e['key']:<18} pmid={rec['pmid_ok']} doi={rec['doi_ok']} arxiv={rec['arxiv_ok']} match={rec['title_match']}", flush=True)
json.dump(out, open('verify_topic4.json','w'), indent=1)
