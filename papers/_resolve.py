"""Resolve paper titles to arXiv IDs via the arXiv API (no summariser in the loop)."""
import sys, time, urllib.parse, urllib.request, re

def q(term, n=3):
    url = ("https://export.arxiv.org/api/query?search_query="
           + urllib.parse.quote(term) + f"&max_results={n}")
    raw = urllib.request.urlopen(url, timeout=40).read().decode()
    entries = raw.split("<entry>")[1:]
    out = []
    for e in entries:
        aid = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", e)
        ti  = re.search(r"<title>(.*?)</title>", e, re.S)
        dt  = re.search(r"<published>([\d-]+)", e)
        if aid and ti:
            out.append((aid.group(1), dt.group(1) if dt else "?",
                        " ".join(ti.group(1).split())))
    return out

for term in sys.argv[1:]:
    print(f"\n### {term}")
    try:
        for aid, dt, ti in q(term):
            print(f"  {aid:16s} {dt}  {ti[:95]}")
    except Exception as ex:
        print("  ERR", ex)
    time.sleep(3.5)
