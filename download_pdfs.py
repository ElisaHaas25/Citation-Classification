import os
import time
import ads
import requests
#from tqdm import tqdm
#from pdfminer.high_level import extract_text
from pathlib import Path
#import csv
from requests.exceptions import ReadTimeout, ConnectionError

os.environ["ADS_API_TOKEN"] = "4obdjlKI9smtjpuSyqLTpI5ZRx3m2BUpeYUXUfLe"

#---------------------------------------------
# get all papers that cite Coryn's distance estimation paper, only need to run this once 
#---------------------------------------------

start = 0
all_papers = []
rows = 200

print("Querying ADS...")

while True:
    query = ads.SearchQuery(
        q= ('citations(author:"Bailer-Jones" ''title:(estimating distance parallax) ''property:refereed)'),
        fl=["bibcode", "title", "author", "year", "identifier","esources", "links_data"],
        rows=rows,
        start=start)
    
    batch = []
    try:    
        for paper in query:
            batch.append(paper)

    except IndexError:
        # ADS iterator bug on final page
        pass

    if len(batch) == 0:
        break

    all_papers.extend(batch)
    start += rows

    print(f"Fetched {len(all_papers)} papers so far...")

print(f"Total papers found: {len(all_papers)}")

#---------------------------------------------
# download PDFs
#---------------------------------------------

PDF_PRIORITY = [
    "ADS_PDF",
    "EPRINT_PDF",
    "ADS_SCAN",
    "AUTHOR_PDF",
    "PUB_PDF",
]

def best_pdf(esources):
    if not esources:
        return None
    for p in PDF_PRIORITY:
        if p in esources:
            return p
    return None


for i in range(len(all_papers)): 
    
    paper = all_papers[i]
    link_type = best_pdf(paper.esources)

    if not link_type:
        print(f"[{i}/{len(all_papers)}] No PDF available for {paper.bibcode}")
        continue

    outfile = Path("ads_papers") / f"{paper.bibcode.replace('/', '_')}.pdf"

    if outfile.exists():
        print(f"[{i}/{len(all_papers)}] Skipping {outfile.name} (already exists)")
        continue    

    url = f"https://ui.adsabs.harvard.edu/link_gateway/{paper.bibcode}/{link_type}"

    try:
        r = requests.get(url, timeout=30)
    
    except (ReadTimeout, ConnectionError) as e:
        print(f"[{i}/{len(all_papers)}] Timeout: {paper.bibcode}")
        time.sleep(5)
        continue

    if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf"):
        with open(outfile, "wb") as f:
            f.write(r.content)

        print(f"[{i}/{len(all_papers)}] Downloaded {paper.bibcode} via {link_type}")
    else:
        print(f"[{i}/{len(all_papers)}] Failed to download {paper.bibcode} via {link_type}")


    time.sleep(1)  # polite rate limiting