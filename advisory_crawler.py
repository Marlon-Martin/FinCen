import os
import csv
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------- CONFIG ---------------- #

START_URL = (
    "https://www.fincen.gov/resources/advisoriesbulletinsfact-sheets/advisories?field_date_release_value=&field_date_release_value_1=&field_tags_advisory_target_id=All&page=0"
)

DOWNLOAD_DIR = "fincen_advisory_pdfs"
CSV_FILE = "fincen_advisories.csv"

REQUEST_DELAY_PAGE = 0.5   # delay between listing pages
REQUEST_DELAY_ITEM = 0.0   # delay between individual advisories

# Single session to reuse connections (faster)
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; FinCEN-Scraper/1.0; +https://example.com)"
})

# ---------------- HELPERS ---------------- #

def get_soup(url: str) -> BeautifulSoup:
    resp = session.get(url)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def find_advisory_table(soup: BeautifulSoup):
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if "Title" in headers and "Date" in headers and "Description" in headers:
            return table
    return None


def extract_rows_from_table(table):
    tbody = table.find("tbody") or table
    return tbody.find_all("tr")


def extract_row_data(row, base_url):
    """Extract title, advisory URL, and date from a table row."""
    cells = row.find_all("td")
    if not cells or len(cells) < 2:
        return None
    a = cells[0].find("a", href=True)
    if not a:
        return None
    title = a.get_text(strip=True)
    advisory_url = urljoin(base_url, a["href"])
    date = cells[1].get_text(strip=True)
    return title, advisory_url, date


def find_pdf_url(advisory_url: str):
    soup = get_soup(advisory_url)

    # Strict: href ends with .pdf
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            return urljoin(advisory_url, href)

    # Fallback: .pdf appears anywhere in href
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if ".pdf" in href:
            return urljoin(advisory_url, a["href"])

    return None


def sanitize_filename(name: str) -> str:
    bad_chars = '<>:"/\\|?*'
    for ch in bad_chars:
        name = name.replace(ch, "_")
    return "_".join(name.split())[:180]


def filename_from_pdf_url(pdf_url, title=None):
    path = urlparse(pdf_url).path
    last_part = os.path.basename(path)
    if last_part and last_part.lower().endswith(".pdf"):
        return last_part
    if title:
        return sanitize_filename(title) + ".pdf"
    return "advisory.pdf"


def download_pdf(pdf_url, filepath):
    print(f"  [PDF ] Downloading {os.path.basename(filepath)}")
    resp = session.get(pdf_url, stream=True)
    resp.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)
    return filepath


def find_next_page_url(soup, current_url):
    a = soup.find("a", rel="next")
    if a and a.get("href"):
        return urljoin(current_url, a["href"])
    for a in soup.find_all("a", href=True):
        if "next" in a.get_text(strip=True).lower():
            return urljoin(current_url, a["href"])
    return None


def append_to_csv(row):
    """Append advisory metadata to the CSV log."""
    exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "date", "pdf_filename", "pdf_url"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_existing_titles():
    """Load already-downloaded advisory titles from the CSV into a set."""
    titles = set()
    if not os.path.exists(CSV_FILE):
        return titles
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("title")
            if t:
                titles.add(t.strip().lower())
    return titles

# ---------------- MAIN CRAWLER ---------------- #

def crawl_all_advisory_pdfs(start_url=START_URL):
    visited_pages = set()
    url = start_url

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"Download folder: {os.path.abspath(DOWNLOAD_DIR)}")

    # Load titles we’ve already downloaded (from past runs)
    existing_titles = load_existing_titles()
    print(f"Loaded {len(existing_titles)} existing titles from CSV.")

    while url and url not in visited_pages:
        print(f"\n[PAGE] {url}")
        visited_pages.add(url)

        soup = get_soup(url)
        table = find_advisory_table(soup)
        if not table:
            print("  [WARN] Could not find advisory table on this page.")
            break

        rows = extract_rows_from_table(table)
        print(f"  Found {len(rows)} advisories on this page.")

        for row in rows:
            data = extract_row_data(row, url)
            if not data:
                continue

            title, advisory_url, date = data
            title_lower = title.strip().lower()

            # Skip Guidance & Spanish advisories
            if "guidance" in title_lower:
                print(f"\n[SKIP 'Guidance'] {title}")
                continue
            if "spanish" in title_lower:
                print(f"\n[SKIP 'Spanish'] {title}")
                continue

            # Skip if this title already exists in our CSV
            if title_lower in existing_titles:
                print(f"\n[SKIP existing title] {title}")
                continue

            print(f"\n[ITEM] {title} ({date})")
            print(f"  [INFO] Advisory page: {advisory_url}")

            try:
                pdf_url = find_pdf_url(advisory_url)
            except Exception as e:
                print(f"  [ERR ] Failed to parse advisory page: {e}")
                continue

            if not pdf_url:
                print("  [WARN] No PDF link found on advisory page.")
                continue

            filename = filename_from_pdf_url(pdf_url, title)
            filepath = os.path.join(DOWNLOAD_DIR, filename)

            # Download and log
            try:
                download_pdf(pdf_url, filepath)
                append_to_csv({
                    "title": title,
                    "date": date,
                    "pdf_filename": filename,
                    "pdf_url": pdf_url
                })
                # Also add to in-memory set so we don't process it again in same run
                existing_titles.add(title_lower)
            except Exception as e:
                print(f"  [ERR ] Failed to download PDF: {e}")

            if REQUEST_DELAY_ITEM > 0:
                time.sleep(REQUEST_DELAY_ITEM)

        # Move to next page (no early stopping now)
        next_url = find_next_page_url(soup, url)
        if not next_url or next_url in visited_pages:
            print("\n[DONE] No more pages.")
            break

        url = next_url
        time.sleep(REQUEST_DELAY_PAGE)


if __name__ == "__main__":
    crawl_all_advisory_pdfs()
