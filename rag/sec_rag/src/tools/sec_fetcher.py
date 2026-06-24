#!/usr/bin/env python3
"""Fetch the most recent 10-K filings for six large-cap tech companies
from the SEC EDGAR public API and save them as HTML to data/html/.

Companies fetched
-----------------
  Apple       AAPL   CIK 0000320193
  Microsoft   MSFT   CIK 0000789019
  Nvidia      NVDA   CIK 0001045810
  Meta        META   CIK 0001326801
  Google      GOOGL  CIK 0001652044  (Alphabet Inc.)
  Amazon      AMZN   CIK 0001018724

EDGAR API used (free, no key required)
---------------------------------------
  Submissions  https://data.sec.gov/submissions/CIK{10-digit}.json
               Returns filing history as parallel columnar arrays.
  Archive      https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dashes}/{primaryDocument}
               The primary document for a 10-K is an iXBRL HTML file.

Rate limiting
-------------
  SEC enforces a hard limit of 10 requests/second.
  We wait 0.12 s between every request (≈ 8 req/s) to stay safely under it.
  A User-Agent header identifying the caller is required; requests without
  one are rejected with HTTP 403.

Output
------
  data/html/{ticker}/{ticker}-{fiscal_period}.htm
  data/html/{ticker}/{ticker}-{fiscal_period}.meta.json   ← sidecar for chunker.py

Usage
-----
  uv add httpx
  uv run python sec_fetcher.py
  uv run python sec_fetcher.py --filings 3        # fetch last 3 10-Ks per company
  uv run python sec_fetcher.py --output-dir raw   # custom output directory
  uv run python sec_fetcher.py --force            # re-download existing files
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sec_fetcher")

# --------------------------------------------------------------------------- #
# Target companies
# --------------------------------------------------------------------------- #
@dataclass
class Company:
    ticker: str
    cik: str        # zero-padded to 10 digits
    name: str

COMPANIES: list[Company] = [
    Company("AAPL",  "0000320193", "Apple Inc."),
    Company("MSFT",  "0000789019", "Microsoft Corporation"),
    Company("NVDA",  "0001045810", "NVIDIA Corporation"),
    Company("META",  "0001326801", "Meta Platforms, Inc."),
    Company("GOOGL", "0001652044", "Alphabet Inc."),
    Company("AMZN",  "0001018724", "Amazon.com, Inc."),
]

# --------------------------------------------------------------------------- #
# EDGAR endpoints
# --------------------------------------------------------------------------- #
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL     = (
    "https://www.sec.gov/Archives/edgar/data"
    "/{cik_plain}/{acc_folder}/{primary_doc}"
)

# The User-Agent is required — SEC returns 403 without it.
HEADERS = {
    "User-Agent": "rag-demo rag-demo@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

# Seconds between every HTTP request — keeps us under the 10 req/s cap.
REQUEST_DELAY = 0.12


# --------------------------------------------------------------------------- #
# EDGAR helpers
# --------------------------------------------------------------------------- #
def get_submissions(client: httpx.Client, cik: str) -> dict:
    """Fetch the submissions JSON for a company (CIK must be 10-digit padded)."""
    url = SUBMISSIONS_URL.format(cik=cik)
    time.sleep(REQUEST_DELAY)
    resp = client.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def extract_10k_filings(submissions: dict, max_filings: int) -> list[dict]:
    """Pull 10-K rows from the columnar submissions payload.

    The ``filings.recent`` object is a dict of parallel arrays — one entry
    per column (form, accessionNumber, primaryDocument, …).  We zip them
    into row dicts and filter for form type "10-K".
    """
    recent = submissions["filings"]["recent"]
    forms       = recent.get("form", [])
    acc_numbers = recent.get("accessionNumber", [])
    report_dates= recent.get("reportDate", [])
    filing_dates= recent.get("filingDate", [])
    primary_docs= recent.get("primaryDocument", [])

    rows = []
    for form, acc, rep_date, fil_date, doc in zip(
        forms, acc_numbers, report_dates, filing_dates, primary_docs
    ):
        if form == "10-K":
            rows.append({
                "form":             form,
                "accessionNumber":  acc,
                "reportDate":       rep_date,
                "filingDate":       fil_date,
                "primaryDocument":  doc,
            })
            if len(rows) >= max_filings:
                break
    return rows


def build_archive_url(cik: str, acc_number: str, primary_doc: str) -> str:
    """Construct the direct URL to the primary HTML document.

    Archive path uses the CIK *without* leading zeros, while the submissions
    endpoint requires the 10-digit padded version.
    """
    cik_plain  = str(int(cik))          # strip leading zeros
    acc_folder = acc_number.replace("-", "")
    return ARCHIVE_URL.format(
        cik_plain=cik_plain,
        acc_folder=acc_folder,
        primary_doc=primary_doc,
    )


def download_html(client: httpx.Client, url: str) -> str:
    time.sleep(REQUEST_DELAY)
    resp = client.get(url, headers=HEADERS, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def fiscal_period_from_report_date(report_date: str) -> str:
    """Convert '2023-09-30' to a human label like 'FY2023'.

    For SEC annual filings the reportDate is the fiscal year-end date.
    We use the calendar year of that date as the fiscal year label.
    """
    if report_date:
        return f"FY{report_date[:4]}"
    return "FYunknown"


# --------------------------------------------------------------------------- #
# Per-company fetch
# --------------------------------------------------------------------------- #
def fetch_company(
    client: httpx.Client,
    company: Company,
    output_dir: Path,
    max_filings: int,
    force: bool,
) -> tuple[int, int]:
    """Fetch and save 10-K filings for one company.

    Returns (saved, skipped) counts.
    """
    log.info("── %s (%s, CIK %s) ──", company.name, company.ticker, company.cik)
    company_dir = output_dir / company.ticker
    company_dir.mkdir(parents=True, exist_ok=True)

    try:
        submissions = get_submissions(client, company.cik)
    except httpx.HTTPStatusError as exc:
        log.error("Failed to fetch submissions: %s", exc)
        return 0, 0

    filings = extract_10k_filings(submissions, max_filings)
    if not filings:
        log.warning("No 10-K filings found for %s.", company.ticker)
        return 0, 0

    log.info("Found %d 10-K filing(s) to process.", len(filings))
    saved = skipped = 0

    for filing in filings:
        fiscal_period = fiscal_period_from_report_date(filing["reportDate"])
        stem          = f"{company.ticker.lower()}-{fiscal_period.lower()}"
        html_path     = company_dir / f"{stem}.htm"
        meta_path     = company_dir / f"{stem}.meta.json"

        if html_path.exists() and not force:
            log.info("  Skip (exists): %s", html_path.name)
            skipped += 1
            continue

        url = build_archive_url(
            company.cik,
            filing["accessionNumber"],
            filing["primaryDocument"],
        )
        log.info(
            "  Downloading %s %s (filed %s) …",
            company.ticker, fiscal_period, filing["filingDate"],
        )

        try:
            html = download_html(client, url)
        except httpx.HTTPStatusError as exc:
            log.error("  Download failed: %s", exc)
            continue

        html_path.write_text(html, encoding="utf-8")

        # Write sidecar metadata for chunker.py
        meta = {
            "cik":           company.cik,
            "ticker":        company.ticker,
            "company":       company.name,
            "form_type":     filing["form"],
            "fiscal_period": fiscal_period,
            "filing_date":   filing["filingDate"],
            "url":           url,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        size_kb = html_path.stat().st_size // 1024
        log.info("  Saved: %s (%d KB)", html_path.name, size_kb)
        saved += 1

    return saved, skipped


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch 10-K filings from SEC EDGAR into data/html/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("data/html"),
        help="Root directory to write HTML files.",
    )
    p.add_argument(
        "--filings", type=int, default=1,
        help="Number of most recent 10-Ks to fetch per company.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-download files that already exist.",
    )
    p.add_argument(
        "--timeout", type=float, default=60.0,
        help="HTTP request timeout in seconds.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_saved = total_skipped = total_failed = 0

    with httpx.Client(timeout=args.timeout) as client:
        for company in COMPANIES:
            try:
                saved, skipped = fetch_company(
                    client, company, output_dir, args.filings, args.force
                )
                total_saved   += saved
                total_skipped += skipped
            except Exception:  # noqa: BLE001
                log.exception("Unexpected error for %s", company.ticker)
                total_failed += 1

    log.info(
        "Done — saved: %d, skipped: %d, failed: %d",
        total_saved, total_skipped, total_failed,
    )
    log.info("Output: %s", output_dir.resolve())
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())