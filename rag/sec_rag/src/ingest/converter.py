#!/usr/bin/env python3
"""Convert SEC HTML filings to Markdown using the Docling library.

Walks an input directory (default: ``data/html``) for ``.html`` / ``.htm``
files, converts each to Markdown with Docling, and writes the result to a
mirrored path under an output directory (default: ``data/markdown``).

The script is idempotent: existing outputs are skipped unless ``--force`` is
given, and a failure on one filing is logged without aborting the batch.

Examples
--------
    uv run python converter.py
    uv run python converter.py --input-dir data/html --output-dir data/markdown
    uv run python converter.py --force        # re-convert everything
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.document_converter import DocumentConverter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("converter")

# EDGAR commonly uses the ".htm" extension; accept both.
HTML_SUFFIXES = {".html", ".htm"}


def find_html_files(input_dir: Path) -> list[Path]:
    """Return all HTML files under ``input_dir``, recursively and sorted."""
    return sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in HTML_SUFFIXES
    )


def output_path_for(src: Path, input_dir: Path, output_dir: Path) -> Path:
    """Map an input HTML path to its mirrored ``.md`` output path."""
    relative = src.relative_to(input_dir).with_suffix(".md")
    return output_dir / relative


def convert_one(converter: DocumentConverter, src: Path, dest: Path) -> bool:
    """Convert a single file. Returns True on a usable result, False on failure."""
    try:
        result = converter.convert(src, raises_on_error=False)
    except Exception:  # noqa: BLE001 - never let one filing kill the batch
        log.exception("Unexpected error converting %s", src)
        return False

    if result.status == ConversionStatus.FAILURE:
        errors = "; ".join(str(e) for e in result.errors) or "unknown error"
        log.error("Failed: %s (%s)", src, errors)
        return False
    if result.status == ConversionStatus.PARTIAL_SUCCESS:
        log.warning("Partial conversion: %s (writing available content)", src)

    try:
        markdown = result.document.export_to_markdown()

        # Content validation: warn if markdown is suspiciously short
        markdown_stripped = markdown.strip()
        if len(markdown_stripped) < 100:
            log.warning(
                "Suspiciously short markdown for %s (%d chars) — may be incomplete",
                src, len(markdown_stripped)
            )

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(markdown, encoding="utf-8")

        # Metadata propagation: copy sidecar .meta.json if it exists
        meta_src = src.with_suffix(".meta.json")
        if meta_src.exists():
            meta_dest = dest.with_suffix(".meta.json")
            meta_content = meta_src.read_text(encoding="utf-8")
            meta_dest.write_text(meta_content, encoding="utf-8")
            log.debug("Copied metadata: %s → %s", meta_src.name, meta_dest.name)

    except Exception:  # noqa: BLE001
        log.exception("Error writing markdown for %s", src)
        return False

    return True


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SEC HTML filings to Markdown with Docling."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/html"),
        help="Directory containing HTML filings (default: data/html).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/markdown"),
        help="Directory to write Markdown into (default: data/markdown).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert files even if the Markdown output already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        log.error("Input directory does not exist: %s", input_dir)
        return 1

    files = find_html_files(input_dir)
    if not files:
        log.warning("No .html/.htm files found under %s", input_dir)
        return 0

    log.info("Found %d HTML file(s) under %s", len(files), input_dir)

    # Restricting allowed_formats keeps the pipeline lean (HTML needs no ML
    # models) and ignores any stray non-HTML files. The converter caches its
    # pipeline, so reusing one instance across all files is efficient.
    converter = DocumentConverter(allowed_formats=[InputFormat.HTML])

    converted = skipped = failed = 0
    started = time.perf_counter()

    for i, src in enumerate(files, start=1):
        dest = output_path_for(src, input_dir, output_dir)

        if dest.exists() and not args.force:
            log.info("[%d/%d] Skip (exists): %s", i, len(files), dest)
            skipped += 1
            continue

        log.info("[%d/%d] Converting: %s", i, len(files), src)
        if convert_one(converter, src, dest):
            converted += 1
        else:
            failed += 1

    elapsed = time.perf_counter() - started
    log.info(
        "Done in %.1fs — converted: %d, skipped: %d, failed: %d",
        elapsed,
        converted,
        skipped,
        failed,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())