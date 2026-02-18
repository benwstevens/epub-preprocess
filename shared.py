"""
Shared utilities for the ebook processing pipelines.

All three scripts (pipeline.py, abridger.py, distiller.py) use these
common functions for EPUB conversion, chapter splitting, heading
detection, chapter selection, Markdown conversion, and API key loading.
"""

import os
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BASE_DIR = Path(__file__).resolve().parent
HEADING_TAGS = ["h1", "h2", "h3", "h4"]


# ===========================================================================
# Book directory setup
# ===========================================================================
def setup_book_dir(epub_path: Path) -> dict[str, Path]:
    """Create a per-book directory structure under books/ and return paths."""
    book_slug = re.sub(r"[^\w\-]", "", epub_path.stem.replace(" ", "_")).lower()
    book_dir = BASE_DIR / "books" / book_slug

    paths = {
        "book_dir": book_dir,
        "source_dir": book_dir / "source",
        "chapters_dir": book_dir / "chapters",
        "chapter_summaries_dir": book_dir / "chapter_summaries",
        "abridged_dir": book_dir / "abridged_chapters",
        "distilled_dir": book_dir / "distilled_chapters",
        "output_dir": book_dir / "output",
        "selection_file": book_dir / "selected_chapters.txt",
    }

    # Create all directories
    for key, p in paths.items():
        if key != "selection_file":
            p.mkdir(parents=True, exist_ok=True)

    # Copy the EPUB into the book's source directory if not already there
    dest_epub = paths["source_dir"] / epub_path.name
    if not dest_epub.exists():
        import shutil
        shutil.copy2(epub_path, dest_epub)
        print(f"Copied {epub_path.name} to {dest_epub.relative_to(BASE_DIR)}")

    return paths


def find_source_file(paths: dict[str, Path]) -> Path:
    """Locate the source file (HTML or EPUB). Convert EPUB to HTML if needed."""
    source_dir = paths["source_dir"]
    html_files = list(source_dir.glob("*.html")) + list(source_dir.glob("*.htm"))
    epub_files = list(source_dir.glob("*.epub"))

    if html_files:
        if len(html_files) > 1:
            print(f"WARNING: Multiple HTML files found:")
            for p in html_files:
                print(f"  {p.name}")
            print("Using the first one.")
        return html_files[0]

    if epub_files:
        if len(epub_files) > 1:
            print(f"WARNING: Multiple EPUB files found:")
            for p in epub_files:
                print(f"  {p.name}")
            print("Using the first one.")
        return convert_epub_to_html(epub_files[0], paths["source_dir"])

    print(f"ERROR: No .html or .epub files found in {source_dir}/")
    sys.exit(1)


# ===========================================================================
# EPUB-to-HTML conversion
# ===========================================================================
def convert_epub_to_html(epub_path: Path, source_dir: Path) -> Path:
    """Convert an EPUB file to a single HTML file."""
    from ebooklib import epub as ep

    print(f"Converting EPUB to HTML: {epub_path.name}")
    book = ep.read_epub(str(epub_path))

    title = book.get_metadata("DC", "title")
    title = title[0][0] if title else epub_path.stem
    author = book.get_metadata("DC", "creator")
    author = author[0][0] if author else ""

    # Build set of top-level TOC hrefs (chapters only, ignore subsections)
    chapter_hrefs: dict[str, str] = {}  # base href -> title
    for toc_entry in book.toc:
        if hasattr(toc_entry, "href") and hasattr(toc_entry, "title"):
            href = toc_entry.href.split("#")[0]
            if toc_entry.title:
                chapter_hrefs[href] = toc_entry.title
        elif isinstance(toc_entry, tuple) and len(toc_entry) == 2:
            section, children = toc_entry
            if hasattr(section, "href") and hasattr(section, "title"):
                href = section.href.split("#")[0]
                if section.title:
                    chapter_hrefs[href] = section.title

    if chapter_hrefs:
        print(f"  Found {len(chapter_hrefs)} top-level TOC entries in EPUB")

    html_parts = [
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
    ]
    if author:
        html_parts.append(f'<meta name="author" content="{author}">\n')
    html_parts.append("</head>\n<body>\n")

    for item in book.get_items_of_type(9):  # ITEM_DOCUMENT
        soup = BeautifulSoup(item.get_content(), "lxml")
        body = soup.find("body")
        body_html = ""
        if body:
            body_html = "".join(str(c) for c in body.children)
        else:
            body_html = soup.get_text()

        item_filename = item.get_name().split("/")[-1]
        toc_title = chapter_hrefs.get(item.get_name()) or chapter_hrefs.get(item_filename)

        if toc_title:
            has_heading = bool(
                BeautifulSoup(body_html, "lxml").find(["h1", "h2", "h3", "h4"])
            )
            if not has_heading:
                html_parts.append(f"<h2>{toc_title}</h2>\n")

        html_parts.append(body_html)
        html_parts.append("\n")

    html_parts.append("</body>\n</html>")

    output_path = source_dir / (epub_path.stem + ".html")
    output_path.write_text("".join(html_parts), encoding="utf-8")
    print(f"Converted to: {output_path.name}")
    return output_path


# ===========================================================================
# Stage 1 – Identify chapter heading tag
# ===========================================================================
def detect_chapter_tag(html_path: Path) -> str:
    """Detect which heading tag marks chapter titles."""
    print("\n" + "=" * 60)
    print("STAGE 1: Identify Chapter Heading Tag")
    print("=" * 60)

    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    heading_info: dict[str, list[str]] = {}
    for tag in HEADING_TAGS:
        elements = soup.find_all(tag)
        texts = [el.get_text(strip=True) for el in elements]
        if texts:
            heading_info[tag] = texts

    if not heading_info:
        all_tags = Counter(tag.name for tag in soup.find_all(True))
        print("ERROR: No heading tags (h1-h4) found in the source file.")
        print("\nMost common tags found:")
        for tag_name, count in all_tags.most_common(10):
            print(f"  <{tag_name}> — {count}")
        print("\nIf converting from EPUB, try deleting the .html file in")
        print("source/ and re-running so it reconverts from the EPUB.")
        sys.exit(1)

    print("\nHeading tag survey:")
    for tag, texts in heading_info.items():
        print(f"  <{tag}> — {len(texts)} occurrence(s)")
        for t in texts[:5]:
            print(f"         . {t[:80]}")
        if len(texts) > 5:
            print(f"         ... and {len(texts) - 5} more")

    scores: dict[str, float] = {}
    chapter_pattern = re.compile(
        r"^(chapter|part|section|book|act|prologue|epilogue|introduction|"
        r"conclusion|foreword|afterword|preface)\s",
        re.IGNORECASE,
    )

    for tag, texts in heading_info.items():
        count = len(texts)
        if count < 2:
            scores[tag] = 0
            continue

        chapter_matches = sum(1 for t in texts if chapter_pattern.search(t))
        chapter_ratio = chapter_matches / count if count else 0

        # Base score from count, but diminishing — 10 chapters and 50 chapters
        # shouldn't be that different
        score = min(count, 20)

        # Chapter-like text is the strongest signal
        score += chapter_matches * 10

        # If most headings at this level look like chapters, big bonus
        if chapter_ratio > 0.3:
            score *= 2.0

        scores[tag] = score

    if not scores or max(scores.values()) == 0:
        print("ERROR: Could not determine chapter heading tag.")
        sys.exit(1)

    best_tag = max(scores, key=lambda t: scores[t])
    runner_up = sorted(scores, key=lambda t: scores[t], reverse=True)

    ambiguous = False
    if len(runner_up) >= 2:
        s1, s2 = scores[runner_up[0]], scores[runner_up[1]]
        if s2 > 0 and s1 / max(s2, 0.01) < 1.5:
            ambiguous = True

    print(f"\nDetected chapter heading tag: <{best_tag}>")

    if ambiguous:
        print(f"  (Runner-up: <{runner_up[1]}> — scores are close, please confirm)")
        answer = input(f"Use <{best_tag}> as the chapter heading? [Y/n] ").strip()
        if answer.lower() == "n":
            alt = input("Enter the tag to use instead (e.g. h2): ").strip().lower()
            if alt in heading_info:
                best_tag = alt
            else:
                print(f"Tag <{alt}> not found in the document. Aborting.")
                sys.exit(1)

    return best_tag


# ===========================================================================
# Stage 2 – Split into chapter files
# ===========================================================================
def sanitize_filename(text: str, max_len: int = 60) -> str:
    text = text.strip()
    text = re.sub(r"&\w+;", "", text)
    text = re.sub(r"[^\w\s\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def split_chapters(html_path: Path, chapter_tag: str, chapters_dir: Path) -> list[Path]:
    """Split the source HTML into individual chapter files."""
    print("\n" + "=" * 60)
    print("STAGE 2: Split into Chapter Files")
    print("=" * 60)

    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    headings = soup.find_all(chapter_tag)
    if not headings:
        print(f"ERROR: No <{chapter_tag}> tags found.")
        sys.exit(1)

    # Collect all content between each heading and the next one.
    # We use string positions rather than DOM sibling traversal,
    # because headings may be nested inside wrappers (section, header,
    # div) that prevent find_next_siblings() from reaching body text.
    body = soup.find("body")
    full_html = str(body) if body else str(soup)

    heading_positions = []
    for heading in headings:
        heading_str = str(heading)
        title = heading.get_text(strip=True)
        pos = full_html.find(heading_str)
        if pos >= 0:
            heading_positions.append((pos, title))

    chapter_data: list[tuple[str, str]] = []
    for i, (pos, title) in enumerate(heading_positions):
        end_pos = heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(full_html)
        chapter_html = full_html[pos:end_pos]
        chapter_data.append((title, chapter_html))

    chapters_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for idx, (title, html_content) in enumerate(chapter_data, start=1):
        safe_title = sanitize_filename(title) or f"Chapter_{idx}"
        filename = f"{idx:02d} - {safe_title}.html"
        filepath = chapters_dir / filename

        full_html = (
            "<!DOCTYPE html>\n<html>\n<head>\n"
            '<meta charset="utf-8">\n'
            f"<title>{title}</title>\n"
            "</head>\n<body>\n"
            f"{html_content}\n"
            "</body>\n</html>"
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_html)
        written.append(filepath)

    print(f"\nSplit into {len(written)} chapters:")
    for p in written:
        print(f"  {p.name}")

    return written


# ===========================================================================
# Chapter selection / filtering
# ===========================================================================
def filter_chapters(
    chapter_files: list[Path], selection_file: Path, label: str = "processing"
) -> list[Path]:
    """Let the user select which chapters to process. Saves selection to file."""
    print("\n" + "=" * 60)
    print("Chapter Selection")
    print("=" * 60)

    if selection_file.exists():
        saved = [
            line.strip()
            for line in selection_file.read_text().splitlines()
            if line.strip()
        ]
        if saved:
            matched = [f for f in chapter_files if f.name in saved]
            if matched:
                print(f"\nFound saved selection ({len(matched)} chapters):")
                for f in matched:
                    print(f"  {f.name}")
                reuse = input("\nUse this selection? [Y/n] ").strip()
                if reuse.lower() != "n":
                    return matched
                print()

    print("\nAll chapters found:")
    for i, f in enumerate(chapter_files, 1):
        print(f"  {i:3d}. {f.stem}")

    print(
        "\nEnter chapter numbers to EXCLUDE (comma-separated, ranges OK), "
        "or press Enter to keep all."
    )
    print("Example: 1-9  or  1,2,15-18  to skip front matter")
    exclude_input = input("\nExclude: ").strip()

    if not exclude_input:
        selected = chapter_files
    else:
        try:
            exclude_nums = set()
            for part in exclude_input.split(","):
                part = part.strip()
                if "-" in part:
                    lo, hi = part.split("-", 1)
                    exclude_nums.update(range(int(lo), int(hi) + 1))
                elif part:
                    exclude_nums.add(int(part))
        except ValueError:
            print("Invalid input. Keeping all chapters.")
            exclude_nums = set()

        selected = [
            f
            for i, f in enumerate(chapter_files, 1)
            if i not in exclude_nums
        ]

    selection_file.write_text(
        "\n".join(f.name for f in selected) + "\n"
    )

    print(f"\nSelected {len(selected)} chapters for {label}:")
    for f in selected:
        print(f"  {f.name}")

    return selected


# ===========================================================================
# Utility functions
# ===========================================================================
def count_words(text: str) -> int:
    """Count words in text, stripping HTML tags first."""
    clean = re.sub(r"<[^>]+>", " ", text)
    return len(clean.split())


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return len(text) // 4


def get_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY not found.\n"
            "Set it in the .env file in the project root:\n"
            "  1. Open the file called .env\n"
            "  2. Replace YOUR_KEY_HERE with your actual API key\n"
            "Get a key at: https://console.anthropic.com/settings/keys"
        )
        sys.exit(1)
    return api_key


def markdown_to_html(text: str) -> str:
    """Convert common Markdown patterns to HTML if Markdown is detected."""
    if not re.search(r"(^#{1,4}\s|\*\*|^>\s|^- )", text, re.MULTILINE):
        return text

    lines = text.split("\n")
    html_lines = []
    in_list = False
    in_blockquote = False

    for line in lines:
        stripped = line.strip()

        heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading_match:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            if in_blockquote:
                html_lines.append("</blockquote>")
                in_blockquote = False
            level = len(heading_match.group(1))
            h_level = min(level + 1, 4)
            html_lines.append(f"<h{h_level}>{heading_match.group(2).strip()}</h{h_level}>")
            continue

        if stripped.startswith("> "):
            if not in_blockquote:
                html_lines.append("<blockquote>")
                in_blockquote = True
            html_lines.append(f"<p>{stripped[2:]}</p>")
            continue
        elif in_blockquote and stripped:
            html_lines.append("</blockquote>")
            in_blockquote = False

        if re.match(r"^[-*]\s+", stripped):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            item_text = re.sub(r"^[-*]\s+", "", stripped)
            html_lines.append(f"<li>{item_text}</li>")
            continue
        elif in_list and not stripped:
            html_lines.append("</ul>")
            in_list = False

        if not stripped:
            continue

        html_lines.append(f"<p>{stripped}</p>")

    if in_list:
        html_lines.append("</ul>")
    if in_blockquote:
        html_lines.append("</blockquote>")

    result = "\n".join(html_lines)
    result = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", result)
    result = re.sub(r"\*(.+?)\*", r"<em>\1</em>", result)

    return result


def detect_book_metadata(source_dir: Path) -> tuple[str, str]:
    """Try to detect book title and author from source HTML."""
    source_files = list(source_dir.glob("*.html")) + list(source_dir.glob("*.htm"))
    book_title = ""
    book_author = ""

    if source_files:
        with open(source_files[0], "r", encoding="utf-8") as f:
            src_soup = BeautifulSoup(f, "lxml")
        title_tag = src_soup.find("title")
        if title_tag and title_tag.get_text(strip=True):
            book_title = title_tag.get_text(strip=True)
        meta_author = src_soup.find("meta", attrs={"name": "author"})
        if meta_author:
            book_author = meta_author.get("content", "")

    if not book_title:
        book_title = input("Enter book title: ").strip() or "Untitled"
    else:
        confirm = input(f'Detected title: "{book_title}". Use this? [Y/n] ').strip()
        if confirm.lower() == "n":
            book_title = input("Enter book title: ").strip() or "Untitled"

    if not book_author:
        book_author = input("Enter author name: ").strip() or "Unknown"
    else:
        confirm = input(f'Detected author: "{book_author}". Use this? [Y/n] ').strip()
        if confirm.lower() == "n":
            book_author = input("Enter author name: ").strip() or "Unknown"

    return book_title, book_author


def wrap_response_html(response_text: str, chapter_path: Path, suffix: str) -> str:
    """Convert Markdown if needed, wrap in full HTML document."""
    response_text = markdown_to_html(response_text)
    title = chapter_path.stem

    if not re.search(r"<\w+[\s>]", response_text):
        response_text = f"<h2>{title}</h2>\n<p>{response_text}</p>\n"

    response_text = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        "</head>\n<body>\n"
        f"{response_text}\n"
        "</body>\n</html>"
    )

    return response_text


def resolve_epub_path(epub_arg: str) -> Path:
    """Resolve the EPUB path from a CLI argument.

    If it's an existing file path, use it directly.
    If it matches a book slug in books/, use that.
    """
    # Direct file path
    p = Path(epub_arg)
    if p.exists() and p.suffix.lower() == ".epub":
        return p.resolve()

    # Check if it's a book slug (books/<slug>/source/*.epub)
    slug_dir = BASE_DIR / "books" / epub_arg / "source"
    if slug_dir.exists():
        epubs = list(slug_dir.glob("*.epub"))
        if epubs:
            return epubs[0]

    # Check in old-style source/ directory
    old_source = BASE_DIR / "source"
    if old_source.exists():
        epubs = list(old_source.glob("*.epub"))
        if epubs:
            return epubs[0]

    print(f"ERROR: Could not find EPUB file: {epub_arg}")
    print("\nUsage: python3 <script>.py <path-to-book.epub>")
    print("   or: python3 <script>.py <book-slug>")
    print("\nAvailable books:")
    books_dir = BASE_DIR / "books"
    if books_dir.exists():
        for d in sorted(books_dir.iterdir()):
            if d.is_dir():
                print(f"  {d.name}")
    sys.exit(1)
