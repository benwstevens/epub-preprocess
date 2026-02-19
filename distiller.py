#!/usr/bin/env python3
"""
Ebook Distiller — Hierarchical Summarization

Distills a full-length book down to ~10,000 words via two passes:
  Pass 1: Summarize each chapter with a proportional word target
  Pass 2: Coherence pass — merge and refine into a unified document

Usage:
    python3 distiller.py mybook.epub              # Run all stages
    python3 distiller.py mybook.epub --stage 3    # Re-run from stage 3
    python3 distiller.py mybook.epub --dry-run    # Stages 1-2 only
    python3 distiller.py mybook.epub --target 10000
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup

from shared import (
    BASE_DIR, setup_book_dir, find_source_file, detect_chapter_tag,
    split_chapters, filter_chapters, get_api_key, estimate_tokens,
    count_words, markdown_to_html, wrap_response_html,
    detect_book_metadata, resolve_epub_path,
)

INSTRUCTIONS_FILE = BASE_DIR / "distiller_instructions.txt"
COHERENCE_INSTRUCTIONS_FILE = BASE_DIR / "distiller_coherence_instructions.txt"


def stage3(chapter_files, output_dir, target_words, dry_run=False):
    """Pass 1: Summarize each chapter with proportional word targets."""
    print("\n" + "=" * 60)
    print("STAGE 3: Pass 1 — Distill Each Chapter")
    print("=" * 60)

    if dry_run:
        print("  --dry-run active: skipping API calls.")
        return []

    api_key = get_api_key()

    if not INSTRUCTIONS_FILE.exists():
        print(f"ERROR: {INSTRUCTIONS_FILE} not found.")
        sys.exit(1)
    instructions = INSTRUCTIONS_FILE.read_text(encoding="utf-8").strip()

    # Read chapters and compute word counts
    chapters = []
    total_words = 0
    for fp in sorted(chapter_files):
        content = fp.read_text(encoding="utf-8")
        wc = count_words(content)
        chapters.append((fp, content, wc))
        total_words += wc

    print(f"\nTotal source words: {total_words:,}")
    print(f"Target output: ~{target_words:,} words")
    print(f"Compression ratio: ~{total_words / target_words:.1f}:1")
    print(f"\nPer-chapter targets:")

    chapter_targets = []
    for fp, content, wc in chapters:
        proportion = wc / total_words
        chapter_target = max(int(target_words * proportion), 150)
        chapter_targets.append((fp, content, chapter_target))
        print(f"  {fp.name}: {wc:,} -> ~{chapter_target:,} words")

    # Cost estimate
    total_input_tokens = sum(estimate_tokens(c) for _, c, _ in chapter_targets)
    instruction_tokens = estimate_tokens(instructions)
    total_input_tokens += instruction_tokens * len(chapter_targets)
    estimated_total = (total_input_tokens / 1_000_000) * 3.0 + (target_words * 5 / 4 / 1_000_000) * 15.0

    print(f"\nEstimated cost (both passes): ~${estimated_total + 0.50:.2f}")
    print()

    confirm = input("Proceed with API calls? [Y/n] ").strip()
    if confirm.lower() == "n":
        print("Aborted by user.")
        sys.exit(0)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    model = "claude-sonnet-4-5-20250929"

    results = []
    failures = []

    for i, (fp, content, word_target) in enumerate(chapter_targets, start=1):
        print(
            f"Distilling chapter {i}/{len(chapter_targets)}: {fp.name} "
            f"(~{word_target} words)…",
            end=" ", flush=True,
        )

        distilled_name = fp.stem + "_distilled.html"
        distilled_path = output_dir / distilled_name
        if distilled_path.exists():
            print("already processed, skipping.")
            results.append((fp, distilled_path.read_text(encoding="utf-8")))
            continue

        user_message = (
            f"TARGET WORD COUNT: {word_target} words (stay within 10% of this).\n"
            f"CHAPTER TITLE: {fp.stem}\n\n"
            f"{content}"
        )

        max_retries = 5
        backoff = 2
        response_text = None

        for attempt in range(max_retries):
            try:
                message = client.messages.create(
                    model=model,
                    max_tokens=8192,
                    system=instructions,
                    messages=[{"role": "user", "content": user_message}],
                )
                response_text = message.content[0].text
                break
            except anthropic.RateLimitError:
                wait = backoff * (2 ** attempt)
                print(f"rate limited, waiting {wait}s…", end=" ", flush=True)
                time.sleep(wait)
            except anthropic.APIError as e:
                wait = backoff * (2 ** attempt)
                print(f"API error ({e}), retrying in {wait}s…", end=" ", flush=True)
                time.sleep(wait)

        if response_text is not None:
            actual_words = count_words(response_text)
            results.append((fp, response_text))
            print(f"done ({actual_words} words).")
        else:
            failures.append((fp, "Max retries exceeded"))
            print("FAILED.")

    if failures:
        print(f"\n{len(failures)} chapter(s) failed:")
        for fp, err in failures:
            print(f"  {fp.name}: {err}")

    return results


def stage4(results, chapter_files, output_dir):
    """Save distilled chapters."""
    print("\n" + "=" * 60)
    print("STAGE 4: Save Distilled Chapters")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    total_words = 0

    for chapter_path, response_text in results:
        distilled_name = chapter_path.stem + "_distilled.html"
        distilled_path = output_dir / distilled_name
        response_text = wrap_response_html(response_text, chapter_path, "distilled")
        distilled_path.write_text(response_text, encoding="utf-8")
        saved.append(distilled_path)
        wc = count_words(response_text)
        total_words += wc
        print(f"  Saved: {distilled_name} ({wc} words)")

    print(f"\nTotal distilled words: {total_words:,}")

    expected = {fp.stem + "_distilled.html" for fp in chapter_files}
    actual = {p.name for p in output_dir.iterdir() if p.suffix == ".html"}
    missing = expected - actual
    if missing:
        print(f"\nWARNING: {len(missing)} distilled file(s) missing.")
    else:
        print(f"All {len(expected)} distilled chapters present.")

    return saved


def stage5(paths, target_words):
    """Pass 2: Coherence pass + assemble EPUB."""
    print("\n" + "=" * 60)
    print("STAGE 5: Pass 2 — Coherence Edit + Final Assembly")
    print("=" * 60)

    from ebooklib import epub

    distilled_dir = paths["distilled_dir"]
    distilled_files = sorted(distilled_dir.glob("*.html"))
    if not distilled_files:
        print(f"ERROR: No distilled files found in {distilled_dir}/.")
        sys.exit(1)

    book_title, book_author = detect_book_metadata(paths["source_dir"])

    # Concatenate all distilled chapters
    combined_parts = []
    total_pre = 0
    for sf in distilled_files:
        content = sf.read_text(encoding="utf-8")
        chap_soup = BeautifulSoup(content, "lxml")
        body = chap_soup.find("body")
        body_html = "".join(str(c) for c in body.children) if body else content
        combined_parts.append(body_html)
        combined_parts.append("\n<hr/>\n")
        total_pre += count_words(body_html)

    combined_html = "\n".join(combined_parts)
    print(f"\nTotal words before coherence pass: {total_pre:,}")

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_title = re.sub(r"\s+", "_", re.sub(r"[^\w\s\-]", "", book_title).strip())
    coherence_cache = output_dir / f"{safe_title}_coherence_cache.html"

    if coherence_cache.exists():
        print("Found cached coherence pass output. Using it.")
        final_html = coherence_cache.read_text(encoding="utf-8")
    elif not COHERENCE_INSTRUCTIONS_FILE.exists():
        print("No coherence instructions found. Using raw chapter summaries.")
        final_html = combined_html
    else:
        print("Running coherence pass…")
        api_key = get_api_key()
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        coherence_instructions = COHERENCE_INSTRUCTIONS_FILE.read_text(encoding="utf-8").strip()

        user_message = (
            f"BOOK TITLE: {book_title}\n"
            f"AUTHOR: {book_author}\n"
            f"TOTAL CHAPTERS: {len(distilled_files)} — every one must appear in your output.\n\n"
            f"Here are the chapter-by-chapter summaries:\n\n"
            f"{combined_html}"
        )

        max_retries = 5
        backoff = 2
        final_html = None

        for attempt in range(max_retries):
            try:
                collected_text = []
                with client.messages.stream(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=32768,
                    system=coherence_instructions,
                    messages=[{"role": "user", "content": user_message}],
                ) as stream:
                    for text in stream.text_stream:
                        collected_text.append(text)
                        if sum(len(t) for t in collected_text) % 2000 < len(text):
                            print(".", end="", flush=True)
                final_html = "".join(collected_text)
                print()
                break
            except anthropic.RateLimitError:
                wait = backoff * (2 ** attempt)
                print(f"  Rate limited, waiting {wait}s…")
                time.sleep(wait)
            except anthropic.APIError as e:
                wait = backoff * (2 ** attempt)
                print(f"  API error ({e}), retrying in {wait}s…")
                time.sleep(wait)

        if final_html is None:
            print("  Coherence pass failed. Using raw chapter summaries.")
            final_html = combined_html
        else:
            final_html = markdown_to_html(final_html)
            print(f"  Coherence pass complete: {count_words(final_html):,} words")
            coherence_cache.write_text(final_html, encoding="utf-8")

    # Build EPUB
    book = epub.EpubBook()
    book.set_identifier("ebook-distilled-" + re.sub(r"\W+", "-", book_title.lower()))
    book.set_title(book_title + " (Distilled)")
    book.set_language("en")
    book.add_author(book_author)

    css_content = """
body { font-family: Georgia, "Times New Roman", serif; line-height: 1.6; margin: 1em; color: #222; }
h1, h2, h3, h4 { margin-top: 1.5em; margin-bottom: 0.5em; line-height: 1.2; }
p { margin-bottom: 0.8em; text-align: justify; }
blockquote { margin: 1em 2em; font-style: italic; color: #555; }
.thesis { background: #f5f5f0; border-left: 3px solid #888; padding: 0.8em 1em; margin: 1em 0 1.5em 0; font-style: italic; }
hr { border: none; border-top: 1px solid #ccc; margin: 2em 0; }
"""
    css = epub.EpubItem(uid="style", file_name="style/default.css",
                        media_type="text/css", content=css_content.encode("utf-8"))
    book.add_item(css)

    final_soup = BeautifulSoup(final_html, "lxml")
    h2_tags = final_soup.find_all("h2")

    epub_chapters = []
    toc = []

    if h2_tags:
        for i, h2 in enumerate(h2_tags, start=1):
            chap_title = h2.get_text(strip=True)
            parts = [str(h2)]
            for sibling in h2.find_next_siblings():
                if sibling.name == "h2":
                    break
                if sibling.name == "hr":
                    continue
                parts.append(str(sibling))
            body_html = "\n".join(parts)

            chap = epub.EpubHtml(title=chap_title, file_name=f"chapter_{i:02d}.xhtml", lang="en")
            chap.content = (
                f'<html><head><link rel="stylesheet" href="style/default.css" '
                f'type="text/css"/></head><body>{body_html}</body></html>'
            ).encode("utf-8")
            chap.add_item(css)
            book.add_item(chap)
            epub_chapters.append(chap)
            toc.append(epub.Link(f"chapter_{i:02d}.xhtml", chap_title, f"ch{i}"))
    else:
        chap = epub.EpubHtml(title=book_title, file_name="content.xhtml", lang="en")
        chap.content = (
            f'<html><head><link rel="stylesheet" href="style/default.css" '
            f'type="text/css"/></head><body>{final_html}</body></html>'
        ).encode("utf-8")
        chap.add_item(css)
        book.add_item(chap)
        epub_chapters = [chap]
        toc = [epub.Link("content.xhtml", book_title, "ch1")]

    book.toc = toc
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + epub_chapters

    epub_path = output_dir / f"{safe_title}_distilled.epub"
    epub.write_epub(str(epub_path), book)
    print(f"\nEPUB saved: {epub_path}")

    html_doc = (
        f"<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{book_title} — Distilled</title>\n<style>{css_content}</style>\n"
        f"</head>\n<body>\n<h1>{book_title} (Distilled)</h1>\n"
        f"{final_html}\n</body>\n</html>"
    )
    html_path = output_dir / f"{safe_title}_distilled.html"
    html_path.write_text(html_doc, encoding="utf-8")
    print(f"HTML saved: {html_path}")
    print(f"\nFinal word count: {count_words(final_html):,}")

    return epub_path


def main():
    parser = argparse.ArgumentParser(description="Ebook Distiller — Hierarchical Summarization")
    parser.add_argument("book", help="Path to EPUB/HTML file or book slug")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5], default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target", type=int, default=10000, help="Target word count (default: 10000)")
    args = parser.parse_args()

    epub_path = resolve_epub_path(args.book)
    paths = setup_book_dir(epub_path)

    print("=" * 60)
    print("  Ebook Distiller — Hierarchical Summarization")
    print("=" * 60)
    print(f"  Book: {epub_path.stem}")
    print(f"  Working dir: {paths['book_dir'].relative_to(BASE_DIR)}")
    print(f"  Target: {args.target:,} words")
    if args.dry_run:
        print("  Mode: DRY RUN")
    print(f"  Starting from stage: {args.stage}")

    html_path = find_source_file(paths)
    chapter_tag = None
    chapter_files = None
    results = None

    if args.stage <= 1:
        chapter_tag = detect_chapter_tag(html_path)

    if args.stage <= 2:
        if chapter_tag is None:
            chapter_tag = detect_chapter_tag(html_path)
        chapter_files = split_chapters(html_path, chapter_tag, paths["chapters_dir"])
    else:
        chapter_files = sorted(paths["chapters_dir"].glob("*.html"))
        if not chapter_files:
            print("ERROR: No chapter files found. Run from stage 1 first.")
            sys.exit(1)

    chapter_files = filter_chapters(chapter_files, paths["selection_file"], "distilling")

    if args.dry_run:
        total_words = sum(count_words(fp.read_text(encoding="utf-8")) for fp in chapter_files)
        print(f"\nTotal source words: {total_words:,}")
        print(f"Target: {args.target:,} words ({total_words / max(args.target, 1):.1f}:1)")
        print("\n" + "=" * 60)
        print("DRY RUN COMPLETE")
        print("=" * 60)
        return

    if args.stage <= 3:
        results = stage3(chapter_files, paths["distilled_dir"], args.target)

    if args.stage <= 4:
        if results is None:
            results = []
        if results:
            stage4(results, chapter_files, paths["distilled_dir"])

    if args.stage <= 5:
        stage5(paths, args.target)

    print("\n" + "=" * 60)
    print("  Distiller pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
