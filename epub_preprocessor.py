"""
EPUB Preprocessor — Structural audit, interactive selection, and normalization.

Analyzes an EPUB file's internal structure (TOC, heading tags, file boundaries),
presents the hierarchy to the user for review, and outputs a normalized HTML file
with unique, descriptive headings that the existing pipeline (shared.py / pipeline.py)
can consume directly.

Usage:
    python epub_preprocessor.py book.epub
    python epub_preprocessor.py book.epub --output source/
    python epub_preprocessor.py book.epub --auto --exclude-keywords "contents,index"
    python epub_preprocessor.py book.epub --split-level section
    python epub_preprocessor.py book.epub --audit-only
    python epub_preprocessor.py book.epub --manifest previous_manifest.json
"""

import argparse
import json
import re
import sys
import warnings
from collections import Counter, OrderedDict
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning
from ebooklib import epub as ep

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
# ebooklib emits many UserWarnings about unrecognized EPUB features
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# Data structures
# ============================================================================

class TOCEntry:
    """A node in the table-of-contents tree."""

    def __init__(self, title: str, href: str, depth: int = 0):
        self.title = title.strip() if title else ""
        self.href = href  # may contain #fragment
        self.base_href = href.split("#")[0] if href else ""
        self.fragment = href.split("#")[1] if href and "#" in href else None
        self.depth = depth
        self.children: list["TOCEntry"] = []

    def __repr__(self):
        indent = "  " * self.depth
        return f"{indent}{self.title} -> {self.href} (depth {self.depth})"

    def flatten(self) -> list["TOCEntry"]:
        """Return a flat list of this entry plus all descendants."""
        result = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result


class ChapterCandidate:
    """A potential chapter identified during structural audit."""

    def __init__(
        self,
        index: int,
        label: str,
        hierarchy_parts: list[str],
        heading_tag: str,
        html_content: str,
        source_file: str = "",
        toc_entry: TOCEntry | None = None,
    ):
        self.index = index
        self.label = label
        self.hierarchy_parts = hierarchy_parts  # e.g. ["Part I", "Section II", "Chapter III"]
        self.heading_tag = heading_tag
        self.html_content = html_content
        self.source_file = source_file
        self.toc_entry = toc_entry
        self._word_count: int | None = None

    @property
    def word_count(self) -> int:
        if self._word_count is None:
            clean = re.sub(r"<[^>]+>", " ", self.html_content)
            self._word_count = len(clean.split())
        return self._word_count

    @property
    def full_label(self) -> str:
        """Build the hierarchical label, e.g. 'Part I, Sec II, Ch III — Of Sympathy'."""
        if self.hierarchy_parts:
            return ", ".join(self.hierarchy_parts)
        return self.label


# ============================================================================
# Phase 1: Structural Audit
# ============================================================================

def read_epub(epub_path: Path) -> ep.EpubBook:
    """Read and return an EpubBook object."""
    print(f"Reading EPUB: {epub_path.name}")
    book = ep.read_epub(str(epub_path))
    title = book.get_metadata("DC", "title")
    title = title[0][0] if title else epub_path.stem
    author = book.get_metadata("DC", "creator")
    author = author[0][0] if author else ""
    if title:
        print(f"  Title:  {title}")
    if author:
        print(f"  Author: {author}")
    return book


def extract_toc_tree(book: ep.EpubBook) -> list[TOCEntry]:
    """Extract the table of contents as a tree of TOCEntry objects.

    Handles both EPUB2 NCX navMap and EPUB3 nav documents, as well as
    ebooklib's representation of them (Link objects and (Section, [children]) tuples).
    """
    def _process_entry(entry, depth: int = 0) -> TOCEntry | None:
        if hasattr(entry, "href") and hasattr(entry, "title"):
            # Simple Link entry
            node = TOCEntry(entry.title or "", entry.href or "", depth)
            return node
        elif isinstance(entry, tuple) and len(entry) == 2:
            # (Section, [children]) — a group with sub-entries
            section, children = entry
            if hasattr(section, "href") and hasattr(section, "title"):
                node = TOCEntry(section.title or "", section.href or "", depth)
            elif hasattr(section, "title"):
                # Section without href (just a grouping label)
                node = TOCEntry(section.title or "", "", depth)
            else:
                return None
            if isinstance(children, list):
                for child in children:
                    child_node = _process_entry(child, depth + 1)
                    if child_node:
                        node.children.append(child_node)
            return node
        return None

    roots: list[TOCEntry] = []
    for entry in book.toc:
        node = _process_entry(entry, depth=0)
        if node:
            roots.append(node)

    return roots


def flatten_toc(roots: list[TOCEntry]) -> list[TOCEntry]:
    """Flatten the TOC tree into a depth-ordered list."""
    flat: list[TOCEntry] = []
    for root in roots:
        flat.extend(root.flatten())
    return flat


def get_spine_items(book: ep.EpubBook) -> list[tuple[str, str]]:
    """Return spine items as (id, href) pairs in reading order."""
    items = []
    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item:
            items.append((item_id, item.get_name()))
    return items


def classify_spine_items(
    spine_items: list[tuple[str, str]],
    flat_toc: list[TOCEntry],
) -> dict[str, str]:
    """Classify each spine item as 'front', 'content', or 'back' matter.

    Uses the TOC to find which files are referenced by Part/Section/Chapter
    entries. Everything before the first such file is front matter; everything
    after the last is back matter.
    """
    content_pattern = re.compile(
        r"^(part|section|chapter|book|act)\b", re.IGNORECASE
    )

    # Collect hrefs from TOC entries whose titles look like content
    content_hrefs: set[str] = set()
    for entry in flat_toc:
        if not entry.base_href:
            continue
        if content_pattern.search(entry.title):
            content_hrefs.add(entry.base_href)
        # If a parent is content-like, include its children
        for child in entry.flatten()[1:]:
            if child.base_href and content_pattern.search(entry.title):
                content_hrefs.add(child.base_href)

    # Fallback: if no content-like titles, use all TOC hrefs
    if not content_hrefs:
        content_hrefs = {e.base_href for e in flat_toc if e.base_href}

    # Normalize to filenames for matching
    content_fnames = {h.split("/")[-1] for h in content_hrefs}

    spine_hrefs = [href for _id, href in spine_items]
    first_content = None
    last_content = None
    for i, href in enumerate(spine_hrefs):
        if href.split("/")[-1] in content_fnames:
            if first_content is None:
                first_content = i
            last_content = i

    result: dict[str, str] = {}
    if first_content is None:
        for href in spine_hrefs:
            result[href] = "content"
        return result

    for i, href in enumerate(spine_hrefs):
        if i < first_content:
            result[href] = "front"
        elif i > last_content:
            result[href] = "back"
        else:
            result[href] = "content"

    return result


def parse_content_files(book: ep.EpubBook) -> OrderedDict:
    """Parse all content documents and return {href: BeautifulSoup} in spine order.

    Returns an OrderedDict keyed by item href (as used in spine), with values
    being (soup, raw_html) tuples.
    """
    spine_order = []
    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item:
            spine_order.append(item)

    docs = OrderedDict()
    for item in spine_order:
        content = item.get_content()
        soup = BeautifulSoup(content, "lxml")
        body = soup.find("body")
        if body:
            body_html = "".join(str(c) for c in body.children)
        else:
            body_html = str(soup)
        docs[item.get_name()] = (soup, body_html)

    return docs


def scan_headings(
    docs: OrderedDict,
    skip_hrefs: set[str] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Scan all heading tags across content files.

    Returns {tag_name: [(text, source_href), ...]}.
    skip_hrefs: files to exclude (e.g. front/back matter, TOC page).
    """
    headings: dict[str, list[tuple[str, str]]] = {}
    for href, (soup, _body_html) in docs.items():
        if skip_hrefs and href in skip_hrefs:
            continue
        for level in range(1, 7):
            tag = f"h{level}"
            for el in soup.find_all(tag):
                text = el.get_text(strip=True)
                if text:
                    headings.setdefault(tag, []).append((text, href))
    return headings


def detect_non_standard_headings(
    docs: OrderedDict,
    skip_hrefs: set[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Detect potential non-standard heading markup (styled <p> or <div> elements).

    Returns [(css_selector, text, source_href), ...] for likely chapter titles
    that aren't using semantic heading tags.
    """
    chapter_pattern = re.compile(
        r"^(chapter|part|section|book|act|prologue|epilogue|introduction|"
        r"conclusion|foreword|afterword|preface)\b",
        re.IGNORECASE,
    )
    results = []
    for href, (soup, _body_html) in docs.items():
        if skip_hrefs and href in skip_hrefs:
            continue
        for p in soup.find_all(["p", "div", "span"]):
            cls = p.get("class", [])
            text = p.get_text(strip=True)
            if not text or len(text) > 120:
                continue
            # Check if class name suggests it's a heading
            cls_str = " ".join(cls).lower() if cls else ""
            is_heading_class = any(
                kw in cls_str
                for kw in ["title", "heading", "chapter", "part", "section", "ct", "head"]
            )
            is_heading_text = bool(chapter_pattern.match(text))
            if is_heading_class or is_heading_text:
                selector = p.name
                if cls:
                    selector += "." + ".".join(cls)
                results.append((selector, text, href))
    return results


def promote_styled_headings(
    docs: OrderedDict,
    heading_map: dict[str, str] | None = None,
    skip_hrefs: set[str] | None = None,
) -> int:
    """Promote div/p/span elements with known CSS classes to semantic heading tags.

    Args:
        docs: OrderedDict from ``parse_content_files()`` — mutated in place.
        heading_map: Optional custom mapping of CSS class → heading tag.
        skip_hrefs: Files to skip (e.g. TOC page, front/back matter).

    Returns:
        The total number of elements promoted across all documents.
    """
    # ------------------------------------------------------------------
    # Default class → heading-tag mapping
    # ------------------------------------------------------------------
    default_map: dict[str, str] = {
        # InDesign / EPUB-converter class conventions.
        "cpt": "h1",   # Chapter Part Title  (e.g. "Part One")
        "cct": "h2",   # Chapter Category/Group Title (e.g. "Section I")
        "ct":  "h3",   # Compact title (e.g. "CHAPTER I." or "SECTION I.")
        "cst": "h4",   # Compact subtitle (e.g. "Of Sympathy.")
        # Broader patterns
        "part-title":    "h1",
        "parttitle":     "h1",
        "book-title":    "h1",
        "booktitle":     "h1",
        "chapter-title": "h2",
        "chaptertitle":  "h2",
        "section-title": "h3",
        "sectiontitle":  "h3",
    }

    active_map = heading_map if heading_map is not None else default_map

    if not active_map:
        return 0

    # ------------------------------------------------------------------
    # 1. Scan: figure out which promotions would fire and what heading
    #    levels already exist in the documents.
    # ------------------------------------------------------------------
    promotion_levels: set[int] = set()   # levels the promotions will occupy
    existing_levels: set[int] = set()    # levels already present as h-tags

    for href, (soup, _body_html) in docs.items():
        if skip_hrefs and href in skip_hrefs:
            continue
        # Check existing heading tags
        for level in range(1, 7):
            if soup.find(f"h{level}"):
                existing_levels.add(level)

        # Check which promotions would fire
        for el in soup.find_all(["div", "p", "span"]):
            classes = el.get("class", [])
            if not classes:
                continue
            for cls in classes:
                cls_lower = cls.lower()
                if cls_lower in active_map:
                    text = el.get_text(strip=True)
                    if text and len(text) <= 200:
                        tag = active_map[cls_lower]
                        promotion_levels.add(int(tag[1]))
                    break

    if not promotion_levels:
        print("\n--- Heading Promotion: no promotable elements found ---")
        return 0

    # ------------------------------------------------------------------
    # 2. Compute demotion shift if promotion targets collide with
    #    existing heading levels.
    # ------------------------------------------------------------------
    overlap = promotion_levels & existing_levels
    shift = 0
    if overlap:
        # Shift existing headings down by enough to clear all promotion levels.
        # E.g. promotions use {1,2}, existing has {1,2} → shift existing by 2.
        max_promo = max(promotion_levels)
        min_existing = min(existing_levels)
        if min_existing <= max_promo:
            shift = max_promo - min_existing + 1
            # Clamp: can't push past h6
            max_existing = max(existing_levels)
            if max_existing + shift > 6:
                shift = 6 - max_existing
            if shift < 1:
                shift = 1

    demotion_summary: dict[str, str] = {}  # "h1 -> h3"

    if shift > 0:
        # Demote existing headings in every document (high levels first to
        # avoid collisions within the same pass).
        for href, (soup, _body_html) in list(docs.items()):
            if skip_hrefs and href in skip_hrefs:
                continue
            changed = False
            for old_level in sorted(existing_levels, reverse=True):
                new_level = min(old_level + shift, 6)
                old_tag = f"h{old_level}"
                new_tag = f"h{new_level}"
                for el in soup.find_all(old_tag):
                    el.name = new_tag
                    changed = True
                if old_tag != new_tag:
                    demotion_summary[old_tag] = new_tag
            if changed:
                body = soup.find("body")
                if body:
                    new_body_html = "".join(str(c) for c in body.children)
                else:
                    new_body_html = str(soup)
                docs[href] = (soup, new_body_html)

    # ------------------------------------------------------------------
    # 3. Promote: rewrite div/p/span elements to heading tags.
    # ------------------------------------------------------------------
    total_promoted = 0
    promotion_summary: dict[str, list[str]] = {}

    for href, (soup, _body_html) in list(docs.items()):
        if skip_hrefs and href in skip_hrefs:
            continue
        promoted_in_file = 0

        for el in soup.find_all(["div", "p", "span"]):
            classes = el.get("class", [])
            if not classes:
                continue

            target_tag = None
            matched_class = None

            for cls in classes:
                cls_lower = cls.lower()
                if cls_lower in active_map:
                    target_tag = active_map[cls_lower]
                    matched_class = cls_lower
                    break

            if target_tag is None:
                continue

            text = el.get_text(strip=True)
            if not text or len(text) > 200:
                continue

            original_tag = el.name
            el.name = target_tag

            promoted_in_file += 1
            key = f"{original_tag}.{matched_class} -> <{target_tag}>"
            promotion_summary.setdefault(key, []).append(text)

        if promoted_in_file:
            body = soup.find("body")
            if body:
                new_body_html = "".join(str(c) for c in body.children)
            else:
                new_body_html = str(soup)
            docs[href] = (soup, new_body_html)
            total_promoted += promoted_in_file

    # ------------------------------------------------------------------
    # 4. Report
    # ------------------------------------------------------------------
    if total_promoted or demotion_summary:
        print(f"\n--- Heading Promotion ({total_promoted} elements promoted) ---")
        if demotion_summary:
            print(f"  Demoted existing headings to make room:")
            for old_tag, new_tag in sorted(demotion_summary.items()):
                print(f"    <{old_tag}> -> <{new_tag}>")
        for key in sorted(promotion_summary.keys()):
            texts = promotion_summary[key]
            print(f"  {key}: {len(texts)} element(s)")
            for t in texts[:5]:
                print(f"       . {t[:80]}")
            if len(texts) > 5:
                print(f"       ... and {len(texts) - 5} more")
    else:
        print("\n--- Heading Promotion: no promotable elements found ---")

    return total_promoted


def cross_reference_toc_headings(
    toc_entries: list[TOCEntry],
    docs: OrderedDict,
) -> list[tuple[TOCEntry, str | None]]:
    """Cross-reference TOC entries against actual heading tags in the HTML.

    Returns [(toc_entry, matched_tag_or_None), ...].
    """
    results = []
    for entry in toc_entries:
        if not entry.base_href:
            results.append((entry, None))
            continue

        # Find the content document
        doc_key = None
        for href in docs:
            # Match by full path or by filename
            if href == entry.base_href or href.split("/")[-1] == entry.base_href.split("/")[-1]:
                doc_key = href
                break

        if doc_key is None:
            results.append((entry, None))
            continue

        soup, _body_html = docs[doc_key]

        # Look for a heading tag that matches the TOC title
        matched_tag = None
        for level in range(1, 7):
            tag = f"h{level}"
            for el in soup.find_all(tag):
                el_text = el.get_text(strip=True)
                # Match by exact text, or if the entry title is contained in heading
                if _text_match(entry.title, el_text):
                    matched_tag = tag
                    break
                # Also check if there's a fragment anchor match
                if entry.fragment:
                    anchor = el.find("a", id=entry.fragment) or el.get("id") == entry.fragment
                    if anchor:
                        matched_tag = tag
                        break
            if matched_tag:
                break

        # If no heading matched, check by fragment ID in any element
        if not matched_tag and entry.fragment:
            target = soup.find(id=entry.fragment) or soup.find("a", attrs={"name": entry.fragment})
            if target:
                # Walk up to see if it's inside a heading
                parent = target.parent
                while parent:
                    if parent.name and re.match(r"h[1-6]", parent.name):
                        matched_tag = parent.name
                        break
                    parent = parent.parent

        results.append((entry, matched_tag))

    return results


def _text_match(toc_title: str, heading_text: str) -> bool:
    """Check if a TOC title matches a heading text (fuzzy)."""
    if not toc_title or not heading_text:
        return False
    # Normalize: strip, collapse whitespace, lowercase
    a = re.sub(r"\s+", " ", toc_title.strip().lower())
    b = re.sub(r"\s+", " ", heading_text.strip().lower())
    if a == b:
        return True
    # Check containment (TOC often truncates or heading has extra numbering)
    if a in b or b in a:
        return True
    # Check if ignoring leading numbers/roman numerals they match
    strip_num = re.compile(r"^[\dIVXLCDMivxlcdm]+[\.\)\s:]+\s*", re.IGNORECASE)
    a_stripped = strip_num.sub("", a)
    b_stripped = strip_num.sub("", b)
    if a_stripped and b_stripped and (a_stripped == b_stripped or a_stripped in b_stripped or b_stripped in a_stripped):
        return True
    return False


def identify_hierarchy(
    heading_data: dict[str, list[tuple[str, str]]],
    toc_tree: list[TOCEntry],
) -> dict[str, str]:
    """Determine what each heading level represents.

    Returns {tag: role} where role is 'part', 'section', 'chapter', etc.
    """
    # Strategy 1: Match ALL TOC entries (including those without hrefs) against
    # heading text to map TOC depth -> heading tag.
    flat_toc = flatten_toc(toc_tree)

    depth_to_tags: dict[int, Counter] = {}
    for entry in flat_toc:
        if not entry.title:
            continue
        for tag, entries_list in heading_data.items():
            for text, _href in entries_list:
                if _text_match(entry.title, text):
                    depth_to_tags.setdefault(entry.depth, Counter())[tag] += 1
                    break

    hierarchy: dict[str, str] = {}
    role_names = ["part", "section", "chapter", "subsection"]

    if depth_to_tags:
        # Assign roles based on TOC depth mapping
        for depth in sorted(depth_to_tags.keys()):
            most_common_tag = depth_to_tags[depth].most_common(1)[0][0]
            if most_common_tag not in hierarchy:
                role_idx = min(depth, len(role_names) - 1)
                hierarchy[most_common_tag] = role_names[role_idx]

    # Strategy 2: Use heading text patterns to supplement (or, if Strategy 1
    # found nothing, to bootstrap) the hierarchy.  For each heading tag not
    # yet in the hierarchy, check whether a significant fraction of its
    # entries start with "Part", "Section", "Chapter", etc.
    pattern_roles = [
        (re.compile(r"^part\b", re.IGNORECASE), "part"),
        (re.compile(r"^section\b", re.IGNORECASE), "section"),
        (re.compile(r"^(chapter|ch\.?\s)\b", re.IGNORECASE), "chapter"),
    ]
    tag_role_votes: dict[str, Counter] = {}
    for tag, entries_list in heading_data.items():
        if tag in hierarchy:
            continue  # already assigned by Strategy 1
        for text, _href in entries_list:
            for pattern, role in pattern_roles:
                if pattern.search(text):
                    tag_role_votes.setdefault(tag, Counter())[role] += 1
                    break

    for tag in sorted(tag_role_votes.keys(), key=lambda t: int(t[1])):
        best_role = tag_role_votes[tag].most_common(1)[0][0]
        if best_role not in hierarchy.values():
            hierarchy[tag] = best_role

    # Strategy 3: Fallback — assign by heading level order
    if not hierarchy:
        available_tags = sorted(heading_data.keys(), key=lambda t: int(t[1]))
        for i, tag in enumerate(available_tags):
            role_idx = min(i, len(role_names) - 1)
            hierarchy[tag] = role_names[role_idx]

    return hierarchy


def run_structural_audit(
    book: ep.EpubBook,
    epub_path: Path,
    heading_map: dict[str, str] | None = None,
) -> dict:
    """Run Phase 1: Structural Audit. Returns audit results dict."""
    print("\n" + "=" * 65)
    print("PHASE 1: Structural Audit")
    print("=" * 65)

    # 1. Extract and display TOC
    toc_tree = extract_toc_tree(book)
    flat_toc = flatten_toc(toc_tree)

    print(f"\n--- Table of Contents ({len(flat_toc)} entries) ---")
    if flat_toc:
        for entry in flat_toc:
            indent = "  " * entry.depth
            print(f"  {indent}{entry.title}")
            if not entry.base_href:
                print(f"  {indent}  (no href — grouping label only)")
    else:
        print("  WARNING: No TOC entries found in this EPUB.")
        print("  Will fall back to heading-tag analysis.")

    # 2. Count spine items and classify front/back matter
    spine_items = get_spine_items(book)
    classification = classify_spine_items(spine_items, flat_toc)
    front_count = sum(1 for v in classification.values() if v == "front")
    back_count = sum(1 for v in classification.values() if v == "back")
    content_count = sum(1 for v in classification.values() if v == "content")
    print(f"\n--- Spine: {len(spine_items)} files ---")
    print(f"  Front matter: {front_count}, Content: {content_count}, Back matter: {back_count}")

    # 3. Parse content files
    docs = parse_content_files(book)

    # Build skip set: front/back matter + TOC-like pages not targeted by TOC
    toc_target_fnames = {e.base_href.split("/")[-1] for e in flat_toc if e.base_href}
    skip_hrefs: set[str] = set()
    for href in docs:
        fname = href.split("/")[-1]
        if classification.get(href) in ("front", "back"):
            skip_hrefs.add(href)
        elif "toc" in fname.lower() and fname not in toc_target_fnames:
            skip_hrefs.add(href)

    # 3b. Promote styled divs/paragraphs to semantic heading tags
    promote_styled_headings(docs, heading_map=heading_map, skip_hrefs=skip_hrefs)

    # 4. Scan headings (content files only)
    heading_data = scan_headings(docs, skip_hrefs=skip_hrefs)
    print(f"\n--- Heading Tag Survey (content files only) ---")
    if heading_data:
        for tag in sorted(heading_data.keys(), key=lambda t: int(t[1])):
            entries = heading_data[tag]
            print(f"  <{tag}>: {len(entries)} occurrence(s)")
            for text, href in entries[:5]:
                print(f"       . {text[:80]}")
            if len(entries) > 5:
                print(f"       ... and {len(entries) - 5} more")
    else:
        print("  WARNING: No heading tags (h1-h6) found.")

    # 5. Detect non-standard headings
    non_standard = detect_non_standard_headings(docs, skip_hrefs=skip_hrefs)
    if non_standard:
        print(f"\n--- Non-Standard Heading Markup ({len(non_standard)} found) ---")
        for selector, text, href in non_standard[:10]:
            print(f"  <{selector}>: \"{text[:60]}\"")
        if len(non_standard) > 10:
            print(f"  ... and {len(non_standard) - 10} more")

    # 6. Cross-reference TOC vs headings
    toc_xref = cross_reference_toc_headings(flat_toc, docs)
    unmatched = [(entry, tag) for entry, tag in toc_xref if tag is None and entry.base_href]
    if unmatched:
        print(f"\n--- TOC/Heading Mismatches ({len(unmatched)}) ---")
        for entry, _ in unmatched[:10]:
            print(f"  WARNING: \"{entry.title}\" — no matching heading tag in HTML")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")

    # 7. Identify hierarchy
    hierarchy = identify_hierarchy(heading_data, toc_tree)
    print(f"\n--- Detected Hierarchy ---")
    if hierarchy:
        for tag in sorted(hierarchy.keys(), key=lambda t: int(t[1])):
            role = hierarchy[tag]
            count = len(heading_data.get(tag, []))
            print(f"  Level ({role.title()}s): <{tag}> — {count} entries")
    else:
        print("  WARNING: Could not determine hierarchy.")

    # Determine deepest split level
    split_tag = None
    if hierarchy:
        # The tag with the most granular role (chapter > section > part)
        role_priority = {"chapter": 0, "subsection": 1, "section": 2, "part": 3}
        split_tag = min(
            hierarchy.keys(),
            key=lambda t: role_priority.get(hierarchy[t], 99),
        )
        count = len(heading_data.get(split_tag, []))
        print(f"\n  Deepest split level: <{split_tag}> ({count} chapters)")

    return {
        "toc_tree": toc_tree,
        "flat_toc": flat_toc,
        "spine_items": spine_items,
        "classification": classification,
        "skip_hrefs": skip_hrefs,
        "docs": docs,
        "heading_data": heading_data,
        "non_standard": non_standard,
        "toc_xref": toc_xref,
        "hierarchy": hierarchy,
        "split_tag": split_tag,
        "epub_path": epub_path,
    }


# ============================================================================
# Build chapter candidates from the EPUB structure
# ============================================================================

def _get_doc_for_href(docs: OrderedDict, target_href: str):
    """Find a document in docs matching the given href."""
    if target_href in docs:
        return target_href, docs[target_href]
    target_fname = target_href.split("/")[-1]
    for href in docs:
        if href.split("/")[-1] == target_fname:
            return href, docs[href]
    return None, None


def _guess_role(text: str) -> str:
    """Guess whether a heading is a part, section, or chapter from its text."""
    if re.match(r"^part\b", text, re.IGNORECASE):
        return "part"
    if re.match(r"^section\b", text, re.IGNORECASE):
        return "section"
    if re.match(r"^(chapter|ch\.?\s)\b", text, re.IGNORECASE):
        return "chapter"
    return ""


def _abbreviate_label(text: str, role: str) -> str:
    """Abbreviate hierarchy labels for display. E.g., 'Section II' -> 'Sec II'.
    Also strips subtitle after dash for Part/Section labels."""
    abbreviations = {
        "part": (r"(?i)^part\b", "Part"),
        "section": (r"(?i)^section\b", "Sec"),
        "chapter": (r"(?i)^chapter\b", "Ch"),
    }
    if role in abbreviations:
        pattern, replacement = abbreviations[role]
        if re.match(pattern, text):
            text = re.sub(pattern, replacement, text, count=1)
    # Strip " - SUBTITLE" from Part/Section labels for brevity
    stripped = re.sub(r"\s*[-—]+\s+.*$", "", text)
    if stripped != text and role in ("part", "section"):
        return stripped
    return text


def _find_sub_chapter_tag(soup) -> str | None:
    """Detect the heading tag used for sub-chapters within a file.

    Looks for h-tags where text starts with 'CHAPTER' or 'INTRODUCTION',
    suggesting sub-chapter divisions. Requires at least 2 matches.
    """
    chapter_re = re.compile(r"^(chapter|introduction|conclusion)\b", re.IGNORECASE)
    body = soup.find("body")
    if not body:
        return None
    tag_counts: Counter = Counter()
    for level in range(1, 7):
        tag = f"h{level}"
        for el in body.find_all(tag):
            text = el.get_text(strip=True)
            if text and chapter_re.match(text):
                tag_counts[tag] += 1
    if not tag_counts:
        return None
    best_tag = max(tag_counts.keys(), key=lambda t: (tag_counts[t], -int(t[1])))
    if tag_counts[best_tag] >= 2:
        return best_tag
    return None


def _get_subtitle_after_heading(soup, heading_tag: str, heading_text: str) -> str:
    """Find the subtitle element that immediately follows a chapter heading.

    E.g. <h5>CHAPTER I.</h5> <h6>Of Sympathy.</h6> -> returns "Of Sympathy."
    """
    h_level = int(heading_tag[1])
    subtitle_tag = f"h{h_level + 1}"
    for el in soup.find_all(heading_tag):
        if el.get_text(strip=True) == heading_text:
            sibling = el.find_next_sibling()
            if sibling and sibling.name == subtitle_tag:
                return sibling.get_text(strip=True)
            break
    return ""


def _split_file_at_headings(
    soup,
    body_html: str,
    heading_tag: str,
) -> list[tuple[str, str, str]]:
    """Split a file's body content at heading-tag boundaries.

    Returns list of (tag_name, heading_text, chunk_html).
    First chunk may have empty heading_text if there's pre-heading content.
    """
    body = soup.find("body")
    if not body:
        return [("", "", body_html)]

    chunks: list[tuple[str, str, str]] = []
    current_parts: list[str] = []
    current_tag = ""
    current_text = ""

    for child in body.children:
        if isinstance(child, NavigableString):
            current_parts.append(str(child))
            continue
        if getattr(child, "name", None) == heading_tag:
            # Save previous chunk
            if current_parts:
                chunks.append((current_tag, current_text, "".join(current_parts)))
            current_tag = heading_tag
            current_text = child.get_text(strip=True)
            current_parts = [str(child)]
        else:
            current_parts.append(str(child))

    if current_parts:
        chunks.append((current_tag, current_text, "".join(current_parts)))

    return chunks


def build_chapter_candidates(
    audit: dict,
    split_level: str | None = None,
) -> list[ChapterCandidate]:
    """Build the list of chapter candidates using a TOC-driven approach.

    Instead of concatenating all HTML and splitting on a single heading tag,
    this processes each TOC entry's file individually:
    1. Walk the TOC tree for structure.
    2. Map each entry to its content file.
    3. For multi-chapter files, split at internal heading boundaries.
    4. Build labels from TOC hierarchy context.
    """
    toc_tree = audit["toc_tree"]
    flat_toc = audit["flat_toc"]
    docs = audit["docs"]
    classification = audit.get("classification", {})

    # Determine target TOC depth
    toc_depths = {e.depth for e in flat_toc if e.base_href}
    max_toc_depth = max(toc_depths) if toc_depths else 0

    do_sub_split = True
    if split_level == "part":
        target_depth = 0
        do_sub_split = False
    elif split_level == "section":
        target_depth = min(1, max_toc_depth)
        do_sub_split = False
    else:
        target_depth = max_toc_depth

    # Build ancestor map: id(entry) -> [ancestor title strings]
    entry_ancestors: dict[int, list[str]] = {}

    def _walk_toc(entries: list[TOCEntry], ancestors: list[str]):
        for entry in entries:
            entry_ancestors[id(entry)] = list(ancestors)
            if entry.children:
                _walk_toc(entry.children, ancestors + [entry.title])

    _walk_toc(toc_tree, [])

    # Get leaf entries at target depth
    leaf_entries: list[TOCEntry] = []
    for e in flat_toc:
        if e.depth == target_depth and e.base_href:
            leaf_entries.append(e)
    # Also include shallower entries that have no deeper children
    for e in flat_toc:
        if e.depth < target_depth and e.base_href:
            has_deeper = any(
                c.depth >= target_depth and c.base_href
                for c in e.flatten()[1:]
            )
            if not has_deeper:
                leaf_entries.append(e)

    # Sort by TOC order
    flat_order = {id(e): i for i, e in enumerate(flat_toc)}
    leaf_entries.sort(key=lambda e: flat_order.get(id(e), 999))

    # Filter out front/back matter entries
    back_matter_re = re.compile(
        r"^(teaser|biographical|textual|notes|index|copyright|title\s*page|"
        r"colophon|acknowledgment|about\s*the|also\s*by|further\s*reading|"
        r"bibliography|glossary|next)",
        re.IGNORECASE,
    )
    content_entries = []
    for entry in leaf_entries:
        if back_matter_re.search(entry.title):
            continue
        doc_href, _ = _get_doc_for_href(docs, entry.base_href)
        if doc_href and classification.get(doc_href) in ("front", "back"):
            continue
        content_entries.append(entry)

    if not content_entries:
        content_entries = leaf_entries

    # Build candidates
    candidates = []
    chapter_idx = 0

    for entry in content_entries:
        doc_href, doc_data = _get_doc_for_href(docs, entry.base_href)
        if doc_data is None:
            continue
        soup, body_html = doc_data
        ancestors = entry_ancestors.get(id(entry), [])

        # Build abbreviated ancestor labels
        ancestor_labels = [
            _abbreviate_label(a, _guess_role(a)) for a in ancestors
        ]

        # Check for sub-chapters in this file
        sub_tag = _find_sub_chapter_tag(soup) if do_sub_split else None

        if sub_tag:
            chunks = _split_file_at_headings(soup, body_html, sub_tag)
            for chunk_tag, chunk_heading, chunk_html in chunks:
                # Skip tiny pre-heading content (section title pages)
                if not chunk_tag:
                    wc = len(re.sub(r"<[^>]+>", " ", chunk_html).split())
                    if wc < 50:
                        continue

                chapter_idx += 1
                parts = list(ancestor_labels)
                parts.append(_abbreviate_label(entry.title, _guess_role(entry.title)))

                if chunk_heading:
                    subtitle = _get_subtitle_after_heading(soup, sub_tag, chunk_heading)
                    if subtitle:
                        parts.append(f"{chunk_heading} — {subtitle}")
                    else:
                        parts.append(chunk_heading)

                candidates.append(ChapterCandidate(
                    index=chapter_idx,
                    label=", ".join(parts),
                    hierarchy_parts=parts,
                    heading_tag=sub_tag,
                    html_content=chunk_html,
                    source_file=doc_href or "",
                    toc_entry=entry,
                ))
        else:
            # Single chapter from this file
            chapter_idx += 1
            parts = list(ancestor_labels)
            parts.append(_abbreviate_label(entry.title, _guess_role(entry.title)))

            candidates.append(ChapterCandidate(
                index=chapter_idx,
                label=", ".join(parts),
                hierarchy_parts=parts,
                heading_tag="toc",
                html_content=body_html,
                source_file=doc_href or "",
                toc_entry=entry,
            ))

    if not candidates:
        print("WARNING: No chapters found via TOC. Falling back to heading scan.")
        # Simple fallback: use heading tags directly
        skip_hrefs = audit.get("skip_hrefs", set())
        split_tag = audit.get("split_tag")
        if split_tag:
            for href, (s, bh) in docs.items():
                if href in skip_hrefs:
                    continue
                body = s.find("body")
                if not body:
                    continue
                for el in body.find_all(split_tag):
                    text = el.get_text(strip=True)
                    if text:
                        chapter_idx += 1
                        candidates.append(ChapterCandidate(
                            index=chapter_idx, label=text,
                            hierarchy_parts=[text], heading_tag=split_tag,
                            html_content=str(el), source_file=href,
                        ))

    return candidates


# ============================================================================
# Phase 2: Interactive Selection
# ============================================================================

def display_chapter_list(candidates: list[ChapterCandidate]) -> list[dict]:
    """Display the proposed chapter list with warnings. Returns list of warning dicts."""
    print("\n" + "=" * 65)
    print("PHASE 2: Chapter Review")
    print("=" * 65)

    total_words = sum(c.word_count for c in candidates)
    print(f"\nProposed chapters ({len(candidates)} total, {total_words:,} words):\n")

    warnings_list = []

    # Track labels for duplicate detection
    label_counts: Counter = Counter()
    for c in candidates:
        label_counts[c.label] += 1

    max_words = 15000
    min_words = 50

    for c in candidates:
        flags = []
        if c.word_count > max_words:
            flags.append(f"LARGE: {c.word_count:,} words")
            warnings_list.append({
                "index": c.index, "type": "large",
                "message": f"Chapter exceeds {max_words:,} word limit ({c.word_count:,} words)"
            })
        if c.word_count < min_words:
            flags.append(f"TINY: {c.word_count} words")
            warnings_list.append({
                "index": c.index, "type": "tiny",
                "message": f"Chapter under {min_words} words ({c.word_count} words) — likely a title page"
            })
        if label_counts[c.label] > 1:
            flags.append("DUPLICATE LABEL")
            warnings_list.append({
                "index": c.index, "type": "duplicate",
                "message": f"Label \"{c.label}\" appears {label_counts[c.label]} times"
            })

        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  {c.index:3d}. {c.full_label} ({c.word_count:,} words){flag_str}")

    if warnings_list:
        print(f"\n  Warnings: {len(warnings_list)} issue(s) flagged above.")

    return warnings_list


def interactive_selection(
    candidates: list[ChapterCandidate],
    audit: dict,
    auto_mode: bool = False,
    exclude_keywords: list[str] | None = None,
) -> tuple[list[ChapterCandidate], str | None]:
    """Phase 2: Let the user review and select chapters.

    Returns (selected_candidates, new_split_level_or_None).
    """
    warnings_list = display_chapter_list(candidates)

    # Auto-mode: apply keyword exclusions and return
    if auto_mode:
        selected = candidates
        if exclude_keywords:
            before_count = len(selected)
            selected = [
                c for c in selected
                if not any(
                    kw.lower() in c.label.lower() or kw.lower() in c.html_content[:200].lower()
                    for kw in exclude_keywords
                )
            ]
            excluded_count = before_count - len(selected)
            if excluded_count:
                print(f"\n  Auto-excluded {excluded_count} items matching keywords: {', '.join(exclude_keywords)}")
        # Re-index
        for i, c in enumerate(selected, 1):
            c.index = i
        print(f"\n  Auto-mode: accepting {len(selected)} chapters.")
        return selected, None

    # Interactive mode
    print("\nOptions:")
    print("  [Enter]     Accept as-is")
    print("  [numbers]   Exclude items (e.g., 1-3,40-42)")
    print("  [s]         Change split level")
    print("  [q]         Quit without writing")

    choice = input("\nYour choice: ").strip()

    if choice.lower() == "q":
        print("Aborted.")
        sys.exit(0)

    if choice.lower() == "s":
        hierarchy = audit["hierarchy"]
        if hierarchy:
            print("\nAvailable split levels:")
            for tag, role in sorted(hierarchy.items(), key=lambda x: int(x[0][1])):
                count = len(audit["heading_data"].get(tag, []))
                print(f"  {role} (<{tag}>) — {count} entries")
            new_level = input("Enter level name (e.g., section, part): ").strip().lower()
            return candidates, new_level
        else:
            print("No hierarchy detected. Cannot change split level.")
            return candidates, None

    if not choice:
        # Accept as-is
        return candidates, None

    # Parse exclusion numbers
    try:
        exclude_nums = set()
        for part in choice.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                exclude_nums.update(range(int(lo), int(hi) + 1))
            elif part:
                exclude_nums.add(int(part))
    except ValueError:
        print("Invalid input. Keeping all chapters.")
        exclude_nums = set()

    selected = [c for c in candidates if c.index not in exclude_nums]
    excluded_count = len(candidates) - len(selected)
    print(f"\n  Excluded {excluded_count} items.")

    # Re-index
    for i, c in enumerate(selected, 1):
        c.index = i

    return selected, None


# ============================================================================
# Phase 3: Normalize and Output
# ============================================================================

def normalize_and_write(
    candidates: list[ChapterCandidate],
    epub_path: Path,
    output_dir: Path,
    book: ep.EpubBook,
    audit: dict,
) -> tuple[Path, Path]:
    """Phase 3: Generate normalized HTML and manifest JSON.

    Returns (html_path, manifest_path).
    """
    print("\n" + "=" * 65)
    print("PHASE 3: Normalize and Output")
    print("=" * 65)

    # Get metadata
    title_meta = book.get_metadata("DC", "title")
    title = title_meta[0][0] if title_meta else epub_path.stem
    author_meta = book.get_metadata("DC", "creator")
    author = author_meta[0][0] if author_meta else ""

    stem = epub_path.stem

    # Build normalized HTML
    html_parts = [
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{_escape_html(title)}</title>\n"
    ]
    if author:
        html_parts.append(f'<meta name="author" content="{_escape_html(author)}">\n')
    html_parts.append("</head>\n<body>\n")

    for c in candidates:
        # Write a normalized <h2> with the full hierarchical label
        heading_text = _escape_html(c.full_label)
        html_parts.append(f"\n<h2>{heading_text}</h2>\n")

        # Extract body content (everything after the original heading tag)
        content = c.html_content
        # Remove the original heading from the content to avoid duplication
        content = _strip_leading_heading(content, c.heading_tag)
        html_parts.append(content)
        html_parts.append("\n")

    html_parts.append("\n</body>\n</html>")

    # Write HTML
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{stem}_normalized.html"
    html_path.write_text("".join(html_parts), encoding="utf-8")
    print(f"\n  Wrote: {html_path}")

    # Build manifest
    hierarchy = audit.get("hierarchy", {})
    total_words = sum(c.word_count for c in candidates)

    manifest = {
        "source_epub": epub_path.name,
        "title": title,
        "author": author,
        "split_level": hierarchy.get(audit.get("split_tag", ""), "chapter"),
        "heading_levels": {tag: role for tag, role in hierarchy.items()},
        "total_chapters": len(candidates),
        "total_words": total_words,
        "chapters": [
            {
                "index": c.index,
                "label": c.full_label,
                "words": c.word_count,
            }
            for c in candidates
        ],
    }

    manifest_path = output_dir / f"{stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Wrote: {manifest_path}")
    print(f"\n  Summary: {len(candidates)} chapters, {total_words:,} words total")

    return html_path, manifest_path


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _strip_leading_heading(html: str, tag_name: str) -> str:
    """Remove leading heading tags from the HTML content to avoid duplication
    with the normalized <h2> we insert."""
    if not tag_name or tag_name == "toc":
        # For TOC-derived chapters, strip all leading heading-like elements
        # (h-tags and known heading divs like ct, cst, cpt, cct, ctag1)
        stripped = html
        for _ in range(8):
            m = re.match(
                r"\s*<(h[1-6])\b[^>]*>.*?</\1>",
                stripped,
                re.DOTALL | re.IGNORECASE,
            )
            if m:
                stripped = stripped[m.end():]
                continue
            m = re.match(
                r'\s*<(div|p)\s+class="(ct|cst|cpt|cct|ctag1)[^"]*"[^>]*>.*?</\1>',
                stripped,
                re.DOTALL | re.IGNORECASE,
            )
            if m:
                stripped = stripped[m.end():]
                continue
            break
        return stripped

    # For heading-tag splits, strip the specific heading and its subtitle
    pattern = re.compile(
        rf"^\s*<{re.escape(tag_name)}[^>]*>.*?</{re.escape(tag_name)}>",
        re.DOTALL | re.IGNORECASE,
    )
    result = pattern.sub("", html, count=1)
    # Also strip a subtitle heading (one level lower) if it immediately follows
    h_level = int(tag_name[1])
    subtitle_tag = f"h{h_level + 1}"
    subtitle_pattern = re.compile(
        rf"^\s*<{re.escape(subtitle_tag)}[^>]*>.*?</{re.escape(subtitle_tag)}>",
        re.DOTALL | re.IGNORECASE,
    )
    result = subtitle_pattern.sub("", result, count=1)
    return result


# ============================================================================
# Manifest replay
# ============================================================================

def apply_manifest(
    manifest_path: Path,
    candidates: list[ChapterCandidate],
) -> list[ChapterCandidate]:
    """Apply a previous manifest's selections to the current candidates."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Get the labels from the manifest
    manifest_labels = {ch["label"] for ch in manifest.get("chapters", [])}

    if not manifest_labels:
        print("WARNING: Manifest has no chapter entries. Using all candidates.")
        return candidates

    # Match by label
    selected = []
    for c in candidates:
        if c.full_label in manifest_labels:
            selected.append(c)

    if not selected:
        print("WARNING: No manifest labels matched current candidates.")
        print("  This may indicate a different EPUB edition. Using all candidates.")
        return candidates

    print(f"  Applied manifest: selected {len(selected)} of {len(candidates)} chapters")
    # Re-index
    for i, c in enumerate(selected, 1):
        c.index = i

    return selected


# ============================================================================
# CLI and main
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EPUB Preprocessor: audit structure, select chapters, normalize for pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python epub_preprocessor.py book.epub
  python epub_preprocessor.py book.epub --output source/
  python epub_preprocessor.py book.epub --auto --exclude-keywords "contents,index,copyright"
  python epub_preprocessor.py book.epub --split-level section
  python epub_preprocessor.py book.epub --audit-only
  python epub_preprocessor.py book.epub --manifest previous_manifest.json
        """,
    )
    parser.add_argument("epub", help="Path to the EPUB file")
    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: same directory as the EPUB)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Non-interactive mode: accept defaults",
    )
    parser.add_argument(
        "--exclude-keywords",
        help='Comma-separated keywords to exclude (e.g., "contents,index,copyright")',
    )
    parser.add_argument(
        "--split-level",
        choices=["part", "section", "chapter", "subsection"],
        help="Override the split level (default: auto-detected deepest level)",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only run the structural audit, don't write any files",
    )
    parser.add_argument(
        "--manifest",
        help="Path to a previous manifest JSON to replay selections",
    )
    parser.add_argument(
        "--heading-map",
        help=(
            "Custom CSS-class to heading-tag mapping as a JSON string or "
            "path to a JSON file.  Example: "
            '\'{"mypart": "h1", "mychap": "h2"}\' '
            "or  heading_map.json"
        ),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve EPUB path
    epub_path = Path(args.epub).resolve()
    if not epub_path.exists():
        print(f"ERROR: File not found: {args.epub}")
        sys.exit(1)
    if epub_path.suffix.lower() != ".epub":
        print(f"ERROR: Not an EPUB file: {args.epub}")
        sys.exit(1)

    # Resolve output directory
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = epub_path.parent

    # Parse exclude keywords
    exclude_keywords = None
    if args.exclude_keywords:
        exclude_keywords = [kw.strip() for kw in args.exclude_keywords.split(",") if kw.strip()]

    # Parse heading map
    heading_map = None
    if args.heading_map:
        raw = args.heading_map.strip()
        if raw.startswith("{"):
            # Inline JSON string
            try:
                heading_map = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"ERROR: Invalid JSON in --heading-map: {exc}")
                sys.exit(1)
        else:
            # Treat as a file path
            hm_path = Path(raw).resolve()
            if not hm_path.exists():
                print(f"ERROR: Heading-map file not found: {raw}")
                sys.exit(1)
            try:
                heading_map = json.loads(hm_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"ERROR: Invalid JSON in heading-map file: {exc}")
                sys.exit(1)
        # Normalize: keys to lowercase, values to lowercase h-tag
        heading_map = {
            k.lower(): v.lower() if v.lower().startswith("h") else f"h{v}"
            for k, v in heading_map.items()
        }
        print(f"  Custom heading map: {heading_map}")

    # --- Phase 1: Structural Audit ---
    book = read_epub(epub_path)
    audit = run_structural_audit(book, epub_path, heading_map=heading_map)

    if args.audit_only:
        print("\n" + "=" * 65)
        print("Audit complete (--audit-only). No files written.")
        print("=" * 65)
        return

    # --- Build chapter candidates ---
    candidates = build_chapter_candidates(audit, split_level=args.split_level)

    if not candidates:
        print("ERROR: Could not identify any chapters in this EPUB.")
        sys.exit(1)

    # --- Apply manifest if provided ---
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        if not manifest_path.exists():
            print(f"ERROR: Manifest not found: {args.manifest}")
            sys.exit(1)
        candidates = apply_manifest(manifest_path, candidates)

    # --- Phase 2: Interactive Selection ---
    while True:
        selected, new_split_level = interactive_selection(
            candidates,
            audit,
            auto_mode=args.auto,
            exclude_keywords=exclude_keywords,
        )

        if new_split_level:
            # Re-build candidates with new split level
            candidates = build_chapter_candidates(audit, split_level=new_split_level)
            if not candidates:
                print(f"ERROR: No chapters found at split level '{new_split_level}'.")
                continue
            # Loop back to show new list
            continue

        break

    # --- Phase 3: Normalize and Output ---
    html_path, manifest_path = normalize_and_write(
        selected, epub_path, output_dir, book, audit,
    )

    print("\n" + "=" * 65)
    print("Done. Next step:")
    print(f"  python pipeline.py {html_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
