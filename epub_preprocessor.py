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

    # 2. Count spine items
    spine_items = get_spine_items(book)
    print(f"\n--- Spine: {len(spine_items)} content files ---")

    # 3. Parse content files
    docs = parse_content_files(book)

    # 3b. Promote styled divs/paragraphs to semantic heading tags
    promote_styled_headings(docs, heading_map=heading_map)

    # 4. Scan headings
    heading_data = scan_headings(docs)
    print(f"\n--- Heading Tag Survey ---")
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
    non_standard = detect_non_standard_headings(docs)
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

def build_chapter_candidates(
    audit: dict,
    split_level: str | None = None,
) -> list[ChapterCandidate]:
    """Build the list of chapter candidates based on audit results and split level.

    Handles edge cases: single-file EPUBs, multi-file chapters, no TOC, etc.
    """
    hierarchy = audit["hierarchy"]
    heading_data = audit["heading_data"]
    docs = audit["docs"]
    toc_tree = audit["toc_tree"]
    flat_toc = audit["flat_toc"]

    # Determine which tag to split on
    if split_level and hierarchy:
        # User-specified split level (e.g., "section", "part", "chapter")
        split_tag = None
        for tag, role in hierarchy.items():
            if role == split_level:
                split_tag = tag
                break
        if not split_tag:
            print(f"WARNING: Split level '{split_level}' not found in hierarchy.")
            print(f"  Available levels: {', '.join(hierarchy.values())}")
            print(f"  Falling back to default split tag.")
            split_tag = audit["split_tag"]
    else:
        split_tag = audit["split_tag"]

    if not split_tag:
        # Last resort: pick the heading tag with the most occurrences
        if heading_data:
            split_tag = max(heading_data.keys(), key=lambda t: len(heading_data[t]))
            print(f"  Using most frequent heading tag: <{split_tag}>")
        else:
            print("ERROR: No heading tags found. Cannot build chapter list.")
            sys.exit(1)

    # Determine parent hierarchy tags (tags that represent levels above the split)
    parent_tags = []
    if hierarchy:
        role_order = ["part", "section", "chapter", "subsection"]
        split_role = hierarchy.get(split_tag, "chapter")
        split_idx = role_order.index(split_role) if split_role in role_order else len(role_order)
        for tag, role in sorted(hierarchy.items(), key=lambda x: int(x[0][1])):
            if role in role_order:
                role_idx = role_order.index(role)
                if role_idx < split_idx:
                    parent_tags.append((tag, role))

    # Build concatenated HTML from all content documents in spine order
    all_html_parts = []
    file_boundaries = []  # (start_pos, href)
    current_pos = 0
    for href, (soup, body_html) in docs.items():
        file_boundaries.append((current_pos, href))
        all_html_parts.append(body_html)
        current_pos += len(body_html)

    full_html_raw = "".join(all_html_parts)
    full_soup = BeautifulSoup(full_html_raw, "lxml")

    # Re-serialize from the parsed soup so that str(el) is guaranteed to
    # match substrings of full_html (avoids serialization mismatches).
    body_tag = full_soup.find("body")
    if body_tag:
        full_html = "".join(str(c) for c in body_tag.children)
    else:
        full_html = str(full_soup)

    # Recompute file boundaries against the re-serialized HTML.
    # (Approximate: walk docs in order and find their first heading or
    # a unique text snippet to anchor the boundary.)
    file_boundaries = []
    boundary_offset = 0
    for href, (_soup, body_html) in docs.items():
        file_boundaries.append((boundary_offset, href))
        # Advance by the approximate length; exact match isn't critical
        # since this is only used for source_file metadata.
        boundary_offset += len(body_html)

    # Find all split-level headings and parent headings.
    # Use progressive search offset so duplicate headings are matched in
    # document order instead of all resolving to the first occurrence.
    all_headings = []
    search_offset = 0
    for el in full_soup.find_all(re.compile(r"^h[1-6]$")):
        tag_name = el.name
        text = el.get_text(strip=True)
        if not text:
            continue
        el_str = str(el)
        pos = full_html.find(el_str, search_offset)
        if pos < 0:
            # Try finding by text content from current offset
            pos = full_html.find(text, search_offset)
        if pos < 0:
            # Last resort: search from beginning (shouldn't normally happen)
            pos = full_html.find(el_str)
            if pos < 0:
                pos = full_html.find(text)
        if pos >= 0:
            search_offset = pos + 1
        all_headings.append((pos, tag_name, text, el_str))

    # all_headings is already in document order from find_all(); the sort
    # is kept as a safety net but should be a no-op.
    all_headings.sort(key=lambda x: x[0])

    # Build chapter candidates by tracking parent context
    candidates = []
    parent_context: dict[str, str] = {}  # role -> current label
    chapter_idx = 0

    parent_tag_names = {tag for tag, _role in parent_tags}

    for i, (pos, tag_name, text, el_str) in enumerate(all_headings):
        if tag_name in parent_tag_names:
            # Update parent context
            role = hierarchy.get(tag_name, "")
            parent_context[role] = text
            # Clear deeper levels when a higher level changes
            role_order = ["part", "section", "chapter", "subsection"]
            if role in role_order:
                role_idx = role_order.index(role)
                for r in role_order[role_idx + 1:]:
                    parent_context.pop(r, None)

        if tag_name == split_tag:
            chapter_idx += 1

            # Build hierarchy label
            hierarchy_parts = []
            role_order = ["part", "section", "chapter", "subsection"]
            for role in role_order:
                if role in parent_context:
                    # Abbreviate: "Part I: Title" stays, "Section II" -> "Sec II"
                    val = parent_context[role]
                    hierarchy_parts.append(_abbreviate_label(val, role))

            # Add the current chapter heading
            hierarchy_parts.append(text)

            # Determine content: from this heading to the next split-level heading
            # (or next parent heading, or end of document)
            start_pos = pos
            end_pos = len(full_html)
            for j in range(i + 1, len(all_headings)):
                next_pos, next_tag, _next_text, _next_str = all_headings[j]
                if next_tag == split_tag or next_tag in parent_tag_names:
                    end_pos = next_pos
                    break

            content_html = full_html[start_pos:end_pos]

            # Find the source file
            source_file = ""
            for boundary_pos, href in reversed(file_boundaries):
                if pos >= boundary_pos:
                    source_file = href
                    break

            # Build descriptive label
            if len(hierarchy_parts) > 1:
                label = ", ".join(hierarchy_parts[:-1]) + " — " + hierarchy_parts[-1]
            else:
                label = hierarchy_parts[0]

            # Find matching TOC entry
            toc_match = None
            for entry in flat_toc:
                if _text_match(entry.title, text):
                    toc_match = entry
                    break

            candidates.append(ChapterCandidate(
                index=chapter_idx,
                label=label,
                hierarchy_parts=hierarchy_parts,
                heading_tag=split_tag,
                html_content=content_html,
                source_file=source_file,
                toc_entry=toc_match,
            ))

    if not candidates:
        print(f"WARNING: No chapters found using <{split_tag}> as split tag.")
        print("  Trying to build chapters from TOC entries directly...")
        candidates = _build_from_toc_fallback(audit)

    return candidates


def _abbreviate_label(text: str, role: str) -> str:
    """Abbreviate hierarchy labels for display. E.g., 'Section II' -> 'Sec II'."""
    abbreviations = {
        "part": (r"(?i)^part\b", "Part"),
        "section": (r"(?i)^section\b", "Sec"),
        "chapter": (r"(?i)^chapter\b", "Ch"),
    }
    if role in abbreviations:
        pattern, replacement = abbreviations[role]
        if re.match(pattern, text):
            return re.sub(pattern, replacement, text, count=1)
    return text


def _build_from_toc_fallback(audit: dict) -> list[ChapterCandidate]:
    """Fallback: build chapter candidates directly from TOC when heading-based
    splitting finds nothing."""
    flat_toc = audit["flat_toc"]
    docs = audit["docs"]

    if not flat_toc:
        return []

    candidates = []
    for idx, entry in enumerate(flat_toc, 1):
        if not entry.base_href:
            continue

        # Find matching document
        doc_key = None
        for href in docs:
            if href == entry.base_href or href.split("/")[-1] == entry.base_href.split("/")[-1]:
                doc_key = href
                break

        if doc_key is None:
            continue

        _soup, body_html = docs[doc_key]
        candidates.append(ChapterCandidate(
            index=idx,
            label=entry.title,
            hierarchy_parts=[entry.title],
            heading_tag="toc",
            html_content=body_html,
            source_file=doc_key,
            toc_entry=entry,
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
    """Remove the first heading tag from the HTML content to avoid duplication
    with the normalized <h2> we insert."""
    # Match the first occurrence of the heading tag (including nested content)
    pattern = re.compile(
        rf"^\s*<{re.escape(tag_name)}[^>]*>.*?</{re.escape(tag_name)}>",
        re.DOTALL | re.IGNORECASE,
    )
    return pattern.sub("", html, count=1)


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
