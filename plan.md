# EPUB Preprocessor — Fix Plan

## Status: IN PROGRESS

---

## 1. Diagnosis (from testing `theoryofmoralsentiment.epub`)

Running the current script with `--auto` produces **82 chapters** (should be ~23–58
depending on split depth) and assigns the wrong parent context ("Part VII, Sec IV")
to every single entry. There is a 47,347-word mega-chapter and 4 tiny (<50 word) entries.

### Root causes

#### Bug A: TOC *page* headings are promoted instead of content headings

The EPUB's Table of Contents **page** (`toc_r1.xhtml`) is a spine item. It contains
`div.cpt` and `div.cct` elements — the same classes the promotion map targets. Those
7 Part headings and 23 Section headings from the *TOC page* get promoted to `<h1>`/`<h2>`,
poisoning the heading analysis. Meanwhile, the actual Part/Section/Chapter headings in
the content files use `div.ct` and `div.cst`, which are **not** in the promotion map.

#### Bug B: `div.ct` and `div.cst` are not promoted

The code comments say ct/cst were "intentionally omitted" as typically redundant.
In this EPUB, they're the **primary** headings for many chapters (Parts III–V, VI Sec I/III,
VII Sec I/IV). Files c07–c16, c17, c19–c20, c23 have **no** h-tags — only div.ct/div.cst.

#### Bug C: Parent context is always "Part VII, Sec IV"

The script concatenates all file HTML, re-parses it, finds all headings, and sorts by
position. Because the promoted Part/Section headings are all on the TOC *page* (early
in the spine), they cluster at the start of the concatenated HTML. The parent-tracking
loop processes all 7 Parts + 23 Sections before any actual content heading, ending
with parent = "Part VII, Sec IV". Then every subsequent h3 inherits that stale context.

#### Bug D: 82 h3 headings instead of ~42 real chapters

After demotion (original h1→h3, h2→h4), h3 contains:
- 13 editorial headings from the Introduction (by Amartya Sen)
- ~30 "CHAPTER I/II/III…" headings from content files
- 25 section headings from Textual Notes (bm1)
- 1 from the "Considerations" essay (tea)
- 1 from next-reads.xhtml

No front/back matter filtering is applied, so everything gets mixed in.

#### Bug E: Full-HTML concatenation is fragile

Re-parsing concatenated HTML with lxml can reorder/discard elements. String-based
position finding (`full_html.find(el_str, offset)`) breaks when lxml re-serializes
HTML differently than the source. This causes headings to get wrong positions.

#### Bug F: All 37 TOC entries are "mismatches"

`cross_reference_toc_headings` looks for heading tags in the file each TOC entry points
to. But the promoted headings are on the TOC page, not in p01/c01/etc. The actual
content files either use un-promoted div.ct/cst or have h1 tags with text like
"CHAPTER I." which doesn't match the TOC text "SECTION I. - OF THE SENSE OF PROPRIETY."

### EPUB structure summary

```
itr  (Introduction)   — 13 x h1 (editorial section headings, NOT book chapters)
p01  (Part I page)     — div.ct "PART I." + div.cst subtitle  (NO h-tags)
c01  (Sec I content)   — div.ct "SECTION I." + 5 x h1 "CHAPTER I/II/…" + 5 x h2 subtitles
c02  (Sec II content)  — same pattern, 1 intro + 5 chapters
c03  (Sec III content) — 3 chapters
p02  (Part II page)    — div.ct/cst only
c04–c06                — same multi-chapter-per-file pattern
p03  (Part III page)   — div.ct/cst only
c07–c12                — ONE chapter per file, div.ct "CHAPTER I." + div.cst subtitle (NO h-tags)
p04–p05, c13–c16       — same single-chapter pattern
p06  (Part VI page)    — div.ct/cst + 1 h1 "INTRODUCTION."
c17                    — div.ct/cst only (single chapter)
c18                    — 1 intro + 3 sub-chapters (h1/h2 pattern)
c19                    — div.ct/cst + h1 "CONCLUSION OF THE SIXTH PART."
p07, c20, c23          — div.ct/cst only (single chapter each)
c21                    — 1 intro + 4 sub-chapters
c22                    — 1 intro + 3 sub-chapters
tea  (Teaser/essay)    — h1 "CONSIDERATIONS" (appendix)
bm1  (Textual Notes)   — 25 x h1 section headings
in1  (Index)           — empty h2 tags
```

---

## 2. Fix Strategy

### Core architectural change: TOC-driven, file-level splitting

Instead of "promote → concatenate → split on one heading tag", the new approach:

1. **Parse the TOC tree** to get the book's intended structure (already works).
2. **Map each TOC entry to its content file(s)** — use file boundaries as the
   primary split mechanism.
3. **Identify front/back matter** by position relative to TOC entries:
   - Everything before the first Part = front matter (exclude by default)
   - Everything after the last Section/Chapter = back matter (flag for user)
4. **For each TOC-mapped content file**, check whether it contains sub-headings
   (multiple chapters within one file). If so, offer sub-splitting.
5. **Build chapter candidates from the TOC-to-file mapping**, not from heading
   tags in a concatenated HTML blob.
6. **Label each chapter** using the TOC hierarchy: Part → Section → Chapter title.

### Specific fixes

| Fix | What | Why |
|-----|------|-----|
| F1 | Skip non-content spine items (TOC page, cover, etc.) during heading analysis | Stops TOC-page headings from poisoning promotion/hierarchy |
| F2 | Add `ct` → appropriate level and `cst` as subtitle to promotion map | Makes div.ct/cst headings visible in files that lack h-tags |
| F3 | Rewrite `build_chapter_candidates` to be TOC-driven | Processes files individually, avoids fragile concatenation |
| F4 | Add front/back matter detection based on TOC position | Filters out Introduction, Textual Notes, Index, etc. |
| F5 | Handle multi-chapter files by detecting internal h-tag boundaries | Splits c01 (5 chapters), c02 (6), etc. correctly |
| F6 | Build hierarchical labels from TOC tree + internal heading text | "Part I, Sec I, Ch I — Of Sympathy" instead of wrong parent |

### Expected output for `theoryofmoralsentiment.epub`

The TOC lists 23 content entries at depth 1 (Sections/Chapters under 7 Parts).
Within those, some files contain multiple sub-chapters:
- c01: 5 sub-chapters, c02: 6, c03: 3, c04: 6, c05: 3, c06: 4
- c18: 4, c21: 5, c22: 4
- c07–c17, c19–c20, c23: 1 each (12 files)

**Deepest split** (section-internal chapters): ~58 chapters
**TOC-level split** (sections as units): 23 chapters
**Default recommendation**: TOC-level split (23 chapters), with option to go deeper.

---

## 3. Implementation Steps

- [x] Step 1: Read spec, current script, test on EPUB, diagnose bugs
- [ ] Step 2: Add front/back matter detection; skip non-content files in analysis
- [ ] Step 3: Fix heading promotion (add ct/cst, handle conflicts)
- [ ] Step 4: Rewrite `build_chapter_candidates` to be TOC-driven
- [ ] Step 5: Fix label building to use TOC hierarchy properly
- [ ] Step 6: Test and verify output on `theoryofmoralsentiment.epub`
- [ ] Step 7: Handle edge case: multi-chapter files (sub-splitting)

---

## 4. Test Results

### Before fixes
```
82 chapters, 194,895 words
- All entries show "Part VII, Sec IV" as parent (WRONG)
- 47,347-word mega-chapter (WRONG)
- Editorial intro, textual notes, index all included (WRONG)
- 49 warnings (duplicates, tiny, large)
```

### After fixes
*(to be filled in after implementation)*
