"""Extract chapter-oriented narration text from PDF input.

This module owns the deterministic cleanup that happens before any optional
model-assisted narration adaptation.  It deliberately preserves paragraph and
chapter structure so later workflow stages can make semantic decisions without
depending on PDF-specific details.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from .text import (
    RE_BOLD,
    RE_CITATIONS_BRACKET,
    RE_CITATIONS_PAREN,
    RE_CODE,
    RE_FIGS,
    RE_HYPHENS,
    RE_IMGS,
    RE_LINKS,
    RE_NEWLINES,
    RE_NUMBERED_CHAPTER,
    RE_PAGENUMS,
    RE_PART_BOOKMARK,
    RE_STANDALONE_PAGE_NUMBER,
    RE_WHITESPACE,
    clean_text_segment,
    has_narratable_body,
    is_skipped_section,
)


RE_MARKDOWN_HEADING = re.compile(r"(?m)^(#{1,6})[ \t]+(\S.*?)[ \t]*$")


def _join_markdown_pages(page_texts: Sequence[str]) -> str:
    """Join page-top sentence continuations without inventing a paragraph."""
    result = ""
    for page_text in page_texts:
        page_text = page_text.strip()
        if not page_text:
            continue
        if not result:
            result = page_text
            continue

        visible_start = page_text.lstrip("#*_ `\t")
        continues_sentence = bool(visible_start) and (
            visible_start[0].islower() or visible_start[0] in ",;:)”’"
        )
        if continues_sentence:
            page_text = re.sub(r"^#+[ \t]*", "", page_text, count=1)
            if result.endswith("-"):
                result = result[:-1] + page_text
            else:
                result += " " + page_text
        else:
            result += "\n\n" + page_text
    return result


def _select_bookmarks(
    outline: Sequence[tuple[int, str, int]],
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], str]:
    """Choose which outline entries start chapters, and which merely end them.

    Numbered entries are the strongest signal a PDF gives: they name chapters
    and nothing else, and roman-numbered parts around them are boundaries
    without being narrated.  Plenty of books number nothing, though, so when no
    entry looks numbered the outline's shallowest level is used instead — that
    is the level a table of contents prints — with apparatus filtered out by
    title.  Returns the chapter starts, the boundaries, and a label for logging.
    """

    numbered: list[tuple[str, int]] = []
    structural: list[tuple[str, int]] = []
    entries: list[tuple[int, str, int]] = []
    for level, raw_title, page_number in outline:
        title = " ".join(raw_title.split())
        if not title or page_number < 1:
            continue
        entries.append((level, title, page_number))
        if RE_NUMBERED_CHAPTER.match(title):
            numbered.append((title, page_number))
            structural.append((title, page_number))
        elif RE_PART_BOOKMARK.match(title):
            structural.append((title, page_number))
    if numbered:
        return numbered, structural, "bookmarks"

    if entries:
        top_level = min(level for level, _title, _page in entries)
        top_entries = [
            (title, page_number)
            for level, title, page_number in entries
            if level == top_level
        ]
        # A single top-level entry describes the book, not its chapters; drop
        # to the heading fallback rather than narrate the whole PDF as "Cover".
        if len(top_entries) > 1:
            narrated = [
                (title, page_number)
                for title, page_number in top_entries
                if not is_skipped_section(title)
            ]
            # Skipped sections still bound the chapter before them.
            return narrated, top_entries, "outline"
    return [], [], "none"


def _split_markdown_headings(markdown: str) -> list[tuple[str, str]]:
    """Split unbookmarked page text on its shallowest Markdown heading level.

    ``pymupdf4llm`` promotes visually larger type to headings, so a PDF with no
    outline at all usually still marks its chapter openers.  Anything short of
    two headings at the same level is not structure worth trusting.
    """

    headings = list(RE_MARKDOWN_HEADING.finditer(markdown))
    if not headings:
        return []
    top_level = min(len(match.group(1)) for match in headings)
    starts = [match for match in headings if len(match.group(1)) == top_level]
    if len(starts) < 2:
        return []

    chapters: list[tuple[str, str]] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(markdown)
        # A heading inferred from large type often carries emphasis markers that
        # belong to the page, not to the chapter's name.
        title = RE_BOLD.sub(r"\1", RE_CODE.sub(r"\1", match.group(2)))
        title = " ".join(RE_LINKS.sub(r"\1", RE_IMGS.sub("", title)).split())
        if not title or is_skipped_section(title):
            continue
        content = clean_text_segment(markdown[match.start() : end])
        if not has_narratable_body(content):
            continue
        chapters.append((title, content))
    return chapters


def parse_pdf_to_chapters(pdf_path: Path) -> list[tuple[str, str]]:
    """Parse a PDF into chapters using bookmarks instead of numbered body text.

    Embedded bookmark page hints are aligned with nearby visible headings,
    structural part bookmarks delimit content without being narrated, and
    common unbookmarked front-matter sections are included when present.

    Structure is taken from the best source the file actually offers, in order:
    numbered chapter bookmarks, the outline's shallowest level, the headings
    ``pymupdf4llm`` infers from the page text, and finally one whole-book
    chapter — so a PDF that numbers nothing still arrives as chapters.
    """
    import pymupdf
    import pymupdf4llm

    print(f"Parsing PDF structure: {pdf_path}...")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = pymupdf.open(pdf_path)
    bookmarks, structural_bookmarks, source = _select_bookmarks(document.get_toc())

    if not bookmarks:
        print("No usable chapter bookmarks detected; looking for headings instead.")
        markdown = RE_STANDALONE_PAGE_NUMBER.sub(
            "", pymupdf4llm.to_markdown(document, page_chunks=False)
        )
        chapters = _split_markdown_headings(markdown)
        if chapters:
            print(f"Detected {len(chapters)} chapters from PDF headings.")
            return chapters
        print("No usable headings either; treating the PDF as one chapter.")
        return [("Audiobook", clean_text_segment(markdown))]

    if source == "bookmarks":
        # Embedded bookmarks often omit preface/introduction entries. Include
        # exact standard-section headings found before the first numbered
        # chapter.  An outline used whole already carries whatever it names.
        first_chapter_page = bookmarks[0][1]
        prelude_titles = {"preface", "introduction", "foreword", "prologue"}
        prelude_pages: dict[str, tuple[str, int]] = {}
        for page_index in range(first_chapter_page - 1):
            lines = [
                line.strip() for line in document[page_index].get_text().splitlines()
            ]
            for line in lines[:8]:
                if line.lower() in prelude_titles:
                    prelude_pages[line.lower()] = (line.title(), page_index + 1)
                    break
        bookmarks.extend(prelude_pages.values())
        structural_bookmarks.extend(prelude_pages.values())

    def searchable(text: str) -> str:
        return "".join(character.lower() for character in text if character.isalnum())

    def align_to_heading(title: str, page_hint: int) -> int:
        needle = searchable(title)
        candidate_pages = [page_hint]
        for offset in range(1, 4):
            candidate_pages.extend((page_hint - offset, page_hint + offset))
        for page_number in candidate_pages:
            if not 1 <= page_number <= document.page_count:
                continue
            if needle in searchable(document[page_number - 1].get_text()):
                return page_number
        return page_hint

    bookmarks = sorted(
        {(title, align_to_heading(title, page)) for title, page in bookmarks},
        key=lambda item: item[1],
    )
    structural_pages = sorted(
        {align_to_heading(title, page) for title, page in structural_bookmarks}
    )
    first_narrated_page = bookmarks[0][1]
    page_chunks = pymupdf4llm.to_markdown(
        document,
        pages=list(range(first_narrated_page - 1, document.page_count)),
        page_chunks=True,
    )
    text_by_page = {
        item["metadata"]["page_number"]: RE_STANDALONE_PAGE_NUMBER.sub(
            "", item["text"]
        )
        for item in page_chunks
    }

    chapters: list[tuple[str, str]] = []
    for title, start_page in bookmarks:
        next_boundaries = [page for page in structural_pages if page > start_page]
        end_page = next_boundaries[0] - 1 if next_boundaries else document.page_count
        markdown = _join_markdown_pages(
            [
                text_by_page.get(page_number, "")
                for page_number in range(start_page, end_page + 1)
            ]
        )
        content = clean_text_segment(markdown)
        spoken_title = re.sub(r"^\d+\s+", "", title).strip()
        if content.lower().startswith(spoken_title.lower()) and not content.startswith("#"):
            content = "# " + content
        if content:
            chapters.append((title, content))

    if not chapters:
        print("Bookmarked sections held no text; treating the PDF as one chapter.")
        markdown = RE_STANDALONE_PAGE_NUMBER.sub(
            "", pymupdf4llm.to_markdown(document, page_chunks=False)
        )
        return [("Audiobook", clean_text_segment(markdown))]

    print(f"Detected {len(chapters)} chapters from the PDF {source}.")
    return chapters


__all__ = [
    "RE_BOLD",
    "RE_CITATIONS_BRACKET",
    "RE_CITATIONS_PAREN",
    "RE_CODE",
    "RE_FIGS",
    "RE_HYPHENS",
    "RE_IMGS",
    "RE_LINKS",
    "RE_NEWLINES",
    "RE_NUMBERED_CHAPTER",
    "RE_PAGENUMS",
    "RE_PART_BOOKMARK",
    "RE_STANDALONE_PAGE_NUMBER",
    "RE_MARKDOWN_HEADING",
    "RE_WHITESPACE",
    "_join_markdown_pages",
    "_select_bookmarks",
    "_split_markdown_headings",
    "clean_text_segment",
    "parse_pdf_to_chapters",
]
