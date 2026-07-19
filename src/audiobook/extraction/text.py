"""Cleanup shared by every book-source backend.

Extraction backends differ in how they find chapters — bookmarks in a PDF,
spine documents and a navigation map in an EPUB — but they converge on the same
Markdown-ish text.  The noise removed here is the noise that survives that
convergence: layout hyphenation, figure captions, bare citation markers, and
inline Markdown emphasis nobody should hear read aloud.
"""

from __future__ import annotations

import re


RE_BOLD = re.compile(r"\*{1,2}([^*]+)\*{1,2}")
RE_CODE = re.compile(r"`([^`]+)`")
RE_LINKS = re.compile(r"\[([^\]]+)\]\([^)]+\)")
RE_IMGS = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
RE_HYPHENS = re.compile(r"\s*-\s*\n\s*")
RE_PAGENUMS = re.compile(r"(\d+)\s*\n\s*(?=\S)")
RE_NEWLINES = re.compile(r"\n{3,}")
RE_WHITESPACE = re.compile(r"[ \t]+")
RE_FIGS = re.compile(r"\b(fig\.|figure|table)\s*\d+[.:]*", re.IGNORECASE)
RE_CITATIONS_PAREN = re.compile(r"\(\s*\d+\s*\)")
RE_CITATIONS_BRACKET = re.compile(r"\[\s*\d+\s*\]")
RE_NUMBERED_CHAPTER = re.compile(r"^\s*\d+\s+\S")
RE_PART_BOOKMARK = re.compile(r"^\s*[IVXLCDM]+\s+\S", re.IGNORECASE)
RE_STANDALONE_PAGE_NUMBER = re.compile(
    r"(?m)^[ \t]*#{0,6}[ \t]*(?:\d[ \t]*)+[ \t]*$"
)
RE_GUTENBERG_SECTION = re.compile(r"project gutenberg", re.IGNORECASE)

# Section names that title apparatus rather than narration.  Reading a table of
# contents aloud is worse than skipping it, and both backends reach for this
# list when they fall back to structure they did not fully trust — an outline
# level, a spine document — so the two agree on what a listener expects to hear.
SKIPPED_SECTIONS = frozenset(
    {
        "about the author",
        "about this book",
        "advertisement",
        "advertisements",
        "bibliography",
        "colophon",
        "contents",
        "copyright",
        "cover",
        "dedication",
        "endnotes",
        "footnotes",
        "further reading",
        "glossary",
        "half title",
        "half-title",
        "index",
        "list of figures",
        "list of illustrations",
        "list of tables",
        "table of contents",
        "title page",
        "transcriber's note",
        "transcriber's notes",
    }
)


def is_skipped_section(title: str) -> bool:
    """Whether a section title names apparatus rather than narration."""
    if RE_GUTENBERG_SECTION.search(title):
        return True
    normalized = re.sub(r"[^a-z' ]+", "", title.casefold().replace("’", "'")).strip()
    return normalized in SKIPPED_SECTIONS


def has_narratable_body(content: str) -> bool:
    """Whether anything remains once the headings are set aside.

    A section whose whole slice is its own title — common where a front-matter
    heading and the section after it live in one document, or where an outline
    entry points at a part divider — is a chapter with nothing to read.
    """

    return any(
        line.strip() and not line.lstrip().startswith("#")
        for line in content.splitlines()
    )


def clean_text_segment(text: str) -> str:
    """Remove PDF and Markdown noise while preserving paragraph boundaries."""
    text = RE_IMGS.sub("", text)
    text = RE_LINKS.sub(r"\1", text)
    text = RE_BOLD.sub(r"\1", text)
    text = RE_CODE.sub(r"\1", text)
    text = RE_HYPHENS.sub("", text)
    text = RE_PAGENUMS.sub(r"\1 ", text)
    text = RE_FIGS.sub("", text)
    text = RE_CITATIONS_PAREN.sub("", text)
    text = RE_CITATIONS_BRACKET.sub("", text)
    text = RE_NEWLINES.sub("\n\n", text)
    text = RE_WHITESPACE.sub(" ", text)
    return text.strip()


__all__ = [
    "RE_BOLD",
    "RE_CITATIONS_BRACKET",
    "RE_CITATIONS_PAREN",
    "RE_CODE",
    "RE_FIGS",
    "RE_GUTENBERG_SECTION",
    "RE_HYPHENS",
    "RE_IMGS",
    "RE_LINKS",
    "RE_NEWLINES",
    "RE_NUMBERED_CHAPTER",
    "RE_PAGENUMS",
    "RE_PART_BOOKMARK",
    "RE_STANDALONE_PAGE_NUMBER",
    "RE_WHITESPACE",
    "SKIPPED_SECTIONS",
    "clean_text_segment",
    "has_narratable_body",
    "is_skipped_section",
]
