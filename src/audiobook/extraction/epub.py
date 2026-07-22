"""Extract chapter-oriented narration text from EPUB input.

An EPUB already carries the structure the PDF backend has to reconstruct: the
package manifest lists the documents, the spine orders them, and the navigation
map names the chapters and points at exactly where each one starts.  So this
backend reads that structure directly rather than paginating the book and
guessing chapter boundaries from page numbers — no page headers, no page
numbers, no layout hyphenation, and chapter breaks that land on the author's
boundary rather than near it.

Everything here is stdlib: an EPUB is a ZIP of XML and XHTML.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from .text import (
    SKIPPED_SECTIONS,
    clean_text_segment,
    has_narratable_body,
    is_skipped_section,
)


CONTAINER_PATH = "META-INF/container.xml"
NCX_MEDIA_TYPE = "application/x-dtbncx+xml"
DOCUMENT_MEDIA_TYPES = frozenset({"application/xhtml+xml", "text/html", "application/x-dtbook+xml"})

# Public-domain conversions append a licence that is longer than some chapters
# and belongs to no one's audiobook.  The marker line ends the book's text, so
# everything from it onward is dropped rather than narrated.
RE_GUTENBERG_BOILERPLATE = re.compile(
    r"\*\*\*\s*(START|END) OF (THE|THIS) PROJECT GUTENBERG.*",
    re.IGNORECASE | re.DOTALL,
)
RE_BLANK_LINES = re.compile(r"\n{3,}")

HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "li",
        "blockquote",
        "tr",
        "pre",
        "figcaption",
        "hr",
        "table",
        "ul",
        "ol",
        "dd",
        "dt",
    }
)
IGNORED_TAGS = frozenset({"script", "style", "head", "svg", "title"})


@dataclass(frozen=True)
class _Document:
    """One spine document: its narratable text and where its anchors sit in it."""

    href: str
    text: str
    anchors: dict[str, int]


class _MarkdownExtractor(HTMLParser):
    """Turn one XHTML document into Markdown-ish text, remembering anchors.

    Anchor offsets are what let a navigation entry start a chapter in the
    middle of a document, which is how anthologies and many public-domain
    conversions are laid out: one file, several chapters, split by ``id``.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._length = 0
        self._ignore_depth = 0
        self._heading_level: int | None = None
        # Whitespace seen but not yet written.  Held back so that a space
        # between two inline runs becomes one space, while the same whitespace
        # sitting against a block boundary disappears into it.
        self._pending_space = False
        self.anchors: dict[str, int] = {}

    def _emit(self, text: str) -> None:
        self._parts.append(text)
        self._length += len(text)

    def _emit_inline(self, text: str) -> None:
        if self._pending_space and self._length and not self._parts[-1].endswith("\n"):
            self._emit(" ")
        self._pending_space = False
        self._emit(text)

    def _break_block(self) -> None:
        self._pending_space = False
        if self._length:
            self._emit("\n\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in IGNORED_TAGS:
            self._ignore_depth += 1
            return
        if self._ignore_depth:
            return

        attributes = dict(attrs)
        if tag == "br":
            # A line break inside a heading is layout, not structure: the
            # heading has to stay one line to still read as one.
            if self._heading_level is not None:
                self._pending_space = True
            else:
                self._pending_space = False
                self._emit("\n")
        elif tag in HEADING_TAGS or tag in BLOCK_TAGS:
            self._break_block()

        # Anchors are recorded after the block break and before any heading
        # marker, so a chapter sliced at its anchor starts at its own heading.
        for attribute in ("id", "name"):
            anchor = attributes.get(attribute)
            # First definition wins: a duplicated id is malformed, and the
            # earlier one is the one a navigation entry was written against.
            if anchor and anchor not in self.anchors:
                self.anchors[anchor] = self._length

        if tag in HEADING_TAGS:
            self._heading_level = HEADING_TAGS[tag]
            self._emit("#" * self._heading_level + " ")
            self._pending_space = False

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag in IGNORED_TAGS:
            self._ignore_depth = max(0, self._ignore_depth - 1)
            return
        if self._ignore_depth:
            return
        if tag in HEADING_TAGS:
            self._heading_level = None
            self._break_block()
        elif tag in BLOCK_TAGS:
            self._break_block()

    def handle_data(self, data: str) -> None:
        if self._ignore_depth:
            return
        # Source line wrapping is whitespace, not a line break: an XHTML
        # paragraph reads as one paragraph however the file happens to be laid
        # out, and only <br/> puts a break inside it.
        text = " ".join(data.replace("\xa0", " ").split())
        if not text:
            self._pending_space = True
            return
        if data[:1].isspace():
            self._pending_space = True
        self._emit_inline(text)
        if data[-1:].isspace():
            self._pending_space = True

    def result(self) -> str:
        """The document text, with offsets still valid for ``anchors``.

        Nothing is collapsed here: every anchor offset is an index into this
        string, so rewriting it would move the chapter boundaries.  Runs of
        blank lines are normalised once the slicing is done.
        """

        return "".join(self._parts)


def _parse_xml(data: bytes) -> ElementTree.Element:
    return ElementTree.fromstring(data)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_package_path(archive: zipfile.ZipFile) -> str:
    """Locate the OPF package document the way a reading system does."""

    try:
        container = _parse_xml(archive.read(CONTAINER_PATH))
    except KeyError as exc:
        raise ValueError(f"EPUB is missing {CONTAINER_PATH}") from exc
    for element in container.iter():
        if _local(element.tag) == "rootfile":
            full_path = element.get("full-path")
            if full_path:
                return full_path
    raise ValueError(f"EPUB {CONTAINER_PATH} names no package document")


def _resolve(base_href: str, relative: str) -> str:
    """Resolve an EPUB-internal href against the document that referenced it."""

    return posixpath.normpath(posixpath.join(posixpath.dirname(base_href), relative))


def _read_text(archive: zipfile.ZipFile, href: str) -> bytes | None:
    try:
        return archive.read(href)
    except KeyError:
        return None


def _parse_package(
    archive: zipfile.ZipFile, package_path: str
) -> tuple[str, dict[str, dict[str, str]], list[str], str | None, str | None]:
    """Read the package document into title, manifest, spine, and nav pointers."""

    package = _parse_xml(archive.read(package_path))
    title = ""
    manifest: dict[str, dict[str, str]] = {}
    spine: list[str] = []
    ncx_id: str | None = None
    nav_href: str | None = None

    for element in package.iter():
        name = _local(element.tag)
        if name == "title" and not title and element.text:
            title = " ".join(element.text.split())
        elif name == "item":
            item_id = element.get("id")
            href = element.get("href")
            if not item_id or not href:
                continue
            manifest[item_id] = {
                "href": _resolve(package_path, href),
                "media_type": element.get("media-type", ""),
                "properties": element.get("properties", ""),
            }
        elif name == "spine":
            ncx_id = element.get("toc")
        elif name == "itemref":
            idref = element.get("idref")
            # linear="no" marks material a reading system shows out of band —
            # covers and colophons — which is exactly what narration skips.
            if idref and element.get("linear", "yes") != "no":
                spine.append(idref)

    for item in manifest.values():
        if "nav" in item["properties"].split():
            nav_href = item["href"]
            break

    ncx_href = None
    if ncx_id and ncx_id in manifest:
        ncx_href = manifest[ncx_id]["href"]
    else:
        for item in manifest.values():
            if item["media_type"] == NCX_MEDIA_TYPE:
                ncx_href = item["href"]
                break

    return title, manifest, spine, nav_href, ncx_href


def _parse_ncx(data: bytes, ncx_href: str) -> list[tuple[str, str]]:
    """Top-level EPUB 2 navPoints as (title, resolved href)."""

    root = _parse_xml(data)
    nav_map = next((element for element in root.iter() if _local(element.tag) == "navMap"), None)
    if nav_map is None:
        return []

    entries: list[tuple[str, str]] = []
    for nav_point in nav_map:
        if _local(nav_point.tag) != "navPoint":
            continue
        label = next(
            (
                "".join(element.itertext())
                for element in nav_point.iter()
                if _local(element.tag) == "text"
            ),
            "",
        )
        content = next(
            (
                element.get("src", "")
                for element in nav_point.iter()
                if _local(element.tag) == "content"
            ),
            "",
        )
        if label.strip() and content:
            entries.append((" ".join(label.split()), _resolve(ncx_href, content)))
    return entries


def _parse_nav_document(data: bytes, nav_href: str) -> list[tuple[str, str]]:
    """Top-level EPUB 3 ``nav[epub:type=toc]`` links as (title, resolved href)."""

    root = _parse_xml(data)
    navs = [element for element in root.iter() if _local(element.tag) == "nav"]
    toc_nav = next(
        (
            nav
            for nav in navs
            if any(key.endswith("type") and value == "toc" for key, value in nav.attrib.items())
        ),
        navs[0] if navs else None,
    )
    if toc_nav is None:
        return []

    ordered_list = next((child for child in toc_nav if _local(child.tag) == "ol"), None)
    if ordered_list is None:
        return []

    entries: list[tuple[str, str]] = []
    for item in ordered_list:
        if _local(item.tag) != "li":
            continue
        link = next((element for element in item.iter() if _local(element.tag) == "a"), None)
        if link is None:
            continue
        href = link.get("href")
        label = " ".join("".join(link.itertext()).split())
        if href and label:
            entries.append((label, _resolve(nav_href, href)))
    return entries


def _extract_documents(
    archive: zipfile.ZipFile,
    manifest: dict[str, dict[str, str]],
    spine: list[str],
) -> list[_Document]:
    documents: list[_Document] = []
    for idref in spine:
        item = manifest.get(idref)
        if item is None or item["media_type"] not in DOCUMENT_MEDIA_TYPES:
            continue
        data = _read_text(archive, item["href"])
        if data is None:
            continue
        extractor = _MarkdownExtractor()
        extractor.feed(data.decode("utf-8", errors="replace"))
        extractor.close()
        documents.append(
            _Document(href=item["href"], text=extractor.result(), anchors=extractor.anchors)
        )
    return documents


def _heading_title(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if heading:
                return heading
        elif stripped:
            break
    return None


def _chapter_text(
    documents: list[_Document],
    start: tuple[int, int],
    end: tuple[int, int] | None,
) -> str:
    """Join the document slices between two navigation positions."""

    last_index = end[0] if end is not None else len(documents) - 1
    pieces: list[str] = []
    for index in range(start[0], min(last_index, len(documents) - 1) + 1):
        text = documents[index].text
        begin = start[1] if index == start[0] else 0
        finish = end[1] if end is not None and index == end[0] else len(text)
        pieces.append(text[begin:finish])
    return RE_BLANK_LINES.sub("\n\n", "\n\n".join(pieces))


def _navigation_positions(
    entries: list[tuple[str, str]],
    documents: list[_Document],
) -> list[tuple[str, tuple[int, int]]]:
    """Place each navigation entry at a (document index, character offset).

    Entries pointing at content that is absent from the spine — or at an anchor
    the document never defines — are dropped rather than guessed at: a chapter
    that starts in the wrong place is harder to notice than one that is missing.
    """

    index_by_href = {document.href: index for index, document in enumerate(documents)}
    positions: list[tuple[str, tuple[int, int]]] = []
    for title, href in entries:
        path, _, fragment = href.partition("#")
        document_index = index_by_href.get(path)
        if document_index is None:
            continue
        offset = documents[document_index].anchors.get(fragment, 0) if fragment else 0
        positions.append((title, (document_index, offset)))
    return sorted(positions, key=lambda item: item[1])


def _fallback_chapters(documents: list[_Document]) -> list[tuple[str, str]]:
    """Without a navigation map, treat each spine document as a chapter."""

    chapters: list[tuple[str, str]] = []
    for index, document in enumerate(documents, start=1):
        content = clean_text_segment(RE_GUTENBERG_BOILERPLATE.sub("", document.text))
        if not content:
            continue
        chapters.append((_heading_title(document.text) or f"Section {index}", content))
    if not chapters:
        return []
    if len(chapters) == 1:
        return [("Audiobook", chapters[0][1])]
    return chapters


def parse_epub_to_chapters(epub_path: Path) -> list[tuple[str, str]]:
    """Parse an EPUB into chapters using its navigation map and spine order."""

    print(f"Parsing EPUB structure: {epub_path}...")
    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB not found: {epub_path}")
    if not zipfile.is_zipfile(epub_path):
        raise ValueError(f"Not a readable EPUB (expected a ZIP container): {epub_path}")

    with zipfile.ZipFile(epub_path) as archive:
        package_path = _find_package_path(archive)
        _title, manifest, spine, nav_href, ncx_href = _parse_package(archive, package_path)
        documents = _extract_documents(archive, manifest, spine)
        if not documents:
            raise ValueError(f"EPUB contains no readable spine documents: {epub_path}")

        entries: list[tuple[str, str]] = []
        if nav_href:
            data = _read_text(archive, nav_href)
            if data is not None:
                entries = _parse_nav_document(data, nav_href)
        if not entries and ncx_href:
            data = _read_text(archive, ncx_href)
            if data is not None:
                entries = _parse_ncx(data, ncx_href)

    positions = _navigation_positions(entries, documents)
    if not positions:
        print("No usable EPUB navigation map; falling back to spine documents.")
        chapters = _fallback_chapters(documents)
        print(f"Detected {len(chapters)} chapters from the EPUB spine.")
        return chapters

    chapters: list[tuple[str, str]] = []
    for entry_index, (title, start) in enumerate(positions):
        end = positions[entry_index + 1][1] if entry_index + 1 < len(positions) else None
        if is_skipped_section(title):
            continue
        raw = RE_GUTENBERG_BOILERPLATE.sub("", _chapter_text(documents, start, end))
        content = clean_text_segment(raw)
        if not has_narratable_body(content):
            continue
        # Keep the spoken chapter title in the text when the document itself
        # opens with it, the way the PDF backend does, so the narrated book and
        # the reviewable markdown agree on where a chapter begins.
        if content.casefold().startswith(title.casefold()) and not content.startswith("#"):
            content = "# " + content
        chapters.append((title, content))

    print(f"Detected {len(chapters)} chapters from the EPUB navigation map.")
    return chapters


__all__ = [
    "SKIPPED_SECTIONS",
    "parse_epub_to_chapters",
]
