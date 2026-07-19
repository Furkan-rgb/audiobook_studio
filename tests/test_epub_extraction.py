import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from audiobook.extraction import (
    parse_book_to_chapters,
    source_media_type,
)
from audiobook.extraction.epub import parse_epub_to_chapters


CONTAINER = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf"
    media-type="application/oebps-package+xml"/></rootfiles>
</container>
"""


def _package(items, spine, *, nav_property=False, ncx=False) -> str:
    manifest = "".join(
        f'<item id="{item_id}" href="{href}" media-type="{media_type}"'
        + (' properties="nav"' if nav_property and item_id == "nav" else "")
        + "/>"
        for item_id, href, media_type in items
    )
    itemrefs = "".join(
        f'<itemref idref="{idref}"' + (f' linear="{linear}"' if linear else "") + "/>"
        for idref, linear in spine
    )
    toc_attribute = ' toc="ncx"' if ncx else ""
    return f"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Fixture Book</dc:title>
  </metadata>
  <manifest>{manifest}</manifest>
  <spine{toc_attribute}>{itemrefs}</spine>
</package>
"""


def _write_epub(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", CONTAINER)
        for name, content in files.items():
            archive.writestr(name, content)
    return path


class EpubExtractionTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.directory = Path(self._temporary.name)

    def test_navigation_document_splits_chapters_at_their_anchors(self):
        """Two chapters in one file are split where the navigation says."""

        body = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
          <h2 id="one">Chapter One</h2>
          <p>The first chapter opens quietly.</p>
          <h2 id="two">Chapter Two</h2>
          <p>The second chapter answers it.</p>
        </body></html>"""
        nav = """<html xmlns="http://www.w3.org/1999/xhtml"
            xmlns:epub="http://www.idpf.org/2007/ops"><body>
          <nav epub:type="toc"><ol>
            <li><a href="body.xhtml#one">Chapter One</a></li>
            <li><a href="body.xhtml#two">Chapter Two</a></li>
          </ol></nav>
        </body></html>"""
        path = _write_epub(
            self.directory / "anchors.epub",
            {
                "OEBPS/content.opf": _package(
                    [
                        ("body", "body.xhtml", "application/xhtml+xml"),
                        ("nav", "nav.xhtml", "application/xhtml+xml"),
                    ],
                    [("body", None)],
                    nav_property=True,
                ),
                "OEBPS/body.xhtml": body,
                "OEBPS/nav.xhtml": nav,
            },
        )

        chapters = parse_epub_to_chapters(path)

        self.assertEqual([title for title, _ in chapters], ["Chapter One", "Chapter Two"])
        self.assertEqual(
            chapters[0][1], "## Chapter One\n\nThe first chapter opens quietly."
        )
        self.assertEqual(
            chapters[1][1], "## Chapter Two\n\nThe second chapter answers it."
        )

    def test_ncx_navigation_spans_documents_and_skips_apparatus(self):
        """EPUB 2 books keep spine order, and a contents page is not narrated."""

        def document(heading: str, sentence: str) -> str:
            return (
                '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
                f"<h1>{heading}</h1><p>{sentence}</p></body></html>"
            )

        ncx = """<?xml version="1.0"?>
        <ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><navMap>
          <navPoint id="a"><navLabel><text>Contents</text></navLabel>
            <content src="toc.xhtml"/></navPoint>
          <navPoint id="b"><navLabel><text>Chapter One</text></navLabel>
            <content src="one.xhtml"/></navPoint>
          <navPoint id="c"><navLabel><text>Chapter Two</text></navLabel>
            <content src="two.xhtml"/></navPoint>
        </navMap></ncx>"""
        path = _write_epub(
            self.directory / "ncx.epub",
            {
                "OEBPS/content.opf": _package(
                    [
                        ("toc", "toc.xhtml", "application/xhtml+xml"),
                        ("one", "one.xhtml", "application/xhtml+xml"),
                        ("two", "two.xhtml", "application/xhtml+xml"),
                        ("extra", "extra.xhtml", "application/xhtml+xml"),
                        ("ncx", "toc.ncx", "application/x-dtbncx+xml"),
                    ],
                    [
                        ("toc", None),
                        ("one", None),
                        ("two", None),
                        ("extra", None),
                    ],
                    ncx=True,
                ),
                "OEBPS/toc.xhtml": document("Contents", "Chapter One. Chapter Two."),
                "OEBPS/one.xhtml": document("Chapter One", "The first chapter."),
                "OEBPS/two.xhtml": document("Chapter Two", "The second chapter."),
                "OEBPS/extra.xhtml": document("Afterword", "A closing note."),
                "OEBPS/toc.ncx": ncx,
            },
        )

        chapters = parse_epub_to_chapters(path)

        self.assertEqual([title for title, _ in chapters], ["Chapter One", "Chapter Two"])
        # An unlisted trailing document belongs to the chapter it follows.
        self.assertIn("A closing note.", chapters[-1][1])

    def test_spine_documents_are_chapters_without_a_navigation_map(self):
        def document(sentence: str) -> str:
            return (
                '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
                f"<p>{sentence}</p></body></html>"
            )

        path = _write_epub(
            self.directory / "bare.epub",
            {
                "OEBPS/content.opf": _package(
                    [
                        ("one", "one.xhtml", "application/xhtml+xml"),
                        ("two", "two.xhtml", "application/xhtml+xml"),
                        ("cover", "cover.xhtml", "application/xhtml+xml"),
                    ],
                    [("one", None), ("two", None), ("cover", "no")],
                ),
                "OEBPS/one.xhtml": document("The first section."),
                "OEBPS/two.xhtml": document("The second section."),
                "OEBPS/cover.xhtml": document("Cover image caption."),
            },
        )

        chapters = parse_epub_to_chapters(path)

        self.assertEqual([content for _, content in chapters],
                         ["The first section.", "The second section."])
        # linear="no" material is shown out of band by readers, and skipped here.
        self.assertNotIn("Cover image caption.", "".join(c for _, c in chapters))

    def test_markup_becomes_narratable_paragraphs(self):
        body = """<html xmlns="http://www.w3.org/1999/xhtml"><head>
          <title>Ignored</title><style>p { color: red }</style></head><body>
          <h1 id="start">A Title</h1>
          <p>A line<br/>broken by markup.</p>
          <p>Emphasis is <em>inline</em> and <a href="x.xhtml">links read</a>
             as their text.</p>
          <script>ignored()</script>
        </body></html>"""
        nav = """<html xmlns="http://www.w3.org/1999/xhtml"
            xmlns:epub="http://www.idpf.org/2007/ops"><body>
          <nav epub:type="toc"><ol>
            <li><a href="body.xhtml#start">A Title</a></li>
          </ol></nav></body></html>"""
        path = _write_epub(
            self.directory / "markup.epub",
            {
                "OEBPS/content.opf": _package(
                    [
                        ("body", "body.xhtml", "application/xhtml+xml"),
                        ("nav", "nav.xhtml", "application/xhtml+xml"),
                    ],
                    [("body", None)],
                    nav_property=True,
                ),
                "OEBPS/body.xhtml": body,
                "OEBPS/nav.xhtml": nav,
            },
        )

        (_title, content), = parse_epub_to_chapters(path)

        self.assertNotIn("ignored()", content)
        self.assertNotIn("color: red", content)
        self.assertIn("Emphasis is inline and links read as their text.", content)
        self.assertIn("A line\nbroken by markup.", content)

    def test_unsupported_and_unreadable_sources_are_reported(self):
        with self.assertRaises(ValueError) as unsupported:
            parse_book_to_chapters(Path("book.mobi"))
        self.assertIn("Unsupported book format", str(unsupported.exception))

        not_an_epub = self.directory / "broken.epub"
        not_an_epub.write_text("plain text, not a container")
        with self.assertRaises(ValueError):
            parse_book_to_chapters(not_an_epub)

    def test_media_type_follows_the_extension(self):
        self.assertEqual(source_media_type(Path("book.pdf")), "application/pdf")
        self.assertEqual(source_media_type(Path("BOOK.EPUB")), "application/epub+zip")


if __name__ == "__main__":
    unittest.main()
