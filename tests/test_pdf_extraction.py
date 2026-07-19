import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audiobook.extraction.pdf import (
    _select_bookmarks,
    _split_markdown_headings,
    parse_pdf_to_chapters,
)


BODY = " ".join(["The keeper trimmed the wick and watched the water."] * 6)


def _write_pdf(path: Path, sections, toc=None) -> Path:
    """Write a one-page-per-section PDF, headings set in visibly larger type."""

    import pymupdf

    document = pymupdf.open()
    for title, body in sections:
        page = document.new_page()
        if title:
            page.insert_text((72, 96), title, fontsize=24, fontname="hebo")
        page.insert_textbox(pymupdf.Rect(72, 130, 520, 700), body, fontsize=11)
    if toc is not None:
        document.set_toc(toc)
    document.save(path)
    document.close()
    return path


class BookmarkSelectionTests(unittest.TestCase):
    def test_numbered_bookmarks_win_and_parts_only_bound(self):
        narrated, structural, source = _select_bookmarks(
            [
                (1, "I  Beginnings", 3),
                (2, "1 The First Thing", 5),
                (2, "2 The Second Thing", 21),
            ]
        )

        self.assertEqual(source, "bookmarks")
        self.assertEqual(narrated, [("1 The First Thing", 5), ("2 The Second Thing", 21)])
        self.assertIn(("I Beginnings", 3), structural)

    def test_unnumbered_outline_becomes_chapters(self):
        narrated, structural, source = _select_bookmarks(
            [
                (1, "Cover", 1),
                (1, "The Lighthouse", 4),
                (1, "The Storm", 30),
                (2, "A subsection", 33),
            ]
        )

        self.assertEqual(source, "outline")
        self.assertEqual(narrated, [("The Lighthouse", 4), ("The Storm", 30)])
        # Apparatus is dropped from narration but still bounds what precedes it.
        self.assertEqual(structural[0], ("Cover", 1))

    def test_single_top_level_entry_is_not_treated_as_structure(self):
        narrated, _structural, source = _select_bookmarks([(1, "Fixture Book", 1)])

        self.assertEqual(source, "none")
        self.assertEqual(narrated, [])

    def test_entries_without_a_destination_are_ignored(self):
        narrated, _structural, source = _select_bookmarks(
            [(1, "Dangling", -1), (1, "Real One", 2), (1, "Real Two", 9)]
        )

        self.assertEqual(source, "outline")
        self.assertEqual(narrated, [("Real One", 2), ("Real Two", 9)])


class HeadingFallbackTests(unittest.TestCase):
    def test_shallowest_heading_level_splits_the_book(self):
        markdown = (
            "# The Lighthouse\n\nIt stood alone.\n\n"
            "## A digression\n\nStill the lighthouse.\n\n"
            "# The Storm\n\nThen the weather turned.\n"
        )

        chapters = _split_markdown_headings(markdown)

        self.assertEqual([title for title, _ in chapters], ["The Lighthouse", "The Storm"])
        self.assertIn("A digression", chapters[0][1])
        self.assertTrue(chapters[1][1].endswith("Then the weather turned."))

    def test_apparatus_and_bodyless_headings_are_dropped(self):
        markdown = (
            "# Contents\n\nThe Lighthouse ... 1\n\n"
            "# Part One\n\n"
            "# The Lighthouse\n\nIt stood alone.\n"
        )

        chapters = _split_markdown_headings(markdown)

        self.assertEqual([title for title, _ in chapters], ["The Lighthouse"])

    def test_a_lone_heading_is_not_structure(self):
        self.assertEqual(_split_markdown_headings("# Only One\n\nAll the text.\n"), [])
        self.assertEqual(_split_markdown_headings("No headings at all.\n"), [])


class UnnumberedPdfTests(unittest.TestCase):
    """The whole ladder, against PDFs that number nothing."""

    SECTIONS = [
        ("Contents", "The Lighthouse 1\nThe Storm 2"),
        ("The Lighthouse", BODY),
        ("The Storm", BODY),
    ]

    def test_unnumbered_outline_yields_chapters(self):
        with TemporaryDirectory() as directory:
            path = _write_pdf(
                Path(directory) / "outline.pdf",
                self.SECTIONS,
                toc=[[1, "Contents", 1], [1, "The Lighthouse", 2], [1, "The Storm", 3]],
            )
            chapters = parse_pdf_to_chapters(path)

        self.assertEqual([title for title, _ in chapters], ["The Lighthouse", "The Storm"])
        self.assertIn("trimmed the wick", chapters[0][1])
        # The table of contents bounds the book without being narrated.
        self.assertNotIn("The Lighthouse 1", chapters[0][1])

    def test_no_outline_falls_back_to_headings(self):
        with TemporaryDirectory() as directory:
            path = _write_pdf(Path(directory) / "headings.pdf", self.SECTIONS[1:])
            chapters = parse_pdf_to_chapters(path)

        self.assertEqual([title for title, _ in chapters], ["The Lighthouse", "The Storm"])
        self.assertIn("trimmed the wick", chapters[1][1])

    def test_no_structure_at_all_yields_one_chapter(self):
        with TemporaryDirectory() as directory:
            path = _write_pdf(Path(directory) / "flat.pdf", [("", BODY)])
            chapters = parse_pdf_to_chapters(path)

        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0][0], "Audiobook")
        self.assertIn("trimmed the wick", chapters[0][1])


if __name__ == "__main__":
    unittest.main()
