"""Book-source extraction backends.

Every backend answers the same question — what are this book's chapters, and
what is the narratable text of each — so the workflow above only has to know
that a source file was chosen, not what format it happens to be in.
"""

from pathlib import Path

from .epub import parse_epub_to_chapters
from .pdf import parse_pdf_to_chapters
from .text import clean_text_segment

# Suffix → (parser, media type recorded in the prepared-book artifact).
_BACKENDS = {
    ".pdf": (parse_pdf_to_chapters, "application/pdf"),
    ".epub": (parse_epub_to_chapters, "application/epub+zip"),
}
SUPPORTED_SOURCE_SUFFIXES = tuple(_BACKENDS)


def _backend(source_path: Path):
    try:
        return _BACKENDS[source_path.suffix.lower()]
    except KeyError:
        supported = ", ".join(SUPPORTED_SOURCE_SUFFIXES)
        raise ValueError(
            f"Unsupported book format: {source_path.name} (supported: {supported})"
        ) from None


def parse_book_to_chapters(source_path: Path) -> list[tuple[str, str]]:
    """Extract chapters from a book file, choosing the backend by extension."""

    parser, _media_type = _backend(source_path)
    return parser(source_path)


def source_media_type(source_path: Path) -> str:
    """The media type recorded alongside a prepared book's source hash."""

    _parser, media_type = _backend(source_path)
    return media_type


__all__ = [
    "SUPPORTED_SOURCE_SUFFIXES",
    "clean_text_segment",
    "parse_book_to_chapters",
    "parse_epub_to_chapters",
    "parse_pdf_to_chapters",
    "source_media_type",
]
