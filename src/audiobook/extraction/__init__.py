"""Book-source extraction backends."""

from .pdf import parse_pdf_to_chapters

__all__ = ["parse_pdf_to_chapters"]
