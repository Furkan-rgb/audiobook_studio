"""Compatibility launcher for the installed :mod:`audiobook` package.

New code should import from ``audiobook`` and invoke the ``audiobook`` console
command. This file remains so existing ``python make_audiobook.py`` commands
and imports continue to work from a source checkout.
"""

from pathlib import Path
import sys


_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Preserve the former monolithic module's public and test-helper surface while
# keeping all implementation inside the package.
from audiobook.cli import *  # noqa: F401,F403,E402
from audiobook.cli import (  # noqa: F401,E402
    _context_head,
    _context_tail,
    _crossfade,
    _fade_in,
    _fade_out,
    _join_markdown_pages,
    _join_units,
    _make_text_units,
    _normalize_paragraph,
    main,
    parse_args,
)


if __name__ == "__main__":
    main()
