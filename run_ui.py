"""Serve the local audiobook frontend.

    python run_ui.py
    python run_ui.py --port 7861 --share
"""

import argparse
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from audiobook.preflight import format_report, passed, run_preflight
from audiobook.ui import launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Expose the app through a public Gradio tunnel.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Start even when preflight checks fail.",
    )
    args = parser.parse_args()

    print("Preflight:")
    results = run_preflight()
    print(format_report(results))
    if not passed(results) and not args.skip_preflight:
        raise SystemExit(
            "\nFix the failures above, or start anyway with --skip-preflight."
        )

    launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
