#!/usr/bin/env python3
"""
CLI tool to visualize iPhone ARKit capture sessions.

Usage:
    python -m stack.scripts.visualize_iphone <session_path>
    python -m stack.scripts.visualize_iphone <session_path> --save

Examples:
    # Interactive viewing
    python -m stack.scripts.visualize_iphone data/raw/session_2026-02-01_142003

    # Save outputs to session_path/viz/
    python -m stack.scripts.visualize_iphone data/raw/session_2026-02-01_142003 --save
"""

import argparse
import sys
from pathlib import Path

from stack.viz.iphone_viz import visualize_session


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize iPhone ARKit capture sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "session_path",
        type=str,
        help="Path to session directory (e.g., data/raw/session_2026-02-01_142003)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save visualizations to session_path/viz/ instead of interactive display",
    )

    args = parser.parse_args()

    session_path = Path(args.session_path)
    if not session_path.exists():
        print(f"Error: Session not found: {session_path}")
        return 1

    visualize_session(str(session_path), save=args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
