"""CLI interface for ImageTagService.

Usage:
    app.exe image.jpg
    app.exe image.jpg --tags "cat,dog,sky"
    app.exe image.jpg --tags "cat,dog" --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_cli(args: list[str]) -> int:
    """
    Entry point for CLI mode.  Returns exit code (0 = success, 1 = error).
    """
    # Import here so that model loading errors don't surface before argparse
    from model import tag_image_with_hash
    from utils import image_hash, validate_image

    parser = argparse.ArgumentParser(
        prog="ImageTagService",
        description="Tag an image using the Recognize Anything Model.",
    )
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument(
        "--tags",
        default="",
        metavar="TAG1,TAG2,...",
        help="Comma-separated list of known/preferred tags to match against.",
    )
    parser.add_argument(
        "--all",
        dest="show_all",
        action="store_true",
        help="Print all RAM tags instead of the filtered/suggested list.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv).",
    )

    parsed = parser.parse_args(args)

    image_path = Path(parsed.image)
    if not image_path.exists():
        print(f"Error: File not found: {image_path}", file=sys.stderr)
        return 1

    try:
        data = image_path.read_bytes()
    except OSError as exc:
        print(f"Error reading file: {exc}", file=sys.stderr)
        return 1

    try:
        pil_image = validate_image(data)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    known_tags = [t.strip() for t in parsed.tags.split(",") if t.strip()]

    try:
        result = tag_image_with_hash(pil_image, image_hash(data), known_tags or None)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if parsed.show_all:
        output_tags = result["all_tags"]
    else:
        # Prefer matched known tags; fall back to RAM suggestions
        output_tags = result["matched_tags"] or result["suggested_tags"]

    if parsed.format == "json":
        import json
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(", ".join(output_tags))

    return 0
