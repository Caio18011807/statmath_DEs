#!/usr/bin/env python3
"""
Command-Line Interface (CLI) for StatVision OCR.

Usage examples:
  python3 cli.py --generate-samples
  python3 cli.py --image samples/normal_iid_sample.png
  python3 cli.py --image path/to/photo.png --json
  python3 cli.py --list-distributions
"""

import argparse
import dataclasses
import json
import os
import sys

from stat_vision.analyzer import StatisticalImageAnalyzer
from stat_vision.knowledge_base import DISTRIBUTIONS_DB
from stat_vision.sample_generator import generate_all_demo_samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="StatVision OCR: Offline Statistical Distribution & Property Image Analyzer"
    )
    parser.add_argument(
        "-i", "--image",
        type=str,
        help="Path to the image of a statistical distribution plot or formula.",
    )
    parser.add_argument(
        "--hint",
        type=str,
        default="",
        help="Optional text/formula hint to supplement OCR.",
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate synthetic sample images of distributions in ./samples/",
    )
    parser.add_argument(
        "--list-distributions",
        action="store_true",
        help="List all distributions in the symbolic knowledge base.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the full analysis report as JSON instead of Markdown.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="",
        help="Optional file path to save the Markdown or JSON report.",
    )

    args = parser.parse_args()

    if args.list_distributions:
        print("Supported Distributions in Symbolic Knowledge Base:")
        print("=" * 60)
        for dist_id, spec in DISTRIBUTIONS_DB.items():
            print(f"- [{dist_id:12s}] {spec.name} ({spec.family}) | Support: {spec.support}")
        return 0

    if args.generate_samples:
        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
        paths = generate_all_demo_samples(sample_dir)
        print(f"Generated {len(paths)} sample distribution images in: {sample_dir}")
        for dist_id, p in paths.items():
            print(f"  - {dist_id}: {p}")
        if not args.image:
            return 0

    if not args.image:
        parser.print_help()
        return 1

    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        return 2

    analyzer = StatisticalImageAnalyzer()
    report = analyzer.analyze(args.image, extra_text_hint=args.hint)

    if args.json:
        output_text = json.dumps(dataclasses.asdict(report), indent=2, ensure_ascii=False)
    else:
        output_text = analyzer.format_markdown_report(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Saved report to {args.output}")
    else:
        print(output_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
