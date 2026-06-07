"""CLI entry point — dispatches to pipeline scripts."""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cattle-weight",
        description="Cattle weight estimation pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="Extract and preprocess the raw data archive")
    sub.add_parser("train", help="Train a weight estimation model")
    sub.add_parser("evaluate", help="Evaluate a trained model on the test set")
    sub.add_parser("predict", help="Run inference on a single image or directory")

    args = parser.parse_args()

    if args.command == "prepare":
        from scripts.prepare_data import run
        run()
    elif args.command == "train":
        from scripts.train import run
        run()
    elif args.command == "evaluate":
        from scripts.evaluate import run
        run()
    else:
        parser.print_help()
        sys.exit(1)
