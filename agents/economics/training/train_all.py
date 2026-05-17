"""Train all small economics JPMaQS artifacts."""

from __future__ import annotations

import argparse
import json

from agents.economics.training.train_common import train_model_type


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="jpmaqs_relevant_download")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yield", "inflation", "growth", "policy"],
        choices=["yield", "inflation", "growth", "policy"],
    )
    args = parser.parse_args()
    results = {}
    for model_type in args.models:
        print(f"Training {model_type}...")
        results[model_type] = train_model_type(
            model_type=model_type,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
