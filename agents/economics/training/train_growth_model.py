"""Train the growth/activity nowcast JPMaQS artifact."""

from __future__ import annotations

import argparse
import json

from agents.economics.training.train_common import train_model_type


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="jpmaqs_relevant_download")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    result = train_model_type(model_type="growth", data_dir=args.data_dir, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
