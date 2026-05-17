"""CLI for LLM-assisted feature discovery by event template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.evidence.catalog import get_default_catalog
from agents.evidence.discovery import discover_feature_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="all")
    parser.add_argument("--out-dir", default="reports/evidence_specs")
    args = parser.parse_args()

    catalog = get_default_catalog()
    templates = list(catalog) if args.template == "all" else [args.template]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for template in templates:
        spec = discover_feature_spec(template)
        out_path = out_dir / f"{template}.json"
        out_path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

