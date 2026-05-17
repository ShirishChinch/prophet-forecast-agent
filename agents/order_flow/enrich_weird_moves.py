"""Generate LLM context features for suspicious historical order-flow rows.

This spends OpenAI credits only for rows that pass the weird-flow gate. The
output file can be consumed by `train_residual_model.py` automatically when it
is placed at `<data-dir>/llm_context_features.csv`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from agents.order_flow.features import build_order_flow_dataset
from agents.order_flow.llm_context import SuspiciousMove, extract_context_features, should_review_with_llm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/kalshi_order_flow_labeled")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-rows", type=int, default=50)
    parser.add_argument("--max-per-market", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out) if args.out else data_dir / "llm_context_features.csv"
    raw_path = out_path.with_suffix(".raw.jsonl")

    dataset = build_order_flow_dataset(data_dir)
    candidates = _candidate_rows(dataset.X, dataset.meta, args.max_rows, args.max_per_market)
    if args.dry_run:
        print(candidates[["market_ticker", "timestamp", "prior", "price_change_1", "price_change_24", "volume_zscore_24", "title"]].to_string(index=False))
        return

    rows: list[dict[str, Any]] = []
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for _, row in candidates.iterrows():
            move = SuspiciousMove(
                market_ticker=str(row["market_ticker"]),
                title=str(row.get("title") or ""),
                timestamp=str(row["timestamp"]),
                prior=float(row["prior"]),
                price_change_1=float(row.get("price_change_1") or 0.0),
                price_change_24=float(row.get("price_change_24") or 0.0),
                volume_zscore_24=float(row.get("volume_zscore_24") or 0.0),
                trade_size=float(row.get("volume") or 0.0),
                trade_size_percentile=float(row.get("trade_size_percentile") or 0.0),
                category=str(row.get("category") or row.get("template_family") or ""),
            )
            features, payload = extract_context_features(move)
            output_row = {
                "market_ticker": move.market_ticker,
                "timestamp": move.timestamp,
                **features,
            }
            rows.append(output_row)
            raw_handle.write(json.dumps({"row": output_row, "payload": payload}, ensure_ascii=True, default=str))
            raw_handle.write("\n")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} LLM context rows to {out_path}")


def _candidate_rows(X: pd.DataFrame, meta: pd.DataFrame, max_rows: int, max_per_market: int) -> pd.DataFrame:
    frame = X.reset_index(drop=True).join(meta.reset_index(drop=True))
    if "volume" in frame.columns:
        frame["trade_size_percentile"] = frame["volume"].rank(pct=True)
    else:
        frame["trade_size_percentile"] = 0.0
    mask = frame.apply(lambda row: should_review_with_llm(row.to_dict()), axis=1)
    candidates = frame[mask].copy()
    candidates["suspicion_score"] = (
        candidates.get("price_change_1", 0.0).abs()
        + 0.5 * candidates.get("price_change_24", 0.0).abs()
        + 0.02 * candidates.get("volume_zscore_24", 0.0).clip(lower=0.0)
        + candidates.get("trade_size_percentile", 0.0)
    )
    candidates = candidates.sort_values("suspicion_score", ascending=False)
    candidates = candidates.drop_duplicates(["market_ticker", "timestamp"], keep="first")
    if max_per_market > 0:
        candidates = candidates.groupby("market_ticker", group_keys=False).head(max_per_market)
    return candidates.head(max_rows)


if __name__ == "__main__":
    main()
