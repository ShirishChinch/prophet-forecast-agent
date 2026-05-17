# Economics JPMaQS Model Layer

This package adds a compact economics modeling layer for Kalshi-style binary
events. The layer is intentionally conservative: it uses the Kalshi midpoint as
the main prior and only adds a JPMaQS model output when relevant point-in-time
features are available.

## Files

- `event_parser.py`: parses economics events into variable, country, threshold,
  bucket, and model type.
- `feature_families.py`: manual feature-family whitelist by model type. This is
  where we avoid feeding all 26k JPMaQS features to a model.
- `jpmaqs_store.py`: lightweight local point-in-time feature store facade. It
  looks for `JPMAQS_DATA_DIR` or `jpmaqs_full_download`.
- `feature_selection.py`: runtime feature cap plus documented train-time
  selection policy.
- `inflation_nowcast_model.py`: CPI/Core CPI release buckets.
- `growth_nowcast_model.py`: GDP, housing, activity releases.
- `yield_threshold_model.py`: rate/yield threshold events.
- `policy_decision_model.py`: cut/hold/hike central-bank decisions.
- `model_router.py`: routes one parsed event to the right model.

## Current Behavior

If JPMaQS parquet data is not present, models return the market prior with a
low-confidence explanation. This prevents accidental fake intelligence and keeps
Prophet predictions valid.

## Data Directory

The smaller recommended download folder is `jpmaqs_relevant_download`.
The older full-archive folder name `jpmaqs_full_download` is also supported.

Set this if you want to be explicit:

```powershell
$env:JPMAQS_DATA_DIR="D:\path\to\jpmaqs_relevant_download"
```

Expected useful files:

- `jpmaqs_catalogue.csv` or `catalogue.csv`
- `jpmaqs_chunk_*.parquet`

The existing notebook `jpmaqs-yield-changes-full-pipeline_final.ipynb` is the
right place to generate those files.

## Training Policy

For each model:

1. Filter by country/region.
2. Filter by model-specific feature families.
3. Prefer fast-moving features for announcement-style forecasts.
4. Use only point-in-time values available as of the forecast date.
5. Drop sparse/stale series inside each training fold.
6. Select features only on training folds.
7. Keep stable features across folds.
8. Cap features:
   - inflation: 120
   - growth: 120
   - yield: 80
   - policy: 80

The next step is adding trained artifacts from rolling backtests. Until then,
the JPMaQS models are conservative adapters, not aggressive predictors.
