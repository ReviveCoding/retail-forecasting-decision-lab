# Data Quality Report

## What was tested

Loaded public M5 sales, calendar, and sell-price data into an item-store-day panel.

## Baseline

Raw M5 files are wide and not directly ready for model training.

## Enhanced method

The pipeline validates schema, date coverage, duplicate keys, and creates SQL-ready marts.

## Key results

Schema report:

```json
{'n_rows': 25200, 'n_series': 60, 'missing_required_columns': [], 'duplicate_item_store_date_keys': 0, 'date_gap_count': 0, 'null_counts_required': {'item_id': 0, 'dept_id': 0, 'cat_id': 0, 'store_id': 0, 'state_id': 0, 'demand': 0, 'date': 0, 'sell_price': 0, 'event_flag': 0, 'snap_flag': 0, 'item_store_id': 0}, 'schema_pass': True}
```

Split info:

```json
{'train_end': '2016-03-27', 'val_start': '2016-03-28', 'test_start': '2016-04-25', 'test_end': '2016-05-22'}
```

## Interpretation

The panel passed the checks required for the sample-mode forecasting pipeline.

## Claim boundary

This does not validate Amazon internal data or real replenishment systems.
