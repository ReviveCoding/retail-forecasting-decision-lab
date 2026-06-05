from __future__ import annotations
from pathlib import Path
import pandas as pd


def create_sql_mart(panel: pd.DataFrame, output_dir: str | Path) -> dict[str, str]:
    """Create lightweight SQL-ready marts without adding a database dependency.

    This MVP intentionally writes CSV marts that can be queried by DuckDB, SQLite,
    pandas, or BI tools. A real `.duckdb` database can be added later without
    changing the canonical table schemas.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demand_cols = [
        "item_store_id", "item_id", "store_id", "dept_id", "cat_id", "state_id",
        "date", "demand", "sell_price", "event_flag", "snap_flag",
    ]
    demand_mart = panel[[c for c in demand_cols if c in panel.columns]].copy()
    demand_path = output_dir / "demand_mart.csv"
    demand_mart.to_csv(demand_path, index=False)

    cohort = demand_mart.groupby(["cat_id", "store_id"], as_index=False).agg(
        n_rows=("demand", "size"),
        mean_demand=("demand", "mean"),
        total_demand=("demand", "sum"),
    )
    cohort_path = output_dir / "sql_cohort_report.csv"
    cohort.to_csv(cohort_path, index=False)
    return {"demand_mart": str(demand_path), "cohort_report": str(cohort_path)}
