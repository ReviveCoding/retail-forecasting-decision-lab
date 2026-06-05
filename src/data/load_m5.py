from __future__ import annotations

from pathlib import Path
import zipfile
import pandas as pd


REQUIRED_M5_FILES = {
    "sales": "sales_train_evaluation.csv",
    "calendar": "calendar.csv",
    "prices": "sell_prices.csv",
}


def _find_file_in_dir(base_dir: Path, filename: str) -> Path:
    direct = base_dir / filename
    if direct.exists():
        return direct

    matches = list(base_dir.glob(f"**/{filename}"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Required M5 file not found under {base_dir}: {filename}")


def _find_file_in_zip(zf: zipfile.ZipFile, filename: str) -> str:
    names = zf.namelist()

    if filename in names:
        return filename

    matches = [name for name in names if name.endswith("/" + filename) or name.endswith(filename)]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Required M5 file not found in zip: {filename}")


def _resolve_format(m5_path: str | Path, m5_format: str = "auto") -> str:
    m5_path = Path(m5_path)

    if m5_format not in {"auto", "directory", "zip"}:
        raise ValueError("m5_format must be one of: auto, directory, zip")

    if m5_format != "auto":
        return m5_format

    if m5_path.is_dir():
        return "directory"

    if m5_path.is_file() and m5_path.suffix.lower() == ".zip":
        return "zip"

    raise FileNotFoundError(
        f"m5_path must be either an extracted M5 folder or a .zip file: {m5_path}"
    )


def _read_m5_csv(
    m5_path: str | Path,
    filename: str,
    m5_format: str = "auto",
    **kwargs,
) -> pd.DataFrame:
    m5_path = Path(m5_path)
    resolved_format = _resolve_format(m5_path, m5_format)

    if resolved_format == "directory":
        file_path = _find_file_in_dir(m5_path, filename)
        return pd.read_csv(file_path, **kwargs)

    if resolved_format == "zip":
        with zipfile.ZipFile(m5_path) as zf:
            member = _find_file_in_zip(zf, filename)
            with zf.open(member) as f:
                return pd.read_csv(f, **kwargs)

    raise ValueError(f"Unsupported M5 format: {resolved_format}")


def _read_filtered_prices(
    m5_path: str | Path,
    item_ids,
    store_ids,
    m5_format: str = "auto",
    chunksize: int = 500_000,
) -> pd.DataFrame:
    m5_path = Path(m5_path)
    resolved_format = _resolve_format(m5_path, m5_format)
    chunks = []

    if resolved_format == "directory":
        price_path = _find_file_in_dir(m5_path, REQUIRED_M5_FILES["prices"])
        reader = pd.read_csv(price_path, chunksize=chunksize)

        for chunk in reader:
            chunk = chunk[
                chunk["item_id"].isin(item_ids)
                & chunk["store_id"].isin(store_ids)
            ]
            if not chunk.empty:
                chunks.append(chunk)

    elif resolved_format == "zip":
        with zipfile.ZipFile(m5_path) as zf:
            member = _find_file_in_zip(zf, REQUIRED_M5_FILES["prices"])
            with zf.open(member) as f:
                for chunk in pd.read_csv(f, chunksize=chunksize):
                    chunk = chunk[
                        chunk["item_id"].isin(item_ids)
                        & chunk["store_id"].isin(store_ids)
                    ]
                    if not chunk.empty:
                        chunks.append(chunk)

    else:
        raise ValueError(f"Unsupported M5 format: {resolved_format}")

    if chunks:
        return pd.concat(chunks, ignore_index=True)

    return pd.DataFrame(columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])


def load_m5_panel(
    m5_path: str | Path,
    m5_format: str = "auto",
    n_items: int = 30,
    n_stores: int = 2,
    max_days: int = 420,
    sample_mode: bool = True,
) -> pd.DataFrame:
    m5_path = Path(m5_path)
    resolved_format = _resolve_format(m5_path, m5_format)

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    sales_header = _read_m5_csv(
        m5_path,
        REQUIRED_M5_FILES["sales"],
        m5_format=resolved_format,
        nrows=0,
    )

    all_d_cols = [c for c in sales_header.columns if c.startswith("d_")]
    d_cols = all_d_cols[-int(max_days):] if max_days else all_d_cols

    sales = _read_m5_csv(
        m5_path,
        REQUIRED_M5_FILES["sales"],
        m5_format=resolved_format,
        usecols=id_cols + d_cols,
    )

    calendar_cols = [
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]

    calendar = _read_m5_csv(
        m5_path,
        REQUIRED_M5_FILES["calendar"],
        m5_format=resolved_format,
        usecols=calendar_cols,
    )

    if sample_mode:
        store_ids = sorted(sales["store_id"].unique())[: int(n_stores)]
        item_ids = sorted(sales["item_id"].unique())[: int(n_items)]
        sales = sales[
            sales["store_id"].isin(store_ids)
            & sales["item_id"].isin(item_ids)
        ].copy()
    else:
        store_ids = sorted(sales["store_id"].unique())
        item_ids = sorted(sales["item_id"].unique())

    long_df = sales[id_cols + d_cols].melt(
        id_vars=id_cols,
        value_vars=d_cols,
        var_name="d",
        value_name="demand",
    )

    calendar = calendar[calendar_cols].copy()
    calendar["date"] = pd.to_datetime(calendar["date"])

    panel = long_df.merge(calendar, on="d", how="left")

    prices = _read_filtered_prices(
        m5_path,
        item_ids=panel["item_id"].unique(),
        store_ids=panel["store_id"].unique(),
        m5_format=resolved_format,
    )

    panel = panel.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    panel["sell_price"] = panel.groupby(["item_id", "store_id"])["sell_price"].transform(
        lambda s: s.ffill().bfill()
    )
    panel["sell_price"] = panel["sell_price"].fillna(panel["sell_price"].median()).fillna(0.0)

    panel["event_flag"] = panel["event_name_1"].notna().astype(int)
    panel["event_name_1"] = panel["event_name_1"].fillna("none")
    panel["event_type_1"] = panel["event_type_1"].fillna("none")

    panel["snap_flag"] = 0
    for state, col in [("CA", "snap_CA"), ("TX", "snap_TX"), ("WI", "snap_WI")]:
        mask = panel["state_id"].eq(state)
        panel.loc[mask, "snap_flag"] = panel.loc[mask, col].fillna(0).astype(int)

    panel["item_store_id"] = panel["item_id"] + "_" + panel["store_id"]
    panel = panel.sort_values(["item_store_id", "date"]).reset_index(drop=True)

    return panel