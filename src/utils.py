
from __future__ import annotations
from pathlib import Path
import json
import yaml
import pandas as pd


def load_yaml(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dirs(*dirs: str | Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def write_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=str)


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # CSV is used for maximum portability in clean local environments.
    if path.suffix.lower() == '.parquet':
        csv_path = path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        return csv_path
    df.to_csv(path, index=False)
    return path


def safe_divide(n, d, default=0.0):
    return default if d == 0 else n / d
