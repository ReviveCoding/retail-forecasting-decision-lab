from __future__ import annotations
from pathlib import Path
import pandas as pd


def write_markdown(path: str | Path, title: str, body: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n\n{body.strip()}\n")


def df_head_markdown(df: pd.DataFrame, n=10) -> str:
    if df is None or df.empty:
        return '_No rows generated._'
    return df.head(n).to_markdown(index=False)


def report_section(tested: str, baseline: str, enhanced: str, result: str, interpretation: str, boundary: str) -> str:
    return f"""
## What was tested

{tested}

## Baseline

{baseline}

## Enhanced method

{enhanced}

## Key results

{result}

## Interpretation

{interpretation}

## Claim boundary

{boundary}
"""
