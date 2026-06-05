
from pathlib import Path
import pandas as pd


def load_instacart_adapter(path: str | Path) -> pd.DataFrame:
    """Optional Instacart basket/reorder behavior adapter hook for v2."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('Instacart data not found. This adapter is optional in MVP v1.')
    return pd.read_csv(path)
