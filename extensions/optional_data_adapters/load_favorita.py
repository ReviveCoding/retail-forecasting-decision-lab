
from pathlib import Path
import pandas as pd


def load_favorita_adapter(path: str | Path) -> pd.DataFrame:
    """Optional adapter hook. Provide Kaggle Favorita files to activate this path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('Favorita data not found. This adapter is intentionally optional in MVP v1.')
    return pd.read_csv(path)
