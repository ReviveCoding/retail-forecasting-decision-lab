
from __future__ import annotations
import numpy as np


def approximate_mde(values, alpha_z=1.96, power_z=0.84):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float((alpha_z + power_z) * np.std(values, ddof=1) / np.sqrt(len(values)))
