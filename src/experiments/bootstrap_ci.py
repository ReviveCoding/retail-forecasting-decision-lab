from __future__ import annotations
import numpy as np
import pandas as pd


def bootstrap_mean_ci(values, n_bootstrap=200, seed=42, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(n_bootstrap)):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    return float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2))


def cluster_bootstrap_mean_ci(frame: pd.DataFrame, value_col: str, cluster_col: str, n_bootstrap=200, seed=42, alpha=0.05):
    """Bootstrap a mean while resampling clusters, not individual rows."""
    if frame.empty or value_col not in frame.columns or cluster_col not in frame.columns:
        return 0.0, 0.0
    clusters = frame[cluster_col].dropna().unique()
    if len(clusters) == 0:
        return bootstrap_mean_ci(frame[value_col].values, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha)
    rng = np.random.default_rng(seed)
    means = []
    grouped = {k: g[value_col].astype(float).values for k, g in frame.groupby(cluster_col)}
    for _ in range(int(n_bootstrap)):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_values = np.concatenate([grouped[c] for c in sampled_clusters if c in grouped])
        means.append(float(np.mean(sampled_values)))
    return float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2))
