"""
Estimación vectorizada de media y covarianza de rendimientos (muestral o Ledoit-Wolf).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

try:
    from .. import config
except ImportError:
    config = None


def _use_ledoit_wolf():
    if config is not None and hasattr(config, "USE_LEDOIT_WOLF"):
        return config.USE_LEDOIT_WOLF
    return True


def get_expected_returns_and_cov(
    rendimientos_diarios: pd.DataFrame,
    annualize: int = 252,
    use_ledoit_wolf: bool = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Media y covarianza anualizadas, vectorizado.
    use_ledoit_wolf: si True, covarianza con Ledoit-Wolf (más estable).
    """
    if rendimientos_diarios.empty:
        raise ValueError("El DataFrame de rendimientos está vacío.")

    use_lw = use_ledoit_wolf if use_ledoit_wolf is not None else _use_ledoit_wolf()
    tickers = rendimientos_diarios.columns.tolist()

    # Media: una operación vectorizada
    mu = rendimientos_diarios.mean() * annualize
    mu = mu.astype(np.float64, copy=False)

    if use_lw:
        # LedoitWolf espera (n_samples, n_features); nosotros tenemos (T, n_activos)
        X = rendimientos_diarios.to_numpy(dtype=np.float64, copy=False)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        lw = LedoitWolf().fit(X)
        cov_matrix = lw.covariance_ * annualize
        sigma = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)
    else:
        sigma = rendimientos_diarios.cov() * annualize

    return mu, sigma
