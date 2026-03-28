"""
Módulo de modelo: ratio de Sharpe, estimación de covarianza y optimizadores (cvxpy / bayesian).
"""
from .sharpe import sharpe_ratio
from .covariance import get_expected_returns_and_cov
from .bayesian import modelo_OB_bayesian

try:
    from .cvxpy_optimizer import modelo_OB_cvxpy
except ImportError:
    modelo_OB_cvxpy = None  # pip install cvxpy para usar optimizador rápido

from ._utils import _get_param

try:
    from .. import config
except ImportError:
    config = None


def modelo_OB(rendimientos_diarios_periodo, risk_free_rate, **kwargs):
    """
    Pesos óptimos maximizando Sharpe. Usa el optimizador configurado en config.OPTIMIZER:
    - "cvxpy": QP convexo (rápido, recomendado). Requiere pip install cvxpy.
    - "bayesian": gp_minimize (lento, para comparar con resultados antiguos).
    """
    optimizer = (config.OPTIMIZER if config is not None and hasattr(config, "OPTIMIZER") else "cvxpy")
    if optimizer == "bayesian" or modelo_OB_cvxpy is None:
        return modelo_OB_bayesian(rendimientos_diarios_periodo, risk_free_rate, **kwargs)
    return modelo_OB_cvxpy(rendimientos_diarios_periodo, risk_free_rate, **kwargs)


__all__ = [
    "sharpe_ratio",
    "get_expected_returns_and_cov",
    "modelo_OB",
    "modelo_OB_bayesian",
    "modelo_OB_cvxpy",
]
