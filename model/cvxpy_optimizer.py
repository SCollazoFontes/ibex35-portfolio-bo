"""
Optimización de cartera por máximo ratio de Sharpe mediante QP convexo (cvxpy).
Sustituto rápido y eficiente de la optimización bayesiana; misma interfaz de salida.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp

from .covariance import get_expected_returns_and_cov
from ._utils import _get_param


def modelo_OB_cvxpy(
    rendimientos_diarios_periodo: pd.DataFrame,
    risk_free_rate: float,
    peso_total: float = None,
    peso_conc_max: float = None,
    umbral: float = None,
    peso_min: float = None,
    peso_max: float = None,
    use_ledoit_wolf: bool = None,
) -> tuple:
    """
    Pesos óptimos maximizando Sharpe mediante programación cuadrática convexa.
    Respeta PESO_MIN y PESO_MAX de config (límite por activo); concentración con PESO_CONC_MAX y UMBRAL.
    """
    if rendimientos_diarios_periodo.empty:
        raise ValueError("El DataFrame de rendimientos diarios está vacío.")

    peso_total = peso_total if peso_total is not None else _get_param("PESO_TOTAL", 1.0)
    peso_conc_max = peso_conc_max if peso_conc_max is not None else _get_param("PESO_CONC_MAX", 1.0)
    umbral = umbral if umbral is not None else _get_param("UMBRAL", 1.0)
    peso_min = peso_min if peso_min is not None else _get_param("PESO_MIN", 0.0)
    peso_max = peso_max if peso_max is not None else _get_param("PESO_MAX", 1.0)

    mu, sigma = get_expected_returns_and_cov(
        rendimientos_diarios_periodo, annualize=252, use_ledoit_wolf=use_ledoit_wolf
    )
    tickers = rendimientos_diarios_periodo.columns.tolist()
    n_activos = len(tickers)

    mu_arr = np.asarray(mu, dtype=np.float64)
    Sigma = np.asarray(sigma, dtype=np.float64)
    # Asegurar que Sigma es PSD (por si acaso)
    Sigma = (Sigma + Sigma.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma += (1e-8 - min_eig) * np.eye(n_activos)

    exceso_ret = mu_arr - risk_free_rate

    # Max Sharpe: min w'Σw s.a. (μ-rf)'w=1, peso_min*sum(w) <= w <= peso_max*sum(w).
    # Tras normalizar w/sum(w), los pesos quedan en [peso_min, peso_max].
    w = cp.Variable(n_activos, nonneg=True)
    constraints = [exceso_ret @ w == 1]
    if peso_min > 0:
        constraints.append(w >= peso_min * cp.sum(w))
    if peso_max < 1.0:
        constraints.append(w <= peso_max * cp.sum(w))
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
    prob.solve(verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        w_val = np.ones(n_activos) / n_activos
    else:
        raw_val = w.value
        if raw_val is None or np.any(np.isnan(raw_val)):
            w_val = np.ones(n_activos) / n_activos
        else:
            w_val = np.asarray(raw_val).ravel()
            w_val = np.maximum(w_val, 0.0)
            total = w_val.sum()
            if total <= 0:
                w_val = np.ones(n_activos) / n_activos
            else:
                w_val = w_val / total

    # Restricción de concentración (opcional): si se excede peso_conc_max en pesos > umbral, reescalar
    if peso_conc_max < 1.0 and umbral < 1.0:
        grande = w_val > umbral
        if np.any(grande) and w_val[grande].sum() > peso_conc_max:
            otros = ~grande
            suma_grande = w_val[grande].sum()
            suma_otros = w_val[otros].sum()
            if suma_grande > 1e-12 and suma_otros >= 1e-12:
                w_val[grande] = peso_conc_max * (w_val[grande] / suma_grande)
                w_val[otros] = (1.0 - peso_conc_max) * (w_val[otros] / suma_otros)
            w_val = w_val / w_val.sum()

    rendimiento_ex_ante = float(np.dot(w_val, mu_arr))
    vol = np.sqrt(np.dot(w_val, np.dot(Sigma, w_val)))
    sharpe_opt = (rendimiento_ex_ante - risk_free_rate) / vol if vol > 1e-12 else 0.0

    pesos_tickers = pd.DataFrame({
        "Ticker": tickers,
        "Pesos": np.round(w_val, 4),
    })
    # Compatibilidad con backtest: evolucion = valores de la función objetivo (en bayesian: -Sharpe)
    evol_bayesiana = [-sharpe_opt]

    return pesos_tickers, sharpe_opt, rendimiento_ex_ante, evol_bayesiana
