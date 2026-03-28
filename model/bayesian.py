"""
Optimización Bayesiana de cartera (maximización del ratio de Sharpe) con gp_minimize.
Más lenta que cvxpy; usar para comparar resultados antiguos vs nuevos.
"""
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from .sharpe import sharpe_ratio
from .covariance import get_expected_returns_and_cov
from ._utils import _get_param


def modelo_OB_bayesian(
    rendimientos_diarios_periodo: pd.DataFrame,
    risk_free_rate: float,
    peso_total: float = None,
    peso_conc_max: float = None,
    umbral: float = None,
    n_calls: int = None,
    n_initial_points: int = None,
    random_state: int = None,
    paciencia: int = None,
    tolerancia: float = None,
    use_ledoit_wolf: bool = None,
):
    """
    Pesos óptimos mediante Optimización Bayesiana (gp_minimize). Misma interfaz que modelo_OB.
    """
    if rendimientos_diarios_periodo.empty:
        raise ValueError("El DataFrame de rendimientos diarios está vacío.")

    peso_total = peso_total if peso_total is not None else _get_param("PESO_TOTAL", 1.0)
    peso_conc_max = peso_conc_max if peso_conc_max is not None else _get_param("PESO_CONC_MAX", 1.0)
    umbral = umbral if umbral is not None else _get_param("UMBRAL", 1.0)
    n_calls = n_calls if n_calls is not None else _get_param("OB_N_CALLS", 400)
    n_initial_points = n_initial_points if n_initial_points is not None else _get_param("OB_N_INITIAL_POINTS", 70)
    random_state = random_state if random_state is not None else _get_param("OB_RANDOM_STATE", 4)
    paciencia = paciencia if paciencia is not None else _get_param("OB_PACIENCIA", 100)
    tolerancia = tolerancia if tolerancia is not None else _get_param("OB_TOLERANCIA", 1e-4)

    promedio_rendimientos, matriz_covarianzas = get_expected_returns_and_cov(
        rendimientos_diarios_periodo, annualize=252, use_ledoit_wolf=use_ledoit_wolf
    )
    tickers = rendimientos_diarios_periodo.columns.tolist()
    n_activos = len(tickers)

    espacio_pesos = [Real(0.0, 1.0, name=f"w{i}") for i in range(n_activos)]

    restricciones = [
        {"type": "eq", "fun": lambda p: np.sum(p) - peso_total},
        {"type": "ineq", "fun": lambda p: peso_conc_max - np.sum(p[p > umbral])},
    ]

    @use_named_args(espacio_pesos)
    def objetivo(**pesos_dict):
        try:
            pesos = np.array(list(pesos_dict.values()))
            pesos /= np.sum(pesos)

            penalizacion = 0.0
            for r in restricciones:
                v = r["fun"](pesos)
                if r["type"] == "eq":
                    penalizacion += 1e3 * v**2
                elif v < 0:
                    penalizacion += 1e3 * v**2

            valor = sharpe_ratio(
                pesos, promedio_rendimientos.values, np.asarray(matriz_covarianzas), risk_free_rate
            ) + penalizacion

            if np.isnan(valor) or np.isinf(valor):
                return 1e8
            return float(valor)
        except Exception:
            return 1e8

    class ParadaSinMejora:
        def __init__(self, paciencia=paciencia, tolerancia=tolerancia):
            self.paciencia = paciencia
            self.tolerancia = tolerancia
            self.mejor = np.inf
            self.espera = 0

        def __call__(self, res):
            if res.fun < self.mejor - self.tolerancia:
                self.mejor = res.fun
                self.espera = 0
            else:
                self.espera += 1
            return self.espera >= self.paciencia

    stopper = ParadaSinMejora(paciencia=paciencia, tolerancia=tolerancia)

    resultado = gp_minimize(
        objetivo,
        espacio_pesos,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_state,
        callback=[stopper],
        verbose=False,
    )

    pesos_optimos = np.array(resultado.x)
    pesos_optimos /= np.sum(pesos_optimos)

    rendimiento_ex_ante = float(np.dot(pesos_optimos, promedio_rendimientos.values))

    pesos_tickers = pd.DataFrame({
        "Ticker": tickers,
        "Pesos": np.round(pesos_optimos, 4),
    })

    sharpe_opt = -resultado.fun
    evol_bayesiana = list(resultado.func_vals)

    return pesos_tickers, sharpe_opt, rendimiento_ex_ante, evol_bayesiana
