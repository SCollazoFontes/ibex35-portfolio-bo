import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from .sharpe import sharpe_ratio
from config import (
    PESO_TOTAL, PESO_CONC_MAX, UMBRAL, MIN_OBS_RATIO,
    OB_N_CALLS, OB_N_INITIAL_POINTS, OB_RANDOM_STATE, OB_PACIENCIA, OB_TOLERANCIA,
)
from data.returns import rendimientos_diarios_filtrados, filtrar_risk_free, rendimientos_anuales_ibex
from evaluation import rendimiento_año_siguiente, sharpe_año_siguiente


class ParadaSinMejora:
    """Criterio de parada por convergencia: detiene la optimización si no hay mejora."""
    def __init__(self, paciencia=OB_PACIENCIA, tolerancia=OB_TOLERANCIA):
        self.paciencia  = paciencia
        self.tolerancia = tolerancia
        self.mejor      = np.inf
        self.espera     = 0

    def __call__(self, res):
        if res.fun < self.mejor - self.tolerancia:
            self.mejor  = res.fun
            self.espera = 0
        else:
            self.espera += 1
        return self.espera >= self.paciencia


def modelo_OB(rendimientos_diarios_periodo, risk_free_rate):
    """Obtiene los pesos óptimos de una cartera mediante Optimización Bayesiana."""
    if rendimientos_diarios_periodo.empty:
        raise ValueError("El DataFrame de rendimientos diarios está vacío.")

    promedio_rendimientos = rendimientos_diarios_periodo.mean() * 252
    matriz_covarianzas    = rendimientos_diarios_periodo.cov() * 252
    tickers               = rendimientos_diarios_periodo.columns.tolist()
    n_activos             = len(tickers)

    espacio_pesos = [Real(0.0, 1.0, name=f"w{i}") for i in range(n_activos)]

    restricciones = [
        {"type": "eq",   "fun": lambda pesos: np.sum(pesos) - PESO_TOTAL},
        {"type": "ineq", "fun": lambda pesos: PESO_CONC_MAX - np.sum(pesos[pesos > UMBRAL])},
    ]

    @use_named_args(espacio_pesos)
    def objetivo(**pesos_dict):
        try:
            pesos  = np.array(list(pesos_dict.values()))
            pesos /= np.sum(pesos)

            penalizacion = 0.0
            for r in restricciones:
                v = r["fun"](pesos)
                if r["type"] == "eq":
                    penalizacion += 1e3 * v ** 2
                elif v < 0:
                    penalizacion += 1e3 * v ** 2

            valor = sharpe_ratio(pesos, promedio_rendimientos, matriz_covarianzas, risk_free_rate) + penalizacion

            if np.isnan(valor) or np.isinf(valor):
                return 1e8
            return float(valor)

        except Exception as err:
            print("Error en objetivo:", err)
            return 1e8

    stopper   = ParadaSinMejora()
    resultado = gp_minimize(
        objetivo, espacio_pesos,
        n_calls=OB_N_CALLS, n_initial_points=OB_N_INITIAL_POINTS,
        random_state=OB_RANDOM_STATE, callback=[stopper], verbose=False,
    )

    pesos_optimos  = np.array(resultado.x)
    pesos_optimos /= np.sum(pesos_optimos)

    rendimiento_ex_ante = np.dot(pesos_optimos, promedio_rendimientos.values).item()

    pesos_tickers = pd.DataFrame({
        "Ticker": tickers,
        "Pesos":  np.round(pesos_optimos, 4),
    })

    sharpe_opt    = -resultado.fun
    evol_bayesiana = resultado.func_vals

    return pesos_tickers, sharpe_opt, rendimiento_ex_ante, evol_bayesiana


def OB(año_inicio, años_atras, año_final, data):
    """Backtest del modelo de Optimización Bayesiana año a año vs IBEX 35."""
    componentes_actualizados = data["componentes_actualizados"]
    precios_componentes      = data["precios_componentes"]
    rendimientos_componentes = data["rendimientos_componentes"]
    risk_free_alineado       = data["risk_free_alineado"]
    indice                   = data["indice"]

    ob_resultados           = {}
    resultados              = []
    rendimiento_cartera_ex_ante = []
    evolucion_optimizacion  = {}

    for año in range(año_inicio, año_final):
        try:
            rendimientos_diarios = rendimientos_diarios_filtrados(
                año, años_atras, componentes_actualizados, precios_componentes, rendimientos_componentes
            )
            min_obs = int(MIN_OBS_RATIO * len(rendimientos_diarios))
            rendimientos_diarios_relevantes = rendimientos_diarios.dropna(axis=1, thresh=min_obs)

            rf_filtrado    = filtrar_risk_free(año, años_atras, risk_free_alineado)
            risk_free_real = rf_filtrado.mean().iloc[0]

            pesos_optimos, sharpe_ratio_opt, rendimiento_ex_ante, evolucion_bayesiana = modelo_OB(
                rendimientos_diarios_relevantes, risk_free_real
            )

            rendimiento_cartera  = rendimiento_año_siguiente(año, pesos_optimos, precios_componentes)
            rendimiento_ibex_val = rendimientos_anuales_ibex(indice, [año + 1])
            sharpe_ex_post       = sharpe_año_siguiente(año, pesos_optimos, precios_componentes, risk_free_alineado)

            resultados.append({
                "Año":                      año + 1,
                "Rendimiento de la Cartera": rendimiento_cartera,
                "Rendimiento del IBEX 35":  rendimiento_ibex_val.get(año + 1, None),
            })

            ob_resultados[año] = {
                "Pesos":               pesos_optimos,
                "Sharpe Ratio Ex Ante": sharpe_ratio_opt,
                "Sharpe Ratio Ex Post": sharpe_ex_post,
            }

            rendimiento_cartera_ex_ante.append({"Año": año + 1, "Rendimiento Ex Ante": rendimiento_ex_ante})
            evolucion_optimizacion[año + 1] = evolucion_bayesiana

        except Exception as e:
            print(f"Error procesando el año {año}: {e}")

    resultados_df = pd.DataFrame(resultados)
    if not resultados_df.empty:
        resultados_df["Diferencia"] = (
            resultados_df["Rendimiento de la Cartera"] - resultados_df["Rendimiento del IBEX 35"]
        )
        for col in ["Rendimiento de la Cartera", "Rendimiento del IBEX 35", "Diferencia"]:
            resultados_df[col] = resultados_df[col].map(lambda x: f"{x * 100:.4f}%")
    else:
        print("No se generaron resultados válidos.")

    return resultados_df, ob_resultados, rendimiento_cartera_ex_ante, evolucion_optimizacion
