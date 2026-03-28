import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .sharpe import sharpe_ratio
from config import PESO_MIN, PESO_MAX, PESO_TOTAL, PESO_CONC_MAX, UMBRAL, MIN_OBS_RATIO
from data.returns import rendimientos_diarios_filtrados, filtrar_risk_free, rendimientos_anuales_ibex
from evaluation import rendimiento_año_siguiente, sharpe_año_siguiente


def modelo_media_varianza(rendimientos_diarios, risk_free_real):
    """Obtiene los pesos óptimos de una cartera mediante el modelo de Media-Varianza de Markowitz."""
    try:
        if rendimientos_diarios.empty:
            raise ValueError("El DataFrame de rendimientos diarios está vacío.")

        promedio_rendimientos = rendimientos_diarios.mean() * 252
        matriz_covarianzas    = rendimientos_diarios.cov() * 252
        num_activos           = len(promedio_rendimientos)

        limites = tuple((PESO_MIN, PESO_MAX) for _ in range(num_activos))

        restricciones = [
            {"type": "eq",   "fun": lambda pesos: np.sum(pesos) - PESO_TOTAL},
            {"type": "ineq", "fun": lambda pesos: PESO_CONC_MAX - np.sum(pesos[pesos > UMBRAL])},
        ]

        pesos_iniciales = np.ones(num_activos) / num_activos

        resultado = minimize(
            sharpe_ratio, pesos_iniciales,
            args=(promedio_rendimientos, matriz_covarianzas, risk_free_real),
            method="SLSQP", bounds=limites, constraints=restricciones,
        )

        pesos_optimos    = resultado.x
        sharpe_ratio_opt = -resultado.fun
        rendimiento_ex_ante = np.dot(pesos_optimos, promedio_rendimientos.values).item()

        pesos_tickers = pd.DataFrame({
            "Ticker": rendimientos_diarios.columns,
            "Pesos":  np.round(pesos_optimos, 4),
        })

        return pesos_tickers, sharpe_ratio_opt, rendimiento_ex_ante

    except Exception as e:
        raise RuntimeError(f"Error al optimizar la cartera (MMV): {e}")


def MMV(año_inicio, años_atras, año_final, data):
    """Backtest del modelo Media-Varianza año a año vs IBEX 35."""
    componentes_actualizados  = data["componentes_actualizados"]
    precios_componentes       = data["precios_componentes"]
    rendimientos_componentes  = data["rendimientos_componentes"]
    risk_free_alineado        = data["risk_free_alineado"]
    indice                    = data["indice"]

    resultados             = []
    markowitz              = {}
    rendimiento_cartera_ex_ante = []

    for año in range(año_inicio, año_final):
        try:
            rendimientos_diarios = rendimientos_diarios_filtrados(
                año, años_atras, componentes_actualizados, precios_componentes, rendimientos_componentes
            )
            min_obs = int(MIN_OBS_RATIO * len(rendimientos_diarios))
            rendimientos_diarios_relevantes = rendimientos_diarios.dropna(axis=1, thresh=min_obs)

            rf_filtrado    = filtrar_risk_free(año, años_atras, risk_free_alineado)
            risk_free_real = rf_filtrado.mean().iloc[0]

            pesos_optimos, sharpe_ratio_opt, rendimiento_ex_ante = modelo_media_varianza(
                rendimientos_diarios_relevantes, risk_free_real
            )

            rendimiento_cartera = rendimiento_año_siguiente(año, pesos_optimos, precios_componentes)
            rendimiento_ibex    = rendimientos_anuales_ibex(indice, [año + 1])
            sharpe_ex_post      = sharpe_año_siguiente(año, pesos_optimos, precios_componentes, risk_free_alineado)

            resultados.append({
                "Año":                      año + 1,
                "Rendimiento de la Cartera": rendimiento_cartera,
                "Rendimiento del IBEX 35":  rendimiento_ibex.get(año + 1, None),
            })

            markowitz[año] = {
                "Pesos":               pesos_optimos,
                "Sharpe Ratio Ex Ante": sharpe_ratio_opt,
                "Sharpe Ratio Ex Post": sharpe_ex_post,
            }

            rendimiento_cartera_ex_ante.append({"Año": año + 1, "Rendimiento Ex Ante": rendimiento_ex_ante})

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

    return resultados_df, markowitz, rendimiento_cartera_ex_ante
