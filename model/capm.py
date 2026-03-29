import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .sharpe import sharpe_ratio
from config import PESO_MIN, PESO_MAX, PESO_TOTAL, PESO_CONC_MAX, UMBRAL, MIN_OBS_RATIO
from data.returns import (
    rendimientos_diarios_filtrados,
    filtrar_risk_free,
    filtrar_rendimientos_ibex,
    rendimientos_anuales_ibex,
)
from evaluation import rendimiento_año_siguiente, sharpe_año_siguiente


def betas(rendimientos_activos, rendimientos_mercado):
    """Calcula la beta de cada activo con respecto al mercado."""
    rendimientos_activos = rendimientos_activos.loc[rendimientos_mercado.index]
    rendimientos_mercado = rendimientos_mercado.loc[rendimientos_activos.index]

    betas_dict = {}
    for ticker in rendimientos_activos.columns:
        r_activo  = rendimientos_activos[ticker].values
        r_mercado = rendimientos_mercado.values.squeeze()
        mask = ~np.isnan(r_activo) & ~np.isnan(r_mercado)
        if mask.sum() == 0:
            betas_dict[ticker] = np.nan
            continue
        cov         = np.cov(r_activo[mask], r_mercado[mask])[0, 1]
        var_mercado = np.var(r_mercado[mask])
        betas_dict[ticker] = cov / var_mercado if var_mercado > 0 else np.nan

    return pd.Series(betas_dict)


def betas_periodo(año_inicio, años_atras, componentes_actualizados, precios_componentes,
                  rendimientos_componentes, rendimientos_ibex):
    """Calcula las betas de los activos del índice en el periodo dado."""
    rendimientos_activos = rendimientos_diarios_filtrados(
        año_inicio, años_atras, componentes_actualizados, precios_componentes, rendimientos_componentes
    )
    min_obs = int(MIN_OBS_RATIO * len(rendimientos_activos))
    rendimientos_activos_relevantes = rendimientos_activos.dropna(axis=1, thresh=min_obs)
    mask_ultima_fecha = rendimientos_activos_relevantes.iloc[-1].notna()
    rendimientos_activos_relevantes = rendimientos_activos_relevantes.loc[:, mask_ultima_fecha]
    rendimientos_mercado = filtrar_rendimientos_ibex(año_inicio, años_atras, rendimientos_ibex)
    return betas(rendimientos_activos_relevantes, rendimientos_mercado)


def rendimiento_esperado_capm(betas_filtradas, año_inicio, años_atras, rendimientos_ibex, risk_free_alineado):
    """Calcula el rendimiento esperado de cada activo usando el CAPM."""
    rendimientos_mercado         = filtrar_rendimientos_ibex(año_inicio, años_atras, rendimientos_ibex)
    rf_filtrado                  = filtrar_risk_free(año_inicio, años_atras, risk_free_alineado)
    risk_free_real               = rf_filtrado.mean().iloc[0]
    rendimiento_esperado_mercado = rendimientos_mercado.mean() * 252

    rendimientos_esperados = {
        ticker: risk_free_real + beta * (rendimiento_esperado_mercado - risk_free_real).iloc[0]
        for ticker, beta in betas_filtradas.items()
    }
    return pd.Series(rendimientos_esperados)


def modelo_capm(rendimientos_diarios, año_inicio, años_atras, risk_free_real,
                componentes_actualizados, precios_componentes, rendimientos_componentes,
                rendimientos_ibex, risk_free_alineado, params=None):
    """Obtiene los pesos óptimos de una cartera mediante el CAPM."""
    p = params or {}
    peso_min      = p.get("PESO_MIN",      PESO_MIN)
    peso_max      = p.get("PESO_MAX",      PESO_MAX)
    peso_total    = p.get("PESO_TOTAL",    PESO_TOTAL)
    peso_conc_max = p.get("PESO_CONC_MAX", PESO_CONC_MAX)
    umbral        = p.get("UMBRAL",        UMBRAL)

    try:
        if rendimientos_diarios.empty:
            raise ValueError("El DataFrame de rendimientos diarios está vacío.")

        betas_filtradas      = betas_periodo(
            año_inicio, años_atras, componentes_actualizados,
            precios_componentes, rendimientos_componentes, rendimientos_ibex
        )
        rendimiento_esperado = rendimiento_esperado_capm(
            betas_filtradas, año_inicio, años_atras, rendimientos_ibex, risk_free_alineado
        )
        # Alinear rendimientos_diarios con los tickers que tienen betas calculadas
        tickers_capm       = rendimiento_esperado.index.intersection(rendimientos_diarios.columns)
        rendimientos_capm  = rendimientos_diarios[tickers_capm]
        rend_esp_alineado  = rendimiento_esperado[tickers_capm]

        matriz_covarianzas = rendimientos_capm.cov() * 252
        num_activos        = len(tickers_capm)

        limites = tuple((peso_min, peso_max) for _ in range(num_activos))
        restricciones = [
            {"type": "eq",   "fun": lambda pesos: np.sum(pesos) - peso_total},
            {"type": "ineq", "fun": lambda pesos: peso_conc_max - np.sum(pesos[pesos > umbral])},
        ]
        pesos_iniciales = np.ones(num_activos) / num_activos

        resultado = minimize(
            sharpe_ratio, pesos_iniciales,
            args=(rend_esp_alineado, matriz_covarianzas, risk_free_real),
            method="SLSQP", bounds=limites, constraints=restricciones,
        )

        pesos_optimos       = resultado.x
        sharpe_ratio_opt    = -resultado.fun
        rendimiento_ex_ante = np.dot(pesos_optimos, rend_esp_alineado.values).item()

        pesos_tickers = pd.DataFrame({
            "Ticker": tickers_capm,
            "Pesos":  np.round(pesos_optimos, 4),
        })
        return pesos_tickers, sharpe_ratio_opt, rendimiento_ex_ante

    except Exception as e:
        raise RuntimeError(f"Error al optimizar la cartera (CAPM): {e}")


def CAPM(año_inicio, años_atras, año_final, data, params=None):
    """Backtest del modelo CAPM año a año vs IBEX 35."""
    componentes_actualizados = data["componentes_actualizados"]
    precios_componentes      = data["precios_componentes"]
    rendimientos_componentes = data["rendimientos_componentes"]
    risk_free_alineado       = data["risk_free_alineado"]
    rendimientos_ibex        = data["rendimientos_ibex"]
    indice                   = data["indice"]

    capm_resultados             = {}
    resultados                  = []
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

            pesos_optimos, sharpe_ratio_opt, rendimiento_ex_ante = modelo_capm(
                rendimientos_diarios_relevantes, año, años_atras, risk_free_real,
                componentes_actualizados, precios_componentes, rendimientos_componentes,
                rendimientos_ibex, risk_free_alineado, params
            )

            rendimiento_cartera  = rendimiento_año_siguiente(año, pesos_optimos, precios_componentes)
            rendimiento_ibex_val = rendimientos_anuales_ibex(indice, [año + 1])
            sharpe_ex_post       = sharpe_año_siguiente(año, pesos_optimos, precios_componentes, risk_free_alineado)

            resultados.append({
                "Año":                       año + 1,
                "Rendimiento de la Cartera": rendimiento_cartera,
                "Rendimiento del IBEX 35":   rendimiento_ibex_val.get(año + 1, None),
            })
            capm_resultados[año] = {
                "Pesos":                pesos_optimos,
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

    return resultados_df, capm_resultados, rendimiento_cartera_ex_ante
