import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import linregress

from .sharpe import sharpe_ratio
from config import PESO_MIN, PESO_MAX, PESO_TOTAL, PESO_CONC_MAX, UMBRAL, MIN_OBS_RATIO, VE_N_SIM
from data.returns import rendimientos_diarios_filtrados, filtrar_risk_free, rendimientos_anuales_ibex
from evaluation import rendimiento_año_siguiente, sharpe_año_siguiente


def calcular_metricas_historicas_VE(rendimiento_ibex):
    """Calcula los parámetros del modelo de Heston a partir de datos históricos."""
    if isinstance(rendimiento_ibex, pd.DataFrame):
        rendimiento_ibex = rendimiento_ibex.iloc[:, 0]

    varianza_diaria = rendimiento_ibex ** 2
    theta           = varianza_diaria.mean()
    varianza_lag    = varianza_diaria.shift(1).dropna()
    delta_varianza  = varianza_diaria.diff().dropna()

    try:
        pendiente, _, _, _, _ = linregress(varianza_lag.loc[delta_varianza.index], delta_varianza)
        kappa = -pendiente
    except Exception as e:
        print(f"Error al calcular kappa: {e}")
        kappa = np.nan

    try:
        sigma = delta_varianza.std()
    except Exception as e:
        print(f"Error al calcular sigma: {e}")
        sigma = np.nan

    try:
        rho = np.corrcoef(rendimiento_ibex.loc[delta_varianza.index], delta_varianza)[0, 1]
    except Exception as e:
        print(f"Error al calcular rho: {e}")
        rho = np.nan

    return {"kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho}


def volatilidades_heston(parametros, rendimientos_componentes):
    """Simula volatilidades dinámicas usando el modelo de Heston (serie completa)."""
    kappa, theta, sigma = parametros["kappa"], parametros["theta"], parametros["sigma"]
    tickers = rendimientos_componentes.columns
    fechas  = rendimientos_componentes.index
    N, dt   = len(fechas), 1 / 252

    v0 = rendimientos_componentes.var().values
    volatilidades_simuladas = pd.DataFrame(index=fechas, columns=tickers)

    for ticker in tickers:
        v             = v0[tickers.get_loc(ticker)]
        volatilidades = np.zeros(N)
        volatilidades[0] = v
        for t in range(1, N):
            dW = np.random.normal()
            dv = kappa * (theta - volatilidades[t-1]) * dt + sigma * np.sqrt(max(volatilidades[t-1], 0)) * dW * np.sqrt(dt)
            volatilidades[t] = max(volatilidades[t-1] + dv, 0)
        volatilidades_simuladas[ticker] = np.sqrt(volatilidades)

    return volatilidades_simuladas


def covarianza_dinamica(volatilidades_simuladas, correlaciones_historicas):
    """Calcula matrices de covarianza dinámicas basadas en las volatilidades simuladas."""
    tickers               = volatilidades_simuladas.columns
    covarianzas_dinamicas = {}

    for fecha in volatilidades_simuladas.index:
        volat_actual    = volatilidades_simuladas.loc[fecha]
        activos_validos = volat_actual > 0

        if activos_validos.any():
            volat_actual          = volat_actual[activos_validos].values
            correlaciones_validas = correlaciones_historicas.values[np.ix_(activos_validos, activos_validos)]
            diag_volat            = np.diag(np.sqrt(volat_actual))
            covarianza            = diag_volat @ correlaciones_validas @ diag_volat
            covarianza           *= 252
        else:
            covarianza = np.zeros((len(tickers), len(tickers)))

        covarianzas_dinamicas[fecha] = covarianza

    return covarianzas_dinamicas


def volatilidades_dinamicas(parametros, rendimientos):
    """Simula volatilidades dinámicas usando el modelo de Heston para un periodo específico."""
    kappa, theta, sigma = parametros["kappa"], parametros["theta"], parametros["sigma"]
    tickers = rendimientos.columns
    fechas  = rendimientos.index
    N, dt   = len(fechas), 1 / 252

    v0 = rendimientos.var().values
    volatilidades_simuladas = pd.DataFrame(index=fechas, columns=tickers)

    for ticker in tickers:
        v             = v0[tickers.get_loc(ticker)]
        volatilidades = np.zeros(N)
        volatilidades[0] = v
        for t in range(1, N):
            dW = np.random.normal(0, np.sqrt(dt))
            dv = kappa * (theta - volatilidades[t-1]) * dt + sigma * np.sqrt(max(volatilidades[t-1], 0)) * dW
            volatilidades[t] = max(volatilidades[t-1] + dv, 0)
        volatilidades_simuladas[ticker] = volatilidades

    return volatilidades_simuladas


def modelo_VE(rendimientos_diarios_periodo, risk_free_real, covarianza, params=None):
    """Obtiene los pesos óptimos de una cartera mediante el modelo de Volatilidad Estocástica de Heston."""
    p = params or {}
    peso_min      = p.get("PESO_MIN",      PESO_MIN)
    peso_max      = p.get("PESO_MAX",      PESO_MAX)
    peso_total    = p.get("PESO_TOTAL",    PESO_TOTAL)
    peso_conc_max = p.get("PESO_CONC_MAX", PESO_CONC_MAX)
    umbral        = p.get("UMBRAL",        UMBRAL)

    try:
        if rendimientos_diarios_periodo.empty:
            raise ValueError("El DataFrame de rendimientos diarios está vacío.")

        promedio_rendimientos = rendimientos_diarios_periodo.mean() * 252
        num_activos           = len(promedio_rendimientos)
        pesos_iniciales       = np.ones(num_activos) / num_activos
        limites               = tuple((peso_min, peso_max) for _ in range(num_activos))

        restricciones = [
            {"type": "eq",   "fun": lambda pesos: np.sum(pesos) - peso_total},
            {"type": "ineq", "fun": lambda pesos: peso_conc_max - np.sum(pesos[pesos > umbral])},
        ]

        resultado = minimize(
            sharpe_ratio, pesos_iniciales,
            args=(promedio_rendimientos, covarianza, risk_free_real),
            method="SLSQP", bounds=limites, constraints=restricciones,
        )

        pesos_optimos       = resultado.x
        sharpe_ratio_opt    = -resultado.fun
        rendimiento_ex_ante = np.dot(pesos_optimos, promedio_rendimientos.values).item()

        pesos_tickers = pd.DataFrame({
            "Ticker": rendimientos_diarios_periodo.columns,
            "Pesos":  np.round(pesos_optimos, 4),
        })
        return pesos_tickers, sharpe_ratio_opt, rendimiento_ex_ante

    except Exception as e:
        raise RuntimeError(f"Error al optimizar la cartera (VE): {e}")


def VE(año_inicio, años_atras, año_final, data, params=None):
    """Backtest del modelo de Volatilidad Estocástica (Heston) año a año vs IBEX 35."""
    componentes_actualizados = data["componentes_actualizados"]
    precios_componentes      = data["precios_componentes"]
    rendimientos_componentes = data["rendimientos_componentes"]
    risk_free_alineado       = data["risk_free_alineado"]
    indice                   = data["indice"]
    correlaciones_historicas = data["correlaciones_historicas"]

    ve_resultados               = {}
    resultados                  = []
    rendimiento_cartera_ex_ante = []

    for año in range(año_inicio, año_final):
        try:
            np.random.seed(1000 + año)

            rendimientos_diarios = rendimientos_diarios_filtrados(
                año, años_atras, componentes_actualizados, precios_componentes, rendimientos_componentes
            )
            min_obs = int(MIN_OBS_RATIO * len(rendimientos_diarios))
            rendimientos_diarios_relevantes = rendimientos_diarios.dropna(axis=1, thresh=min_obs)

            rendimientos_ibex_periodo = indice.loc[:f"{año}-12-31", "Price"].pct_change().dropna()
            parametros_historicos     = calcular_metricas_historicas_VE(rendimientos_ibex_periodo)

            suma_cov = None
            for _ in range(VE_N_SIM):
                vols_i   = volatilidades_dinamicas(parametros_historicos, rendimientos_diarios_relevantes)
                cov_i    = covarianza_dinamica(vols_i, correlaciones_historicas)
                cov_prom = np.mean(list(cov_i.values()), axis=0)
                suma_cov = cov_prom if suma_cov is None else suma_cov + cov_prom

            covarianza_promedio = suma_cov / VE_N_SIM

            rf_filtrado    = filtrar_risk_free(año, años_atras, risk_free_alineado)
            risk_free_real = rf_filtrado.mean().iloc[0]

            pesos_optimos, sharpe_ratio_opt, rendimiento_ex_ante = modelo_VE(
                rendimientos_diarios_relevantes, risk_free_real, covarianza_promedio, params
            )

            rendimiento_cartera  = rendimiento_año_siguiente(año, pesos_optimos, precios_componentes)
            rendimiento_ibex_val = rendimientos_anuales_ibex(indice, [año + 1])
            sharpe_ex_post       = sharpe_año_siguiente(año, pesos_optimos, precios_componentes, risk_free_alineado)

            resultados.append({
                "Año":                       año + 1,
                "Rendimiento de la Cartera": rendimiento_cartera,
                "Rendimiento del IBEX 35":   rendimiento_ibex_val.get(año + 1, None),
            })
            ve_resultados[año] = {
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

    return resultados_df, ve_resultados, rendimiento_cartera_ex_ante
