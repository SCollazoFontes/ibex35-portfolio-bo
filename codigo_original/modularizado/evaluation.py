import numpy as np
import pandas as pd


def rendimiento_año_siguiente(año_inicio, pesos_optimos, precios_componentes):
    """Calcula el rendimiento acumulado de la cartera en el año siguiente al de optimización."""
    try:
        if "Ticker" not in pesos_optimos.columns or "Pesos" not in pesos_optimos.columns:
            raise ValueError("El DataFrame de pesos debe contener las columnas 'Ticker' y 'Pesos'.")

        tickers = pesos_optimos["Ticker"].values
        pesos   = pesos_optimos["Pesos"].values

        año_siguiente = año_inicio + 1
        precios_año   = precios_componentes[precios_componentes.index.year == año_siguiente]
        if precios_año.empty:
            raise ValueError(f"No hay datos disponibles para el año {año_siguiente}.")
        precios_año = precios_año[tickers]

        rendimientos_diarios = precios_año.pct_change(fill_method=None).dropna()
        if rendimientos_diarios.empty:
            raise ValueError(f"No se pudieron calcular rendimientos diarios para {año_siguiente}.")

        rendimiento_diario_cartera = rendimientos_diarios.dot(pesos)
        return (1 + rendimiento_diario_cartera).prod() - 1

    except Exception as e:
        raise RuntimeError(f"Error al calcular el rendimiento de la cartera: {e}")


def sharpe_año_siguiente(año_inicio, pesos_optimos, precios_componentes, risk_free_alineado):
    """Calcula el Ratio de Sharpe ex post de la cartera en el año siguiente al de optimización."""
    try:
        if "Ticker" not in pesos_optimos.columns or "Pesos" not in pesos_optimos.columns:
            raise ValueError("El DataFrame de pesos debe contener las columnas 'Ticker' y 'Pesos'.")

        tickers = pesos_optimos["Ticker"].values
        pesos   = pesos_optimos["Pesos"].values

        año_siguiente = año_inicio + 1
        precios_año   = precios_componentes[precios_componentes.index.year == año_siguiente]
        if precios_año.empty:
            raise ValueError(f"No hay datos disponibles para el año {año_siguiente}.")
        precios_año = precios_año[tickers]

        rendimientos_diarios = precios_año.pct_change(fill_method=None).dropna()
        if rendimientos_diarios.empty:
            raise ValueError(f"No se pudieron calcular rendimientos diarios para {año_siguiente}.")

        rendimiento_diario_cartera = rendimientos_diarios.dot(pesos)

        rf_diario = risk_free_alineado.reindex(rendimiento_diario_cartera.index).ffill()["Yield"] / 252
        exceso    = rendimiento_diario_cartera - rf_diario

        sharpe = exceso.mean() / exceso.std(ddof=0) * np.sqrt(252)
        return sharpe

    except Exception as e:
        raise RuntimeError(f"Error al calcular el Sharpe ex post: {e}")
