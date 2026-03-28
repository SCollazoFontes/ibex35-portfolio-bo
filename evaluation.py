"""
Evaluación ex post: rendimiento y ratio de Sharpe del año siguiente.
"""
import numpy as np
import pandas as pd

from .data.returns import filtrar_risk_free


def _rendimientos_y_pesos(
    año_inicio: int,
    pesos_optimos: pd.DataFrame,
    precios_componentes: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Devuelve (rendimientos_diarios, pesos_alineados) para el año siguiente."""
    if "Ticker" not in pesos_optimos.columns or "Pesos" not in pesos_optimos.columns:
        raise ValueError("El DataFrame debe contener las columnas 'Ticker' y 'Pesos'.")

    tickers = pesos_optimos["Ticker"].values
    pesos = pesos_optimos["Pesos"].values

    año_siguiente = año_inicio + 1
    precios = precios_componentes[precios_componentes.index.year == año_siguiente]
    if precios.empty:
        raise ValueError(f"No hay datos disponibles para el año {año_siguiente}.")

    tickers_presentes = [t for t in tickers if t in precios.columns]
    if not tickers_presentes:
        raise ValueError(f"Ningún ticker de la cartera está en precios para el año {año_siguiente}.")

    precios = precios[tickers_presentes]
    idx = [int(np.where(tickers == t)[0][0]) for t in tickers_presentes]
    pesos_alineados = np.asarray(pesos)[idx]
    pesos_alineados = pesos_alineados / pesos_alineados.sum()

    rendimientos = precios.pct_change(fill_method=None).dropna()
    if rendimientos.empty:
        raise ValueError(f"No se pudieron calcular rendimientos para el año {año_siguiente}.")

    return rendimientos, pesos_alineados


def rendimiento_año_siguiente(
    año_inicio: int,
    pesos_optimos: pd.DataFrame,
    precios_componentes: pd.DataFrame,
) -> float:
    """Calcula el rendimiento acumulado de la cartera en el año siguiente."""
    rendimientos, pesos = _rendimientos_y_pesos(año_inicio, pesos_optimos, precios_componentes)
    rendimiento_diario = rendimientos.dot(pesos)
    return float((1 + rendimiento_diario).prod() - 1)


def sharpe_año_siguiente(
    año_inicio: int,
    pesos_optimos: pd.DataFrame,
    precios_componentes: pd.DataFrame,
    risk_free_alineado: pd.DataFrame,
) -> float:
    """Calcula el ratio de Sharpe ex post de la cartera en el año siguiente."""
    rendimientos, pesos = _rendimientos_y_pesos(año_inicio, pesos_optimos, precios_componentes)
    rendimiento_diario = rendimientos.dot(pesos)

    rf_filtrado = filtrar_risk_free(risk_free_alineado, año_inicio + 1, 1)
    rf_diario = rf_filtrado.reindex(rendimiento_diario.index).ffill()["Yield"] / 252

    exceso = rendimiento_diario - rf_diario
    std_exceso = exceso.std(ddof=0)
    return float((exceso.mean() / std_exceso * np.sqrt(252)) if std_exceso > 1e-12 else 0.0)
