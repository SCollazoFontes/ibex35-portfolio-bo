"""
Preparación de los DataFrames cargados: índices, fechas y risk-free alineado.
"""
import pandas as pd
from .tickers import actualizar_tickers


def prepare_data(raw: dict, cambios_tickers: dict) -> dict:
    """
    Prepara todos los datos para el pipeline: índices datetime,
    risk_free alineado con el índice, componentes con tickers unificados,
    rendimientos del índice y de componentes.

    Parameters
    ----------
    raw : dict
        Salida de load_excel (componentes, indice, precios_componentes, datos_risk_free)
    cambios_tickers : dict
        Mapeo de tickers antiguos a actuales

    Returns
    -------
    dict con: componentes_actualizados, indice, precios_componentes,
              risk_free_alineado, rendimientos_componentes, rendimientos_ibex
    """
    componentes = raw["componentes"]
    indice = raw["indice"]
    precios_componentes = raw["precios_componentes"]
    datos_risk_free = raw["datos_risk_free"]

    # Constituents: encabezados con fechas
    componentes.columns = ["Ticker"] + pd.to_datetime(
        componentes.iloc[0, 1:], dayfirst=True
    ).dt.strftime("%Y-%m-%d").tolist()
    componentes = componentes.iloc[1:]

    # Componentes con tickers unificados
    componentes_actualizados = actualizar_tickers(componentes, cambios_tickers)

    # Index: índice por fecha
    indice.set_index("Date", inplace=True)
    indice.index = pd.to_datetime(indice.index)

    # Precios: índice por fecha
    precios_componentes.set_index(precios_componentes.columns[0], inplace=True)
    precios_componentes.index = pd.to_datetime(precios_componentes.index)

    # Risk-free: columna Yield, alineado con índice del IBEX
    datos_risk_free.set_index("Date", inplace=True)
    datos_risk_free.index = pd.to_datetime(datos_risk_free.index)
    risk_free = datos_risk_free[["Yield"]].dropna() / 100
    risk_free.index = pd.to_datetime(risk_free.index)
    risk_free_alineado = risk_free.reindex(indice.index).ffill()

    # Rendimientos (fill_method=None evita FutureWarning en pandas 2.x)
    rendimientos_ibex = pd.DataFrame(indice["Price"].pct_change(fill_method=None))
    rendimientos_ibex.fillna(0, inplace=True)

    rendimientos_componentes = precios_componentes.pct_change(fill_method=None)

    return {
        "componentes_actualizados": componentes_actualizados,
        "indice": indice,
        "precios_componentes": precios_componentes,
        "risk_free_alineado": risk_free_alineado,
        "rendimientos_componentes": rendimientos_componentes,
        "rendimientos_ibex": rendimientos_ibex,
    }
