"""
Carga del archivo Excel con los datos del IBEX 35.
"""
import pandas as pd
from pathlib import Path


def load_excel(path: Path):
    """
    Carga las hojas del archivo Excel del IBEX.

    - Constituents: composición histórica del índice
    - Index: valores históricos del índice
    - Prices: precios históricos de los componentes
    - RiskFree: bono español 10 años

    Returns
    -------
    dict con keys: componentes, indice, precios_componentes, datos_risk_free
    """
    componentes = pd.read_excel(path, sheet_name="Constituents", header=None)
    indice = pd.read_excel(path, sheet_name="Index")
    precios_componentes = pd.read_excel(path, sheet_name="Prices")
    datos_risk_free = pd.read_excel(path, sheet_name="RiskFree")

    return {
        "componentes": componentes,
        "indice": indice,
        "precios_componentes": precios_componentes,
        "datos_risk_free": datos_risk_free,
    }
