import pandas as pd
from pathlib import Path


def load_excel(path: Path) -> dict:
    """
    Carga las 4 hojas del Excel de datos del IBEX 35.

    Devuelve un dict con las hojas en bruto (sin procesar):
        - 'componentes'       : composición histórica del índice
        - 'indice'            : valores históricos del índice
        - 'precios'           : precios históricos de los componentes
        - 'risk_free'         : datos históricos del bono español a 10 años
    """
    componentes  = pd.read_excel(path, sheet_name="Constituents", header=None)
    indice       = pd.read_excel(path, sheet_name="Index")
    precios      = pd.read_excel(path, sheet_name="Prices")
    risk_free    = pd.read_excel(path, sheet_name="RiskFree")

    return {
        "componentes": componentes,
        "indice":      indice,
        "precios":     precios,
        "risk_free":   risk_free,
    }
