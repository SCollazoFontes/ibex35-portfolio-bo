"""
Módulo de datos: carga, preparación, tickers y rendimientos.
"""
from .load import load_excel
from .prepare import prepare_data
from .tickers import composicion_ibex, fechas_relevantes, actualizar_tickers
from .returns import (
    filtrar_risk_free,
    rendimientos_diarios_filtrados,
    rendimientos_anuales_ibex,
)

__all__ = [
    "load_excel",
    "prepare_data",
    "composicion_ibex",
    "fechas_relevantes",
    "actualizar_tickers",
    "filtrar_risk_free",
    "rendimientos_diarios_filtrados",
    "rendimientos_anuales_ibex",
]
