"""
Cálculo y filtrado de rendimientos (índice, componentes, risk-free).
"""
import pandas as pd
from .tickers import composicion_ibex


def filtrar_risk_free(risk_free_alineado: pd.DataFrame, año_inicio: int, años_atras: int) -> pd.DataFrame:
    """Filtra el risk-free para el rango [año_inicio - años_atras + 1, año_inicio]."""
    fecha_inicio = f"{año_inicio - años_atras + 1}-01-01"
    fecha_fin = f"{año_inicio}-12-31"
    return risk_free_alineado.loc[fecha_inicio:fecha_fin]


def rendimientos_diarios_filtrados(
    año_inicio: int,
    años_atras: int,
    componentes_actualizados: pd.DataFrame,
    precios_componentes: pd.DataFrame,
    rendimientos_componentes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rendimiento diario de las acciones que componían el IBEX al cierre de año_inicio,
    en el periodo [año_inicio - años_atras + 1, año_inicio].
    """
    fin_de_año = f"{año_inicio}-12-31"
    tickers_fin_de_año = composicion_ibex(fin_de_año, componentes_actualizados)

    if isinstance(tickers_fin_de_año, str):
        raise ValueError(tickers_fin_de_año)

    tickers_disponibles = [t for t in tickers_fin_de_año if t in precios_componentes.columns]
    if not tickers_disponibles:
        raise ValueError(f"Ninguno de los tickers del IBEX está en precios para {fin_de_año}.")

    fecha_inicio = f"{año_inicio - años_atras + 1}-01-01"
    fecha_final = f"{año_inicio}-12-31"
    return rendimientos_componentes.loc[fecha_inicio:fecha_final, tickers_disponibles]


def rendimientos_anuales_ibex(indice: pd.DataFrame, años: list) -> dict:
    """
    Rendimiento anual del IBEX 35 para cada año en la lista.
    Vectorizado: último precio por año y pct_change en una sola pasada.
    """
    if indice.empty or "Price" not in indice.columns:
        return {}
    # Último precio por año (una fila por año)
    ultimo_por_año = indice.groupby(indice.index.year)["Price"].last()
    # Rendimiento anual = (precio_fin_año - precio_fin_año_anterior) / precio_fin_año_anterior
    ret_anual = ultimo_por_año.pct_change(fill_method=None)
    # Mapear años solicitados a valores (sin bucle)
    return {int(año): float(ret_anual.loc[año]) for año in años if año in ret_anual.index and pd.notna(ret_anual.loc[año])}
