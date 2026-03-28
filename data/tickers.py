"""
Gestión de tickers y composición histórica del IBEX 35.
"""
import pandas as pd
from datetime import datetime


def fecha_proxima(fecha_dada, fechas_disponibles):
    """Encuentra la fecha más cercana anterior o igual a una fecha dada."""
    try:
        fecha_dada = pd.to_datetime(fecha_dada)
        fechas_disponibles = pd.to_datetime(fechas_disponibles)
        fecha_cercana = fechas_disponibles[fechas_disponibles <= fecha_dada].max()
        return fecha_cercana
    except Exception as e:
        raise ValueError(f"Error al procesar las fechas: {e}") from e


def composicion_ibex(fecha, datos_componentes: pd.DataFrame):
    """Obtiene la composición del IBEX para una fecha específica."""
    try:
        fecha_cercana = fecha_proxima(fecha, datos_componentes.columns[1:])
        if pd.isna(fecha_cercana):
            return f"No hay datos disponibles para la fecha {fecha} o anteriores."

        composicion = datos_componentes[fecha_cercana.strftime("%Y-%m-%d")].dropna().tolist()
        return composicion
    except Exception as e:
        return f"Error al obtener la composición: {e}"


def fechas_relevantes(componentes: pd.DataFrame, años_atras: int, año_actual=None):
    """Selecciona las columnas (fechas) de composición en los últimos X años."""
    if año_actual is None:
        año_actual = datetime.now().year
    años_recientes = [str(a) for a in range(año_actual - años_atras + 1, año_actual + 1)]
    return [col for col in componentes.columns if any(a in col for a in años_recientes)]


def tickers_historicos(componentes: pd.DataFrame, columnas: list) -> set:
    """Obtiene todos los tickers presentes en las columnas dadas."""
    return set(componentes[columnas].stack().dropna())


def tickers_comunes(componentes: pd.DataFrame, columnas: list) -> set:
    """Obtiene los tickers comunes en todas las fechas seleccionadas."""
    return set.intersection(*[set(componentes[col].dropna()) for col in columnas])


def comparar_tickers(componentes: pd.DataFrame, precios_componentes: pd.DataFrame, fechas_recientes: list) -> set:
    """Diferencia simétrica entre tickers en componentes y en precios."""
    tickers_comp = set(componentes[fechas_recientes].stack().dropna())
    return tickers_comp.symmetric_difference(precios_componentes.columns)


def actualizar_tickers(componentes: pd.DataFrame, cambios_tickers: dict) -> pd.DataFrame:
    """Unifica nombres de tickers al nombre más reciente."""
    return componentes.replace(cambios_tickers)
