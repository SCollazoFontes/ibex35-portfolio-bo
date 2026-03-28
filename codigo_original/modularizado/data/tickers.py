import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# Composición del índice
# ---------------------------------------------------------------------------

def fecha_proxima(fecha_dada, fechas_disponibles):
    """Encuentra la fecha más cercana anterior o igual a una fecha dada."""
    try:
        fecha_dada = pd.to_datetime(fecha_dada)
        fechas_disponibles = pd.to_datetime(fechas_disponibles)
        fecha_cercana = fechas_disponibles[fechas_disponibles <= fecha_dada].max()
        return fecha_cercana
    except Exception as e:
        raise ValueError(f"Error al procesar las fechas: {e}")


def composicion_ibex(fecha, componentes_actualizados):
    """Obtiene la composición del IBEX para una fecha específica."""
    try:
        fecha_cercana = fecha_proxima(fecha, componentes_actualizados.columns[1:])
        if pd.isna(fecha_cercana):
            return f"No hay datos disponibles para la fecha {fecha} o anteriores."
        composicion = componentes_actualizados[fecha_cercana.strftime("%Y-%m-%d")].dropna().tolist()
        return composicion
    except Exception as e:
        return f"Error al obtener la composición: {e}"


# ---------------------------------------------------------------------------
# Utilidades de tickers
# ---------------------------------------------------------------------------

def tickers_historicos(componentes, columnas):
    """Obtiene todos los tickers presentes en el dataset seleccionado."""
    return set(componentes[columnas].stack().dropna())


def fechas_relevantes(componentes, años_atras, año_actual=None):
    """Selecciona las fechas con cambios en la composición del índice en los últimos X años."""
    if año_actual is None:
        año_actual = datetime.now().year
    años_recientes = [str(año) for año in range(año_actual - años_atras + 1, año_actual + 1)]
    return [col for col in componentes.columns if any(año in col for año in años_recientes)]


def tickers_comunes(componentes, columnas):
    """Obtiene los tickers presentes en todos los años seleccionados."""
    return set.intersection(*[set(componentes[col].dropna()) for col in columnas])


def comparar_tickers(componentes, precios_componentes, fechas_recientes):
    """Compara los tickers entre los componentes del índice y los datos de precios."""
    tickers_componentes = set(componentes[fechas_recientes].stack().dropna())
    return tickers_componentes.symmetric_difference(precios_componentes.columns)


def actualizar_tickers(componentes, cambios_tickers):
    """Unifica los nombres de tickers que han cambiado a lo largo del tiempo."""
    return componentes.replace(cambios_tickers)
