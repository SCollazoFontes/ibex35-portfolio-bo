import pandas as pd
from .tickers import composicion_ibex


# ---------------------------------------------------------------------------
# Risk free
# ---------------------------------------------------------------------------

def filtrar_risk_free(año_inicio, años_atras, risk_free_alineado):
    """Filtra los tipos libres de riesgo para el rango de fechas dado."""
    fecha_inicio = f"{año_inicio - años_atras + 1}-01-01"
    fecha_fin    = f"{año_inicio}-12-31"
    return risk_free_alineado.loc[fecha_inicio:fecha_fin]


# ---------------------------------------------------------------------------
# Rendimientos del índice
# ---------------------------------------------------------------------------

def rendimientos_anuales_ibex(indice, años):
    """Obtiene el rendimiento anual del IBEX 35 para los años dados."""
    rendimientos_anuales = {}
    for año in años:
        try:
            precio_inicio = indice[indice.index.year == (año - 1)]["Price"].iloc[-1]
            precio_final  = indice[indice.index.year == año]["Price"].iloc[-1]
            rendimientos_anuales[año] = (precio_final - precio_inicio) / precio_inicio
        except IndexError:
            print(f"Datos insuficientes para calcular el rendimiento del año {año}.")
    return rendimientos_anuales


def filtrar_rendimientos_ibex(año_inicio, años_atras, rendimientos_ibex):
    """Filtra los rendimientos diarios del IBEX 35 para el rango de fechas dado."""
    fecha_inicio = f"{año_inicio - años_atras + 1}-01-01"
    fecha_fin    = f"{año_inicio}-12-31"
    return rendimientos_ibex.loc[fecha_inicio:fecha_fin]


# ---------------------------------------------------------------------------
# Rendimientos de los componentes
# ---------------------------------------------------------------------------

def rendimientos_diarios_filtrados(año_inicio, años_atras, componentes_actualizados,
                                   precios_componentes, rendimientos_componentes):
    """
    Obtiene el rendimiento diario de las acciones que componían el índice
    al final del año dado, dentro del periodo de lookback especificado.
    """
    try:
        fin_de_año = f"{año_inicio}-12-31"
        tickers_fin_de_año = composicion_ibex(fin_de_año, componentes_actualizados)

        tickers_disponibles = [t for t in tickers_fin_de_año if t in precios_componentes.columns]
        if not tickers_disponibles:
            raise ValueError(f"Ningún ticker del IBEX disponible en precios para {fin_de_año}.")

        fecha_inicio = f"{año_inicio - años_atras + 1}-01-01"
        fecha_final  = f"{año_inicio}-12-31"

        return rendimientos_componentes.loc[fecha_inicio:fecha_final, tickers_disponibles]

    except Exception as e:
        raise RuntimeError(f"Error al filtrar rendimientos diarios: {e}")


def rendimientos_anuales_tickers(año_inicio, años_atras, componentes_actualizados,
                                 precios_componentes):
    """
    Obtiene el rendimiento anual de las acciones que componían el índice
    al final del año dado, dentro del periodo de lookback especificado.
    """
    try:
        fin_de_año = f"{año_inicio}-12-31"
        tickers_fin_de_año = composicion_ibex(fin_de_año, componentes_actualizados)

        tickers_disponibles = [t for t in tickers_fin_de_año if t in precios_componentes.columns]
        if not tickers_disponibles:
            raise ValueError(f"Ningún ticker del IBEX disponible en precios para {fin_de_año}.")

        rendimientos_anuales = {}
        for año in range(año_inicio - años_atras + 1, año_inicio + 1):
            try:
                fin_año_anterior = precios_componentes[precios_componentes.index.year == año - 1].index.max()
                fin_año_actual   = precios_componentes[precios_componentes.index.year == año].index.max()
                precio_inicio    = precios_componentes.loc[fin_año_anterior, tickers_disponibles]
                precio_final     = precios_componentes.loc[fin_año_actual,   tickers_disponibles]
                rendimientos_anuales[año] = (precio_final - precio_inicio) / precio_inicio
            except Exception as e:
                print(f"No se pudo calcular el rendimiento para el año {año}: {e}")

        return pd.DataFrame(rendimientos_anuales)

    except Exception as e:
        raise RuntimeError(f"Error al calcular rendimientos anuales por componente: {e}")
