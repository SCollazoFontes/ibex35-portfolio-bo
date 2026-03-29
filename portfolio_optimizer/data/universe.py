"""
universe.py — Composición histórica de índices bursátiles.

Por qué este módulo existe:
    El backtest debe saber qué acciones formaban el índice en cada fecha
    concreta. Sin esto, podríamos optimizar con acciones que aún no existían
    o que ya habían sido expulsadas (survivorship bias).

    Ejemplo: si optimizamos para el año 2012, solo debemos incluir los
    tickers que estaban en el IBEX 35 a 1 de enero de 2012, no los de hoy.

Fuente de datos:
    CSVs en data/universe/{index}.csv con columnas:
        ticker   : formato yfinance (ej: SAN.MC)
        date_in  : fecha de entrada al índice
        date_out : fecha de salida (vacío = sigue en el índice hoy)
        notes    : anotaciones opcionales

Carga:
    El CSV se lee UNA SOLA VEZ al importar el módulo y queda en memoria.
    Por qué: get_components() se llama en cada iteración del backtest
    (una por año). Releer el CSV 13 veces no tiene sentido.
"""

from __future__ import annotations

import logging
from pathlib import Path
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

# Directorio donde viven los CSVs de universo
_UNIVERSE_DIR = Path(__file__).resolve().parent / "universe"

# Mapeo de nombre de índice a fichero CSV
_INDEX_FILES = {
    "IBEX35": _UNIVERSE_DIR / "ibex35.csv",
    "SP500":  _UNIVERSE_DIR / "sp500.csv",
}


# ---------------------------------------------------------------------------
# Carga interna (una sola vez por índice)
# ---------------------------------------------------------------------------

def _load_universe(index: str) -> pd.DataFrame:
    """
    Carga el CSV de composición histórica del índice y lo prepara para consultas.

    Returns:
        DataFrame con columnas: ticker, date_in (Timestamp), date_out (Timestamp).
        date_out de tickers aún activos = pd.Timestamp.max (centinela de "sin fecha fin").

    Por qué pd.Timestamp.max como centinela:
        Permite comparar fecha <= date_out sin tratar None como caso especial.
        "¿Estaba SAN.MC el 2015-01-01?" → date_in <= 2015-01-01 <= date_out.
        Si date_out fuera None, necesitaríamos un if adicional en cada comparación.
    """
    path = _INDEX_FILES.get(index.upper())
    if path is None:
        raise ValueError(
            f"Índice '{index}' no reconocido. Disponibles: {list(_INDEX_FILES.keys())}"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el CSV de composición en {path}. "
            f"Asegúrate de que el archivo existe."
        )

    # Leemos el CSV ignorando líneas que empiecen por '#' (comentarios).
    # Por qué comment='#': el CSV puede tener cabeceras de documentación.
    df = pd.read_csv(path, comment="#", dtype=str)

    # Convertimos fechas a Timestamp para comparaciones eficientes.
    df["date_in"] = pd.to_datetime(df["date_in"])

    # date_out vacío → el ticker sigue activo → usamos Timestamp.max como centinela.
    df["date_out"] = df["date_out"].apply(
        lambda x: pd.Timestamp.max if pd.isna(x) or str(x).strip() == ""
        else pd.Timestamp(x)
    )

    logger.debug("Universo '%s' cargado: %d períodos, %d tickers únicos",
                 index, len(df), df["ticker"].nunique())
    return df


# lru_cache guarda el resultado de _load_universe por nombre de índice.
# La primera llamada con "IBEX35" lee el CSV; las siguientes devuelven
# el DataFrame ya en memoria sin tocar disco.
# Por qué lru_cache y no una variable global: es más limpio (no contamina
# el espacio de nombres del módulo) y se puede limpiar en tests con
# _load_universe_cached.cache_clear().
@lru_cache(maxsize=None)
def _load_universe_cached(index: str) -> pd.DataFrame:
    return _load_universe(index)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def get_components(index: str, date: str) -> list[str]:
    """
    Devuelve los tickers activos en el índice en la fecha dada.

    Un ticker está activo si: date_in <= date <= date_out.

    Args:
        index : "IBEX35" o "SP500"
        date  : fecha en formato "YYYY-MM-DD"

    Returns:
        Lista de tickers en formato yfinance, ordenada alfabéticamente.
        Ejemplo: ["ACS.MC", "BBVA.MC", "IBE.MC", ...]

    Comportamiento en fechas anteriores al primer registro:
        Si date < min(date_in) del CSV, devuelve la composición más antigua
        disponible y loggea un warning. Evita errores silenciosos cuando se
        pide una fecha fuera del rango cubierto.
    """
    ts = pd.Timestamp(date)
    universe = _load_universe_cached(index)

    # Filtramos los tickers cuyo período de pertenencia incluye 'date'
    mask = (universe["date_in"] <= ts) & (universe["date_out"] >= ts)
    result = sorted(universe.loc[mask, "ticker"].tolist())

    if result:
        return result

    # Si no hay ningún resultado, la fecha es anterior al primer registro.
    # Devolvemos la composición más antigua y avisamos.
    min_date = universe["date_in"].min()
    logger.warning(
        "get_components('%s', '%s'): fecha anterior al primer registro (%s). "
        "Devolviendo composición más antigua disponible.",
        index, date, min_date.strftime("%Y-%m-%d")
    )
    oldest_mask = universe["date_in"] <= min_date + pd.Timedelta(days=1)
    return sorted(universe.loc[oldest_mask, "ticker"].tolist())


def clear_cache() -> None:
    """
    Limpia la caché interna de universos cargados.

    Útil en tests para forzar una recarga del CSV después de modificarlo,
    o para liberar memoria si se trabaja con muchos índices.
    """
    _load_universe_cached.cache_clear()
    logger.debug("Caché de universos limpiada.")
