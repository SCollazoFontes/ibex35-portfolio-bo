"""
cache.py — Caché local de precios en SQLite.

Por qué existe este módulo:
    yfinance tiene límites de peticiones y es lento para rangos largos.
    Sin caché, cada ejecución del backtest descarga años de datos desde internet.
    Con caché, la segunda ejecución es instantánea.

Por qué SQLite y no CSV/pickle:
    - Consultas parciales: puedo pedir solo los tickers que faltan.
    - Transaccional: si el proceso muere a mitad de escritura, los datos
      existentes no se corrompen (a diferencia de un CSV que se sobreescribe).
    - Sin dependencias extra: sqlite3 es parte de la librería estándar de Python.

Esquema de la tabla 'prices':
    ticker  TEXT   — identificador del activo (ej: "SAN.MC")
    date    TEXT   — fecha en formato ISO "YYYY-MM-DD"
    close   REAL   — precio de cierre ajustado (total return)
    source  TEXT   — fuente del dato: "yfinance" o "stooq"

    Clave primaria compuesta (ticker, date): garantiza que no haya
    duplicados si se llama a save_to_cache() varias veces con los mismos datos.
"""

# from __future__ import annotations permite usar la sintaxis de type hints
# de Python 3.10+ (list[str], X | None) en Python 3.9.
# Por qué: el venv usa Python 3.9.6 pero la sintaxis moderna es más legible.
from __future__ import annotations

import sqlite3
import logging
from pathlib import Path

import pandas as pd

# Importamos la ruta de la BD desde config.py.
# Por qué no hardcodear la ruta aquí: si en el futuro quieres tener
# una caché de tests separada de la de producción, solo cambias config.py.
from portfolio_optimizer.config import CACHE_DB_PATH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """
    Abre (o crea) la base de datos SQLite y garantiza que la tabla
    'prices' existe.

    Por qué una función privada y no una conexión global:
        Una conexión global en Python tiene problemas con multihilo.
        Abrir/cerrar la conexión en cada llamada es ligeramente más lento
        pero completamente seguro y simple. Para el volumen de datos de
        este proyecto (miles de filas, no millones) la diferencia es
        imperceptible.
    """
    # Creamos el directorio padre si no existe.
    # Por qué exist_ok=True: si ya existe, no lanza error.
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(CACHE_DB_PATH)

    # CREATE TABLE IF NOT EXISTS: solo crea si no existe ya.
    # Por qué PRIMARY KEY (ticker, date): evita duplicados automáticamente.
    # Si intentas insertar el mismo (ticker, date) dos veces, SQLite lo ignora
    # gracias al INSERT OR IGNORE que usamos en save_to_cache().
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker  TEXT NOT NULL,
            date    TEXT NOT NULL,
            close   REAL NOT NULL,
            source  TEXT NOT NULL,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def get_cached(tickers: list[str], start: str, end: str) -> pd.DataFrame | None:
    """
    Devuelve los datos cacheados si existen y están completos. None si no.

    'Completo' significa: todos los tickers tienen datos en el rango [start, end].
    Si un solo ticker falta completamente, devuelve None para ese ticker.

    Por qué devolver None en vez de un DataFrame parcial:
        El llamador (loader.py) necesita saber exactamente qué tickers
        descargar. Devolver None para cada ticker ausente es más claro
        que devolver un DataFrame con huecos y que el llamador tenga que
        detectarlos.

    Args:
        tickers: lista de tickers a buscar
        start:   fecha inicio "YYYY-MM-DD" (inclusiva)
        end:     fecha fin   "YYYY-MM-DD" (inclusiva)

    Returns:
        DataFrame [fecha × ticker] con precios, o None si algún ticker falta.
        Si todos los tickers están en caché: DataFrame completo.
        Si alguno falta: None (el llamador descargará los que faltan).
    """
    conn = _get_connection()
    try:
        # Leemos todos los tickers de una vez con una sola query.
        # Por qué usar placeholders (?) en vez de f-strings:
        # previene SQL injection, aunque aquí los tickers vengan de código
        # propio y no de input de usuario. Es buena práctica siempre.
        placeholders = ",".join("?" * len(tickers))
        query = f"""
            SELECT ticker, date, close
            FROM prices
            WHERE ticker IN ({placeholders})
              AND date >= ?
              AND date <= ?
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn, params=[*tickers, start, end])

        if df.empty:
            return None

        # Comprobamos que todos los tickers solicitados tienen algún dato.
        # Por qué no comparar número de filas exacto:
        # los días de mercado no son predecibles (festivos varían por año),
        # así que no podemos saber cuántas filas "deberían" estar.
        # Lo que sí podemos verificar es que cada ticker aparece al menos una vez.
        tickers_en_cache = set(df["ticker"].unique())
        tickers_solicitados = set(tickers)
        tickers_faltantes = tickers_solicitados - tickers_en_cache

        if tickers_faltantes:
            logger.debug(
                "Cache miss para %d ticker(s): %s",
                len(tickers_faltantes),
                sorted(tickers_faltantes),
            )
            # Devolvemos None para indicar que hay que descargar los que faltan.
            # El llamador es responsable de descargar solo los ausentes.
            return None

        # Pivotamos de formato largo (una fila por ticker×fecha)
        # a formato ancho (columnas = tickers, índice = fechas).
        # Por qué este formato: es el formato estándar que esperan todos
        # los módulos de optimización (renders diarios, covarianzas, etc.).
        result = df.pivot(index="date", columns="ticker", values="close")
        result.index = pd.to_datetime(result.index)
        result.index.name = "date"

        # Reordenamos las columnas para que coincidan con el orden de la lista
        # de entrada. Facilita la comparación en tests.
        result = result.reindex(columns=tickers)

        logger.info("Cache hit: %d tickers, %d días", len(tickers), len(result))
        return result

    finally:
        conn.close()


def save_to_cache(df: pd.DataFrame, source: str) -> None:
    """
    Guarda un DataFrame de precios en la caché SQLite.

    Args:
        df:     DataFrame con índice de fechas y columnas = tickers.
                Mismo formato que devuelve get_prices() y get_cached().
        source: fuente de los datos, "yfinance" o "stooq".

    Por qué INSERT OR IGNORE y no INSERT OR REPLACE:
        INSERT OR REPLACE borraría y reinsertaría la fila, cambiando el rowid
        y siendo más lento. INSERT OR IGNORE simplemente salta duplicados,
        que es lo que queremos: si el dato ya está, no lo tocamos.
    """
    conn = _get_connection()
    try:
        # Convertimos el DataFrame ancho (columnas = tickers) a formato largo
        # (una fila por ticker×fecha) para almacenar en la tabla relacional.
        # Por qué formato largo en BD: es más eficiente para queries parciales
        # ("dame solo SAN.MC entre 2020 y 2022").
        df_long = df.stack().reset_index()
        df_long.columns = ["date", "ticker", "close"]
        df_long["date"] = df_long["date"].dt.strftime("%Y-%m-%d")
        df_long["source"] = source

        # Eliminamos NaN antes de insertar.
        # Por qué: yfinance puede devolver NaN para días festivos o
        # datos faltantes. No tiene sentido cachear ausencias.
        df_long = df_long.dropna(subset=["close"])

        conn.executemany(
            "INSERT OR IGNORE INTO prices (ticker, date, close, source) VALUES (?,?,?,?)",
            df_long[["ticker", "date", "close", "source"]].values.tolist(),
        )
        conn.commit()
        logger.info(
            "Guardadas %d filas en caché (fuente: %s)", len(df_long), source
        )
    finally:
        conn.close()


def invalidate(tickers: list[str] | None = None) -> None:
    """
    Invalida la caché. Si tickers=None, borra todo.

    Útil cuando sabes que los datos de un ticker concreto están corruptos
    o cuando quieres forzar una re-descarga completa.

    Args:
        tickers: lista de tickers a invalidar, o None para borrar todo.
    """
    conn = _get_connection()
    try:
        if tickers is None:
            conn.execute("DELETE FROM prices")
            logger.warning("Caché completamente vaciada.")
        else:
            placeholders = ",".join("?" * len(tickers))
            conn.execute(
                f"DELETE FROM prices WHERE ticker IN ({placeholders})", tickers
            )
            logger.info("Caché invalidada para: %s", tickers)
        conn.commit()
    finally:
        conn.close()
