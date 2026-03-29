"""
config.py — Panel de control central del portfolio optimizer.

REGLA DE ORO: ningún módulo hardcodea parámetros.
Todo valor configurable vive aquí y solo aquí.

Este archivo crece fase a fase: solo se añaden los parámetros
que el código de esa fase necesita. No se escriben constantes
"para el futuro" que nadie usa todavía.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas base
# ---------------------------------------------------------------------------

# Directorio raíz del proyecto portfolio_optimizer/.
# __file__ apunta a este config.py, .parent es su carpeta.
BASE_DIR = Path(__file__).resolve().parent

# Ruta de la base de datos SQLite de caché de precios.
# Se crea automáticamente la primera vez que se usa.
# Por qué SQLite y no CSV: permite consultas parciales (solo los tickers
# que faltan), es transaccional (no corrompe datos si el proceso muere a
# mitad de escritura) y no requiere librerías extra.
CACHE_DB_PATH = BASE_DIR / "data" / "cache.db"

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

# Porcentaje mínimo de observaciones que debe tener un activo para
# incluirse en la optimización. Un activo con menos datos que este umbral
# se descarta con un warning.
# Por qué 0.80: si un activo cotizó solo 4 de los 5 años de lookback,
# sus estadísticas (media, varianza) son menos fiables. 80% es el mismo
# criterio que usaba el TFG original (MIN_OBS_RATIO).
MIN_OBS_RATIO = 0.80
