"""
Configuración del proyecto: ruta del Excel y parámetros de optimización.
El Excel y el código original están en esta misma carpeta (inputs/ y codigo_original/).
"""
from pathlib import Path

# Ruta del archivo Excel: copia local dentro del proyecto
_PATH_THIS = Path(__file__).resolve().parent
PATH_EXCEL = _PATH_THIS / "inputs" / "Evolucion IBEX.xlsx"

# Años hacia atrás para el análisis
AÑOS_ATRAS = 20

# Mapeo de tickers antiguos a actuales (unificación de nombres)
CAMBIOS_TICKERS = {
    "REE": "RED",
    "GAS": "NTGY",
    "GAM": "SGRE",
    "CAR": "COL",
    "SYV": "SCYR",
    "CRI": "CABK",
    "EVA": "EBRO",
    "CIN": "FER",
    "LOR": "MTS",
    "ABG": "ABG.P",
}

# Parámetros de optimización (pesos)
PESO_MIN = 0.0   # Peso mínimo por activo (0 = sin cortos)
PESO_MAX = 1.0   # Peso máximo por activo (1.0 = hasta 100% por nombre)
PESO_TOTAL = 1.0
# Concentración: suma de pesos con w > UMBRAL no puede superar PESO_CONC_MAX.
# Para que limite, usar UMBRAL < 1 (ej. 0.2) y PESO_CONC_MAX < 1 (ej. 0.8).
PESO_CONC_MAX = 1.0
UMBRAL = 1.0

# Optimización Bayesiana (gp_minimize)
OB_N_CALLS = 400
OB_N_INITIAL_POINTS = 70
OB_RANDOM_STATE = 4
OB_PACIENCIA = 100
OB_TOLERANCIA = 1e-4

# Filtro de activos: mínimo de observaciones (ratio del periodo)
MIN_OBS_RATIO = 0.8

# Optimizador: "cvxpy" (rápido, QP convexo) o "bayesian" (gp_minimize, lento pero comparable)
OPTIMIZER = "cvxpy"

# Estimar covarianza con Ledoit-Wolf (más estable) en lugar de muestral
USE_LEDOIT_WOLF = True
