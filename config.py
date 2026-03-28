from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PATH_EXCEL = Path(__file__).resolve().parent / "inputs" / "Evolucion IBEX.xlsx"

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
AÑO_INICIO = 2010
AÑO_FINAL  = 2023
AÑOS_ATRAS = 5

# ---------------------------------------------------------------------------
# Cambios de tickers históricos
# Unifica nombres que han cambiado a lo largo del tiempo
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Restricciones de pesos de la cartera
# ---------------------------------------------------------------------------
PESO_MIN      = 0.0   # Peso mínimo por activo (0 = sin posiciones cortas)
PESO_MAX      = 1.0   # Peso máximo por activo
PESO_TOTAL    = 1.0   # La cartera debe estar 100% invertida
PESO_CONC_MAX = 1.0   # Peso máximo acumulado de los activos con peso > UMBRAL
UMBRAL        = 1.0   # Umbral a partir del cual se mide la concentración

# ---------------------------------------------------------------------------
# Filtro de activos
# Activos con menos del MIN_OBS_RATIO del periodo disponible se descartan
# ---------------------------------------------------------------------------
MIN_OBS_RATIO = 0.8

# ---------------------------------------------------------------------------
# Display
# Activos con peso inferior a LIM_INF no se muestran en las tablas de pesos
# ---------------------------------------------------------------------------
LIM_INF = 0.01

# ---------------------------------------------------------------------------
# Optimización Bayesiana (gp_minimize)
# ---------------------------------------------------------------------------
OB_N_CALLS         = 400
OB_N_INITIAL_POINTS = 70
OB_RANDOM_STATE    = 4
OB_PACIENCIA       = 100    # Iteraciones sin mejora antes de parar
OB_TOLERANCIA      = 1e-4   # Mejora mínima para resetear la paciencia

# ---------------------------------------------------------------------------
# Volatilidad Estocástica - Monte Carlo
# ---------------------------------------------------------------------------
VE_N_SIM = 500   # Número de simulaciones para promediar la covarianza
