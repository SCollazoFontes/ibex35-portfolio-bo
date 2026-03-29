"""
returns.py — Cálculo de rendimientos y tasa libre de riesgo.

Por qué este módulo está separado de loader.py:
    loader.py descarga precios (datos crudos).
    returns.py transforma esos precios en rendimientos (datos procesados).
    Separar responsabilidades permite testear cada capa de forma independiente
    y reutilizar la lógica de rendimientos con cualquier fuente de precios.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from portfolio_optimizer.config import MIN_OBS_RATIO, FRED_API_KEY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rendimientos diarios
# ---------------------------------------------------------------------------

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula rendimientos diarios logarítmicos a partir de precios.

    Fórmula: r_t = ln(P_t / P_{t-1})

    Por qué logarítmicos y no simples (P_t/P_{t-1} - 1):
        Los rendimientos logarítmicos son ADITIVOS en el tiempo:
            ln(P3/P1) = ln(P3/P2) + ln(P2/P1)
        Esto significa que el rendimiento acumulado de un período es
        exactamente la SUMA de los rendimientos diarios de ese período.
        Con rendimientos simples esta propiedad no se cumple de forma exacta,
        lo que introduce pequeños errores en backtests de varios años.

        Adicionalmente, los rendimientos logarítmicos son simétricos:
        una subida del 50% seguida de una bajada del 50% en log
        da exactamente 0, lo que es matemáticamente correcto.

    Args:
        prices: DataFrame con precios. Columnas = tickers, índice = fechas.

    Returns:
        DataFrame de rendimientos con una fila menos que precios.
        La primera fila de precios no tiene rendimiento anterior, se elimina.
        Los NaN por huecos en precios se propagan (no se rellenan aquí).
    """
    # np.log(prices / prices.shift(1)) calcula ln(P_t / P_{t-1}) para cada celda.
    # prices.shift(1) desplaza el DataFrame un paso hacia abajo:
    # la fila de hoy contiene los precios de ayer, por lo que la división
    # da el ratio diario de cada activo.
    returns = np.log(prices / prices.shift(1))

    # Eliminamos la primera fila: tiene NaN porque no hay P_{t-1} para el primer día.
    # Por qué dropna(how="all") y no iloc[1:]:
    #   dropna(how="all") elimina filas donde TODOS los valores son NaN,
    #   no solo la primera. Esto protege contra DataFrames con fechas faltantes
    #   al principio pero datos reales más adelante.
    returns = returns.dropna(how="all")

    return returns


# ---------------------------------------------------------------------------
# Filtrado por cobertura de datos
# ---------------------------------------------------------------------------

def filter_by_coverage(
    returns: pd.DataFrame,
    min_ratio: float | None = None,
) -> pd.DataFrame:
    """
    Elimina tickers que no tienen suficientes datos en el período.

    Un activo que cotizó solo parte del período de entrenamiento tiene
    estadísticas menos fiables (su media y varianza están calculadas
    sobre menos observaciones). Excluirlos reduce el riesgo de optimizar
    hacia activos con datos insuficientes.

    Args:
        returns:   DataFrame de rendimientos diarios.
        min_ratio: Fracción mínima de días con datos (0.0 a 1.0).
                   Si None, usa MIN_OBS_RATIO de config.py.

    Returns:
        DataFrame con solo los tickers que superan el umbral.
        El orden de las columnas se mantiene.

    Ejemplo:
        5 años de datos = ~1260 días hábiles.
        min_ratio=0.80 → el activo debe tener datos en al menos 1008 días.
        Si entró al índice en el año 3, solo tiene ~756 días → se elimina.
    """
    if min_ratio is None:
        min_ratio = MIN_OBS_RATIO

    n_total = len(returns)
    if n_total == 0:
        return returns

    # Calculamos la cobertura de cada ticker: fracción de filas no-NaN
    coverage = returns.notna().sum() / n_total

    # Separamos los tickers en dos grupos
    tickers_ok      = coverage[coverage >= min_ratio].index.tolist()
    tickers_dropped = coverage[coverage <  min_ratio].index.tolist()

    if tickers_dropped:
        # Loggeamos qué se elimina y por qué, con detalle suficiente para depurar.
        for ticker in tickers_dropped:
            pct = coverage[ticker] * 100
            days = int(coverage[ticker] * n_total)
            logger.warning(
                "filter_by_coverage: '%s' eliminado — %.1f%% de datos (%d/%d días). "
                "Umbral: %.0f%%.",
                ticker, pct, days, n_total, min_ratio * 100
            )
        logger.info(
            "filter_by_coverage: %d ticker(s) eliminados, %d conservados.",
            len(tickers_dropped), len(tickers_ok)
        )

    return returns[tickers_ok]


# ---------------------------------------------------------------------------
# Tasa libre de riesgo
# ---------------------------------------------------------------------------

# Series de FRED por mercado.
# Por qué estas series concretas:
#   ES → "IRSTCI01ESM156N": tipo de interés a 3 meses del mercado monetario
#        español. Frecuencia mensual, en % anual.
#   US → "TB3MS": T-Bill a 3 meses del Tesoro americano. Frecuencia mensual,
#        en % anual. Es la referencia estándar para el ratio de Sharpe en USD.
_FRED_SERIES = {
    "ES": "IRSTCI01ESM156N",
    "US": "TB3MS",
}

_MARKET_NAMES = {
    "ES": "España (bono soberano 3M)",
    "US": "EE.UU. (T-Bill 3M)",
}


def get_risk_free_rate(market: str, start: str, end: str) -> pd.Series:
    """
    Descarga la tasa libre de riesgo diaria del mercado correspondiente.

    Flujo:
        1. Descarga serie mensual de FRED (en % anual).
        2. Convierte a decimal: divide entre 100.
        3. Convierte a tasa diaria: divide entre 252 (días hábiles por año).
        4. Reindexar a días hábiles con forward-fill.
           Por qué forward-fill: los datos son mensuales pero necesitamos
           un valor para cada día hábil. Usamos el último valor mensual
           conocido hasta el siguiente, igual que hacía el TFG original.

    Args:
        market: "ES" para España, "US" para EE.UU.
        start:  fecha inicio "YYYY-MM-DD"
        end:    fecha fin   "YYYY-MM-DD"

    Returns:
        Serie con índice de días hábiles y tasa diaria en decimal.
        Ejemplo: 0.04 anual → 0.04/252 ≈ 0.0001587 diario.

    Raises:
        ValueError:       Si market no está en ("ES", "US").
        RuntimeError:     Si FRED_API_KEY está vacía (instrucciones de cómo obtenerla).
        ConnectionError:  Si no hay conexión a internet o FRED no responde.
    """
    if market not in _FRED_SERIES:
        raise ValueError(
            f"Mercado '{market}' no reconocido. Disponibles: {list(_FRED_SERIES.keys())}"
        )

    if not FRED_API_KEY:
        raise RuntimeError(
            "FRED_API_KEY está vacía. Para obtener una clave gratuita:\n"
            "  1. Ve a https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "  2. Crea una cuenta gratuita\n"
            "  3. Copia tu API key\n"
            "  4. Añádela en portfolio_optimizer/config.py: FRED_API_KEY = 'tu_key'\n"
        )

    series_id = _FRED_SERIES[market]
    market_name = _MARKET_NAMES[market]

    logger.info(
        "Descargando tasa libre de riesgo para %s: serie FRED '%s' (%s a %s)",
        market_name, series_id, start, end
    )

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)

        # observation_start/end filtran el rango en la descarga.
        raw = fred.get_series(
            series_id,
            observation_start=start,
            observation_end=end,
        )

    except Exception as e:
        raise ConnectionError(
            f"No se pudo descargar la tasa libre de riesgo de FRED "
            f"(serie '{series_id}'): {e}"
        ) from e

    if raw.empty:
        raise ValueError(
            f"FRED no devolvió datos para la serie '{series_id}' "
            f"en el rango {start} – {end}."
        )

    # Convertimos % anual → decimal → tasa diaria
    # Ejemplo: 4.0% anual → 0.04 decimal → 0.04/252 ≈ 0.000159 diario
    rf_decimal = raw / 100.0
    rf_daily   = rf_decimal / 252.0

    # Reindexamos a días hábiles rellenando hacia adelante (forward-fill).
    # Por qué forward-fill y no interpolación:
    #   El tipo de interés del banco central es constante entre anuncios.
    #   No tiene sentido interpolar linealmente entre dos decisiones del BCE.
    #   Forward-fill replica el comportamiento real: el tipo de hoy es el
    #   último tipo publicado.
    business_days = pd.bdate_range(start=start, end=end)
    rf_daily = rf_daily.reindex(business_days).ffill()

    # Rellenamos posibles NaN al principio si la serie empieza después de start
    # (puede pasar con algunas series de FRED que tienen retrasos en publicación).
    rf_daily = rf_daily.bfill()

    rf_daily.name = f"rf_{market.lower()}_daily"

    logger.info(
        "Tasa libre de riesgo descargada: %.4f%% diario (media del período)",
        rf_daily.mean() * 100
    )

    return rf_daily
