"""
conftest.py — Fixtures compartidas entre todos los tests del proyecto.

Por qué conftest.py y no importar fixtures directamente:
    pytest descubre conftest.py automáticamente en el directorio de tests
    y lo pone a disposición de todos los archivos de test sin que hagan
    ningún import. Es el mecanismo estándar de pytest para compartir fixtures.

Criterios de los datos sintéticos:
    - Estructura realista: correlaciones no triviales, volatilidad variable.
    - Sin red: todo generado con numpy.random, sin llamadas a yfinance.
    - Reproducibles: np.random.seed fijo → los tests siempre dan el mismo resultado.
    - Tamaño manejable: 500 días × 5 tickers. Suficiente para probar la lógica,
      rápido de crear en cada test.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Datos de precios sintéticos
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """
    DataFrame de precios sintéticos con 5 tickers y 500 días de negociación.

    Estructura del dato sintético:
        - 5 tickers: SAN.MC, ITX.MC, IBE.MC, TEF.MC, REP.MC
        - 500 días desde 2018-01-01 (solo días hábiles)
        - Precios generados con paseo aleatorio (random walk) correlacionado

    Por qué paseo aleatorio correlacionado y no precios independientes:
        Los activos reales tienen correlaciones entre sí. Si los datos de test
        son independientes, algunos tests de covarianza y Sharpe no detectarían
        errores que sí ocurrirían con datos reales.

    Por qué 500 días:
        Es suficiente para calcular estadísticas anualizadas (necesitamos ~252)
        y probar ventanas deslizantes, sin ser tan grande que ralentice los tests.
    """
    np.random.seed(42)  # Semilla fija → tests reproducibles

    tickers = ["SAN.MC", "ITX.MC", "IBE.MC", "TEF.MC", "REP.MC"]
    n_days = 500

    # Fechas: solo días hábiles (freq="B" = business days).
    # Por qué días hábiles y no calendario: los mercados no cotizan fines de
    # semana ni festivos. Usar días naturales crearía filas con NaN que
    # complicarían los tests sin añadir valor.
    dates = pd.bdate_range(start="2018-01-01", periods=n_days)

    # Generamos rendimientos diarios con estructura de correlación realista.
    # La matriz de correlación tiene correlaciones positivas entre todos los
    # activos (como ocurre en la realidad para acciones del mismo índice).
    #
    # Construcción de la matriz de correlación:
    #   - Diagonal = 1 (cada activo correlaciona 1 consigo mismo)
    #   - Fuera de diagonal = 0.3 (correlación moderada entre todos los pares)
    n_assets = len(tickers)
    correlation = np.full((n_assets, n_assets), 0.3)
    np.fill_diagonal(correlation, 1.0)

    # Volatilidades diarias distintas para cada activo (más realismo).
    # Un banco (SAN) tiene más vol que una utility (IBE).
    daily_vols = np.array([0.015, 0.012, 0.010, 0.014, 0.016])

    # Convertimos la correlación en covarianza: Σ = D * ρ * D
    # donde D es la matriz diagonal de desviaciones estándar.
    D = np.diag(daily_vols)
    cov = D @ correlation @ D

    # np.random.multivariate_normal genera vectores de rendimientos correlacionados.
    # media=0: sin tendencia, para que los tests de rendimiento sean predecibles.
    returns = np.random.multivariate_normal(mean=np.zeros(n_assets), cov=cov, size=n_days)

    # Convertimos rendimientos en precios: P_t = P_{t-1} * (1 + r_t)
    # Precio inicial = 10 para todos (arbitrario, los rendimientos no dependen de esto).
    prices = 10 * np.cumprod(1 + returns, axis=0)

    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def sample_prices_with_gaps(sample_prices) -> pd.DataFrame:
    """
    Igual que sample_prices pero con un ticker que tiene solo el 70% de los datos.

    Útil para testear filter_by_coverage(): el ticker "TEF.MC" debe ser
    eliminado con MIN_OBS_RATIO=0.80 porque solo tiene el 70% de observaciones.

    Por qué TEF.MC: es el cuarto ticker. Elegirlo facilita verificar que
    los otros 4 se mantienen intactos.
    """
    prices_with_gaps = sample_prices.copy()

    # Ponemos NaN en el 30% de las filas de TEF.MC.
    # np.random.choice selecciona el 30% de los índices de fila sin reemplazo.
    n_rows = len(prices_with_gaps)
    gap_indices = np.random.choice(n_rows, size=int(n_rows * 0.30), replace=False)
    prices_with_gaps.iloc[gap_indices, prices_with_gaps.columns.get_loc("TEF.MC")] = np.nan

    return prices_with_gaps


@pytest.fixture
def tmp_cache_path(tmp_path) -> object:
    """
    Ruta temporal para la BD de caché durante los tests.

    tmp_path es un fixture de pytest que crea un directorio temporal
    que se borra automáticamente al terminar el test.

    Por qué una caché temporal y no la de producción:
        Los tests no deben modificar datos reales. Si un test corrompe la
        caché de producción, el siguiente backtest real fallaría.
        Con tmp_path cada test empieza con una caché limpia.
    """
    return tmp_path / "test_cache.db"
