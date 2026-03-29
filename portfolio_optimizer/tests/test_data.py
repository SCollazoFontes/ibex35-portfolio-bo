"""
test_data.py — Tests para data/loader.py, data/cache.py y data/universe.py.

Convenciones:
    - Tests sin red: funcionan siempre, usan mocks o datos sintéticos.
    - Tests con red: marcados con @pytest.mark.network, se saltan con
      pytest -m "not network".

Por qué tests con mocks en vez de datos reales:
    Los datos reales de mercado cambian cada día. Un test que comprueba
    "SAN.MC tenía precio X el día Y" puede empezar a fallar si yfinance
    cambia su API o ajusta datos históricos. Los mocks son deterministas.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Configuración del marker de red para pytest
# ---------------------------------------------------------------------------
# Registramos el marker "network" para que pytest no lo marque como warning
# de marker desconocido. El archivo pytest.ini o conftest.py del proyecto
# debería incluir esto, pero lo añadimos aquí por claridad.


# ---------------------------------------------------------------------------
# Tests de data/cache.py
# ---------------------------------------------------------------------------

class TestCache:
    """Tests del módulo de caché SQLite."""

    def test_cache_roundtrip(self, sample_prices, tmp_cache_path, monkeypatch):
        """
        Guarda datos en caché y los recupera idénticos.

        Este es el test más fundamental de cache.py: si el roundtrip no funciona,
        nada más funciona.
        """
        # Apuntamos la caché al directorio temporal para no tocar la de producción.
        import portfolio_optimizer.data.cache as cache_module
        monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_cache_path)

        tickers = list(sample_prices.columns)
        start = sample_prices.index[0].strftime("%Y-%m-%d")
        end = sample_prices.index[-1].strftime("%Y-%m-%d")

        # Guardamos
        cache_module.save_to_cache(sample_prices, source="yfinance")

        # Recuperamos
        result = cache_module.get_cached(tickers, start, end)

        assert result is not None, "get_cached() debería devolver datos después de save_to_cache()"

        # Verificamos que los tickers están todos presentes
        assert set(result.columns) == set(tickers)

        # Verificamos que los valores son iguales (tolerancia numérica por float→text→float)
        for ticker in tickers:
            original = sample_prices[ticker].dropna()
            cached = result[ticker].dropna()
            np.testing.assert_allclose(
                original.values, cached.values, rtol=1e-6,
                err_msg=f"Precios de {ticker} no coinciden tras roundtrip de caché"
            )

    def test_cache_miss_returns_none(self, tmp_cache_path, monkeypatch):
        """
        Si los datos no están en caché, get_cached() devuelve None.
        """
        import portfolio_optimizer.data.cache as cache_module
        monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_cache_path)

        result = cache_module.get_cached(["SAN.MC", "ITX.MC"], "2020-01-01", "2020-12-31")
        assert result is None

    def test_cache_miss_for_unknown_ticker(self, sample_prices, tmp_cache_path, monkeypatch):
        """
        Si pedimos un ticker que no está en caché, aunque otros sí estén, devuelve None.

        Por qué este test: queremos asegurarnos de que la caché parcial
        (algunos tickers sí, otros no) fuerza una re-descarga completa.
        """
        import portfolio_optimizer.data.cache as cache_module
        monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_cache_path)

        # Guardamos solo SAN.MC e ITX.MC
        cache_module.save_to_cache(sample_prices[["SAN.MC", "ITX.MC"]], source="yfinance")

        # Pedimos SAN.MC, ITX.MC e IBE.MC (este último no está)
        start = sample_prices.index[0].strftime("%Y-%m-%d")
        end = sample_prices.index[-1].strftime("%Y-%m-%d")
        result = cache_module.get_cached(["SAN.MC", "ITX.MC", "IBE.MC"], start, end)

        assert result is None, "Debe devolver None si falta aunque sea un ticker"

    def test_invalidate_clears_all(self, sample_prices, tmp_cache_path, monkeypatch):
        """
        invalidate() sin argumentos borra todos los datos de la caché.
        """
        import portfolio_optimizer.data.cache as cache_module
        monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_cache_path)

        tickers = list(sample_prices.columns)
        start = sample_prices.index[0].strftime("%Y-%m-%d")
        end = sample_prices.index[-1].strftime("%Y-%m-%d")

        cache_module.save_to_cache(sample_prices, source="yfinance")
        cache_module.invalidate()  # Borra todo

        result = cache_module.get_cached(tickers, start, end)
        assert result is None, "Después de invalidate() total, no debe haber nada en caché"

    def test_invalidate_specific_ticker(self, sample_prices, tmp_cache_path, monkeypatch):
        """
        invalidate(["SAN.MC"]) borra solo ese ticker, el resto permanece.
        """
        import portfolio_optimizer.data.cache as cache_module
        monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_cache_path)

        start = sample_prices.index[0].strftime("%Y-%m-%d")
        end = sample_prices.index[-1].strftime("%Y-%m-%d")

        cache_module.save_to_cache(sample_prices, source="yfinance")
        cache_module.invalidate(["SAN.MC"])

        # SAN.MC borrado → pedir solo SAN.MC debe dar None
        result_san = cache_module.get_cached(["SAN.MC"], start, end)
        assert result_san is None

        # ITX.MC intacto → debe devolver datos
        result_itx = cache_module.get_cached(["ITX.MC"], start, end)
        assert result_itx is not None

    def test_duplicate_insert_ignored(self, sample_prices, tmp_cache_path, monkeypatch):
        """
        Guardar los mismos datos dos veces no duplica filas ni lanza error.

        Por qué: usamos INSERT OR IGNORE. Este test verifica que la lógica
        de deduplicación funciona correctamente.
        """
        import portfolio_optimizer.data.cache as cache_module
        import sqlite3
        monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_cache_path)

        cache_module.save_to_cache(sample_prices, source="yfinance")
        cache_module.save_to_cache(sample_prices, source="yfinance")  # Segunda vez

        # Contamos filas directamente en SQLite
        conn = sqlite3.connect(tmp_cache_path)
        count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        conn.close()

        # Número esperado de filas: días con datos × tickers
        # (dropna() en save_to_cache elimina NaN, así que contamos los no-NaN)
        expected = sample_prices.notna().sum().sum()
        assert count == expected, (
            f"Esperadas {expected} filas únicas, encontradas {count}. "
            "Puede indicar que INSERT OR IGNORE no funciona."
        )


# ---------------------------------------------------------------------------
# Tests de data/loader.py
# ---------------------------------------------------------------------------

class TestLoader:
    """Tests del módulo de descarga de precios."""

    def _make_fake_yfinance_response(self, tickers: list[str], n_days: int = 252) -> pd.DataFrame:
        """
        Crea un DataFrame sintético que simula lo que devuelve yf.download().

        yfinance con auto_adjust=True devuelve columnas MultiIndex:
            (Open, ticker), (High, ticker), (Close, ticker), ...
        """
        dates = pd.bdate_range("2020-01-01", periods=n_days)
        np.random.seed(0)

        # Simulamos el MultiIndex que devuelve yfinance
        data = {}
        for ticker in tickers:
            prices = 10 * np.cumprod(1 + np.random.normal(0, 0.01, n_days))
            data[("Close", ticker)] = prices

        df = pd.DataFrame(data, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    def test_get_prices_returns_dataframe(self, tmp_cache_path, monkeypatch):
        """
        get_prices() devuelve un DataFrame con el formato correcto:
            - índice DatetimeIndex
            - columnas = tickers solicitados (los que tuvieron éxito)
            - valores numéricos positivos
        """
        import portfolio_optimizer.data.cache as cache_module
        import portfolio_optimizer.data.loader as loader_module
        monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_cache_path)

        tickers = ["SAN.MC", "ITX.MC"]
        fake_response = self._make_fake_yfinance_response(tickers)

        with patch("yfinance.download", return_value=fake_response):
            result = loader_module.get_prices(tickers, "2020-01-01", "2020-12-31")

        assert isinstance(result, pd.DataFrame), "Debe devolver un DataFrame"
        assert isinstance(result.index, pd.DatetimeIndex), "El índice debe ser DatetimeIndex"
        assert set(result.columns) == set(tickers), "Las columnas deben ser los tickers"
        assert (result > 0).all().all(), "Los precios deben ser positivos"

    def test_get_prices_uses_cache_on_second_call(self, sample_prices, tmp_cache_path, monkeypatch):
        """
        La segunda llamada a get_prices() con los mismos parámetros usa la caché
        y no llama a yfinance.

        Por qué es importante este test: el propósito de la caché es evitar
        llamadas repetidas a la red. Si la caché no se usa, el módulo es inútil.
        """
        import portfolio_optimizer.data.cache as cache_module
        import portfolio_optimizer.data.loader as loader_module
        monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_cache_path)

        tickers = list(sample_prices.columns)
        start = sample_prices.index[0].strftime("%Y-%m-%d")
        end = sample_prices.index[-1].strftime("%Y-%m-%d")

        # Precargamos la caché manualmente
        cache_module.save_to_cache(sample_prices, source="yfinance")

        # La llamada a get_prices NO debe llamar a yfinance
        with patch("yfinance.download") as mock_yf:
            result = loader_module.get_prices(tickers, start, end)
            mock_yf.assert_not_called()

        assert result is not None
        assert set(result.columns) == set(tickers)

    def test_fallback_to_stooq_on_yfinance_failure(self, tmp_cache_path, monkeypatch):
        """
        Cuando yfinance falla para un ticker, se usa Stooq como fallback.

        Este test mockea yfinance para que devuelva un DataFrame vacío
        y comprueba que se llama a pandas_datareader con el ticker correcto.
        """
        import portfolio_optimizer.data.cache as cache_module
        import portfolio_optimizer.data.loader as loader_module
        monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_cache_path)

        tickers = ["SAN.MC"]
        dates = pd.bdate_range("2020-01-01", periods=252)
        fake_stooq = pd.DataFrame(
            {"Close": 10 * np.cumprod(1 + np.random.normal(0, 0.01, 252))},
            index=dates,
        )

        # yfinance falla (devuelve vacío), Stooq tiene éxito
        with patch("yfinance.download", return_value=pd.DataFrame()), \
             patch("pandas_datareader.data.DataReader", return_value=fake_stooq) as mock_stooq:

            result = loader_module.get_prices(tickers, "2020-01-01", "2020-12-31")

            # Verificamos que se llamó a Stooq
            mock_stooq.assert_called_once()
            call_args = mock_stooq.call_args
            assert call_args[0][1] == "stooq", "Debe llamar a DataReader con fuente 'stooq'"

        assert "SAN.MC" in result.columns, "El ticker debe estar en el resultado aunque viniera de Stooq"

    def test_failed_ticker_excluded_gracefully(self, tmp_cache_path, monkeypatch):
        """
        Un ticker que falla en yfinance Y en Stooq se excluye sin romper
        la descarga de los demás.

        Por qué este test: un ticker corrupto no debe interrumpir el backtest.
        """
        import portfolio_optimizer.data.cache as cache_module
        import portfolio_optimizer.data.loader as loader_module
        monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_cache_path)

        tickers = ["SAN.MC", "TICKER_INEXISTENTE"]
        dates = pd.bdate_range("2020-01-01", periods=252)

        # SAN.MC tiene datos, TICKER_INEXISTENTE no
        def fake_download(ticker, *args, **kwargs):
            if ticker == "SAN.MC":
                prices = 10 * np.cumprod(1 + np.random.normal(0, 0.01, 252))
                df = pd.DataFrame(
                    {("Close", "SAN.MC"): prices},
                    index=dates,
                )
                df.columns = pd.MultiIndex.from_tuples(df.columns)
                return df
            return pd.DataFrame()  # Falla para el ticker inexistente

        with patch("yfinance.download", side_effect=fake_download), \
             patch("pandas_datareader.data.DataReader", side_effect=Exception("Not found")):

            result = loader_module.get_prices(tickers, "2020-01-01", "2020-12-31")

        assert "SAN.MC" in result.columns, "SAN.MC debe estar en el resultado"
        assert "TICKER_INEXISTENTE" not in result.columns, "El ticker fallido debe excluirse"

    def test_stooq_ticker_format_us(self):
        """
        Para mercado US, el ticker se convierte al formato de Stooq agregando '.US'.
        """
        from portfolio_optimizer.data.loader import _ticker_to_stooq

        assert _ticker_to_stooq("AAPL", "US") == "AAPL.US"
        assert _ticker_to_stooq("MSFT", "US") == "MSFT.US"

    def test_stooq_ticker_format_es(self):
        """
        Para mercado ES, el ticker no cambia (ya tiene sufijo .MC).
        """
        from portfolio_optimizer.data.loader import _ticker_to_stooq

        assert _ticker_to_stooq("SAN.MC", "ES") == "SAN.MC"
        assert _ticker_to_stooq("ITX.MC", "ES") == "ITX.MC"

    @pytest.mark.network
    def test_real_download_ibex_ticker(self, tmp_cache_path, monkeypatch):
        """
        Test de integración real: descarga SAN.MC de yfinance.
        Requiere conexión a internet. Se salta con: pytest -m "not network"
        """
        import portfolio_optimizer.data.cache as cache_module
        import portfolio_optimizer.data.loader as loader_module
        monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_cache_path)

        result = loader_module.get_prices(["SAN.MC"], "2023-01-01", "2023-12-31")

        assert not result.empty, "La descarga real de SAN.MC no debe devolver vacío"
        assert "SAN.MC" in result.columns
        assert len(result) > 200, "Un año de trading debe tener ~252 días hábiles"
        assert (result["SAN.MC"] > 0).all(), "Los precios deben ser positivos"


# ---------------------------------------------------------------------------
# Tests de data/universe.py
# ---------------------------------------------------------------------------

class TestUniverse:
    """
    Tests del módulo de composición histórica de índices.

    Estos tests usan el CSV real de ibex35.csv porque su contenido
    es estático (datos históricos que no cambian). No hace falta mockear.
    """

    def setup_method(self):
        """Limpiamos la caché antes de cada test para evitar estado compartido."""
        from portfolio_optimizer.data.universe import clear_cache
        clear_cache()

    def test_get_components_returns_list(self):
        """get_components() devuelve una lista no vacía."""
        from portfolio_optimizer.data.universe import get_components
        result = get_components("IBEX35", "2015-01-01")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_components_known_date(self):
        """
        Para una fecha conocida, verifica que hay aproximadamente 35 tickers.

        Por qué ~35 y no exactamente 35: puede haber fechas en las que el
        índice tenga 34 o 36 si un cambio se produjo exactamente ese día.
        """
        from portfolio_optimizer.data.universe import get_components
        result = get_components("IBEX35", "2020-01-02")
        assert 30 <= len(result) <= 36, (
            f"Se esperan ~35 tickers, encontrados {len(result)}: {result}"
        )

    def test_get_components_tickers_have_mc_suffix(self):
        """
        Todos los tickers devueltos tienen el sufijo .MC (formato yfinance Madrid).

        Por qué este test: si falta el sufijo, yfinance no encuentra el activo.
        """
        from portfolio_optimizer.data.universe import get_components
        result = get_components("IBEX35", "2018-01-02")
        for ticker in result:
            assert ticker.endswith(".MC"), (
                f"Ticker '{ticker}' no tiene sufijo .MC — yfinance no lo encontrará"
            )

    def test_ree_present_before_rename(self):
        """
        REE.MC aparece antes de su salida (junio 2022) y RED.MC después.

        Este es el test de cambio de ticker más importante del IBEX:
        Red Eléctrica se llamaba REE y pasó a llamarse RED en junio 2022.
        Si el backtest usa REE en 2023, yfinance no encontrará datos.
        """
        from portfolio_optimizer.data.universe import get_components

        # Antes del cambio: REE debe estar, RED no
        before = get_components("IBEX35", "2022-01-01")
        assert "REE.MC" in before, "REE.MC debe estar en el índice antes de junio 2022"
        assert "RED.MC" not in before, "RED.MC no debe estar antes de junio 2022"

        # Después del cambio: RED debe estar, REE no
        after = get_components("IBEX35", "2023-01-01")
        assert "RED.MC" in after, "RED.MC debe estar en el índice después de junio 2022"
        assert "REE.MC" not in after, "REE.MC no debe estar después de junio 2022"

    def test_stable_tickers_always_present(self):
        """
        Tickers que han estado siempre en el IBEX 35 desde 2005 deben aparecer
        en cualquier fecha del rango.

        Por qué este test: verifica que la lógica de date_in/date_out no
        excluye accidentalmente tickers que deberían estar siempre.
        """
        from portfolio_optimizer.data.universe import get_components

        always_present = ["SAN.MC", "BBVA.MC", "IBE.MC", "ITX.MC", "TEF.MC"]
        test_dates = ["2008-01-02", "2012-06-01", "2017-03-15", "2021-09-01"]

        for date in test_dates:
            result = get_components("IBEX35", date)
            for ticker in always_present:
                assert ticker in result, (
                    f"{ticker} debería estar en el IBEX35 en {date} pero no aparece"
                )

    def test_excluded_ticker_not_present_after_exit(self):
        """
        Un ticker excluido no aparece en fechas posteriores a su salida.

        Ejemplo: Banco Popular (POP.MC) fue resuelto en junio 2017.
        No debe aparecer en el índice en 2018.
        """
        from portfolio_optimizer.data.universe import get_components

        result_2018 = get_components("IBEX35", "2018-01-02")
        assert "POP.MC" not in result_2018, (
            "POP.MC fue resuelto en junio 2017 y no debe aparecer en 2018"
        )

    def test_invalid_index_raises(self):
        """
        Un índice no reconocido lanza ValueError con mensaje claro.
        """
        from portfolio_optimizer.data.universe import get_components
        with pytest.raises(ValueError, match="no reconocido"):
            get_components("DAX40", "2020-01-01")

    def test_results_sorted_alphabetically(self):
        """
        Los tickers devueltos están ordenados alfabéticamente.

        Por qué importa: el orden determinista facilita los tests y hace
        que las matrices de covarianza sean reproducibles entre ejecuciones.
        """
        from portfolio_optimizer.data.universe import get_components
        result = get_components("IBEX35", "2019-06-01")
        assert result == sorted(result), "Los tickers deben estar ordenados alfabéticamente"


# ---------------------------------------------------------------------------
# Tests de data/returns.py
# ---------------------------------------------------------------------------

class TestReturns:
    """Tests del módulo de cálculo de rendimientos."""

    def test_daily_returns_shape(self, sample_prices):
        """
        El DataFrame de rendimientos tiene exactamente una fila menos que los precios.

        Por qué una fila menos: el primer rendimiento requiere dos precios
        (P_0 y P_1). P_0 no tiene P_{-1}, así que no hay rendimiento para él.
        """
        from portfolio_optimizer.data.returns import daily_returns

        result = daily_returns(sample_prices)
        assert result.shape == (len(sample_prices) - 1, sample_prices.shape[1]), (
            f"Esperado {len(sample_prices) - 1} filas, obtenido {result.shape[0]}"
        )

    def test_daily_returns_are_logarithmic(self, sample_prices):
        """
        Verifica que el cálculo es ln(P_t / P_{t-1}), no (P_t - P_{t-1}) / P_{t-1}.

        Por qué testear la fórmula exacta y no solo el resultado:
            Un error de implementación (usar returns simples en vez de log)
            daría valores muy similares para retornos pequeños pero erróneos.
            Este test calcula el valor esperado manualmente y lo compara.
        """
        from portfolio_optimizer.data.returns import daily_returns

        result = daily_returns(sample_prices)

        # Calculamos el valor esperado para la primera fila y el primer ticker
        ticker = sample_prices.columns[0]
        p0 = sample_prices[ticker].iloc[0]
        p1 = sample_prices[ticker].iloc[1]
        expected = np.log(p1 / p0)

        actual = result[ticker].iloc[0]
        np.testing.assert_almost_equal(
            actual, expected, decimal=10,
            err_msg=f"Se esperaba ln({p1:.4f}/{p0:.4f})={expected:.6f}, obtenido {actual:.6f}"
        )

    def test_daily_returns_not_simple(self, sample_prices):
        """
        Verifica que NO se usan rendimientos simples (P_t/P_{t-1} - 1).

        Para retornos pequeños la diferencia entre log y simple es mínima,
        pero para retornos grandes (>5%) la diferencia es apreciable.
        Creamos un caso con retorno grande para que el test sea robusto.
        """
        from portfolio_optimizer.data.returns import daily_returns

        # Creamos precios con un salto grande: de 100 a 120 (+20%)
        prices_big_move = pd.DataFrame(
            {"A": [100.0, 120.0, 144.0]},
            index=pd.bdate_range("2020-01-01", periods=3)
        )
        result = daily_returns(prices_big_move)

        expected_log    = np.log(120 / 100)   # ≈ 0.1823
        wrong_simple    = (120 - 100) / 100   # = 0.2000

        actual = result["A"].iloc[0]
        np.testing.assert_almost_equal(actual, expected_log, decimal=10)
        assert abs(actual - wrong_simple) > 0.01, (
            "El valor parece ser un rendimiento simple, no logarítmico"
        )

    def test_daily_returns_no_nan_first_row(self, sample_prices):
        """
        La primera fila del resultado no contiene NaN.

        La fila con NaN (la primera, que no tiene P_{t-1}) debe eliminarse,
        no quedar como fila de NaN en el resultado.
        """
        from portfolio_optimizer.data.returns import daily_returns

        result = daily_returns(sample_prices)
        first_row = result.iloc[0]
        assert not first_row.isna().any(), (
            "La primera fila no debe contener NaN — debería haberse eliminado"
        )

    def test_daily_returns_columns_preserved(self, sample_prices):
        """Los tickers (columnas) se mantienen iguales en el resultado."""
        from portfolio_optimizer.data.returns import daily_returns

        result = daily_returns(sample_prices)
        assert list(result.columns) == list(sample_prices.columns)

    # --- Tests de filter_by_coverage ---

    def test_filter_removes_sparse_ticker(self, sample_prices_with_gaps):
        """
        Un ticker con solo el 70% de datos se elimina con min_ratio=0.80.

        Usamos el fixture sample_prices_with_gaps que tiene TEF.MC con 70% de datos.
        """
        from portfolio_optimizer.data.returns import daily_returns, filter_by_coverage

        returns = daily_returns(sample_prices_with_gaps)
        result  = filter_by_coverage(returns, min_ratio=0.80)

        assert "TEF.MC" not in result.columns, (
            "TEF.MC tiene 70% de datos y debe eliminarse con min_ratio=0.80"
        )

    def test_filter_keeps_dense_ticker(self, sample_prices_with_gaps):
        """
        Tickers con >=80% de datos se mantienen.

        Los 4 tickers completos (SAN, ITX, IBE, REP) deben estar en el resultado.
        """
        from portfolio_optimizer.data.returns import daily_returns, filter_by_coverage

        returns = daily_returns(sample_prices_with_gaps)
        result  = filter_by_coverage(returns, min_ratio=0.80)

        for ticker in ["SAN.MC", "ITX.MC", "IBE.MC", "REP.MC"]:
            assert ticker in result.columns, (
                f"{ticker} tiene datos completos y no debería eliminarse"
            )

    def test_filter_uses_config_default(self, sample_prices_with_gaps):
        """
        Sin pasar min_ratio, filter_by_coverage usa MIN_OBS_RATIO de config.py (0.80).

        Este test verifica que el default funciona sin pasar el argumento.
        """
        from portfolio_optimizer.data.returns import daily_returns, filter_by_coverage

        returns = daily_returns(sample_prices_with_gaps)
        # Llamamos SIN min_ratio → debe usar config.py
        result_default  = filter_by_coverage(returns)
        result_explicit = filter_by_coverage(returns, min_ratio=0.80)

        assert list(result_default.columns) == list(result_explicit.columns), (
            "filter_by_coverage() sin argumentos debe dar el mismo resultado "
            "que filter_by_coverage(min_ratio=0.80)"
        )

    def test_filter_logs_excluded_tickers(self, sample_prices_with_gaps, caplog):
        """
        filter_by_coverage loggea los tickers eliminados con nivel WARNING.

        Por qué testear el logging:
            Si en producción un ticker clave se elimina silenciosamente,
            el usuario nunca sabría por qué su cartera tiene 29 activos
            en vez de 30. El warning en el log es la única señal.
        """
        import logging
        from portfolio_optimizer.data.returns import daily_returns, filter_by_coverage

        returns = daily_returns(sample_prices_with_gaps)

        with caplog.at_level(logging.WARNING, logger="portfolio_optimizer.data.returns"):
            filter_by_coverage(returns, min_ratio=0.80)

        # Verificamos que se loggeó algo mencionando TEF.MC
        warnings_text = " ".join(caplog.messages)
        assert "TEF.MC" in warnings_text, (
            "Se esperaba un WARNING mencionando TEF.MC en el log"
        )

    # --- Tests de get_risk_free_rate ---

    def test_get_risk_free_empty_key_raises(self):
        """
        Sin FRED_API_KEY configurada, get_risk_free_rate lanza RuntimeError
        con un mensaje claro indicando cómo obtener la clave.
        """
        import portfolio_optimizer.data.returns as returns_module
        import portfolio_optimizer.config as config_module
        from unittest.mock import patch

        # Parchamos FRED_API_KEY para que esté vacía en este test
        with patch.object(returns_module, "FRED_API_KEY", ""):
            with pytest.raises(RuntimeError, match="FRED_API_KEY"):
                returns_module.get_risk_free_rate("ES", "2020-01-01", "2020-12-31")

    def test_get_risk_free_invalid_market_raises(self):
        """
        Un mercado no reconocido lanza ValueError.
        """
        from portfolio_optimizer.data.returns import get_risk_free_rate
        from unittest.mock import patch
        import portfolio_optimizer.data.returns as returns_module

        with patch.object(returns_module, "FRED_API_KEY", "fake_key"):
            with pytest.raises(ValueError, match="no reconocido"):
                get_risk_free_rate("JP", "2020-01-01", "2020-12-31")

    @pytest.mark.network
    def test_get_risk_free_real_download(self):
        """
        Test de integración: descarga real del T-Bill USA desde FRED.
        Requiere FRED_API_KEY configurada y conexión a internet.
        """
        from portfolio_optimizer.data.returns import get_risk_free_rate
        from portfolio_optimizer.config import FRED_API_KEY

        if not FRED_API_KEY:
            pytest.skip("FRED_API_KEY no configurada — salta test de red")

        result = get_risk_free_rate("US", "2023-01-01", "2023-12-31")

        assert isinstance(result, pd.Series)
        assert len(result) > 200, "Debe devolver ~252 días hábiles"
        assert (result >= 0).all(), "La tasa libre de riesgo no puede ser negativa"
        assert result.mean() < 0.001, "La tasa diaria debe ser pequeña (< 0.1% diario)"
