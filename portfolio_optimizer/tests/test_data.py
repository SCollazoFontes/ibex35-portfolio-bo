"""
test_data.py — Tests para data/loader.py y data/cache.py.

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
