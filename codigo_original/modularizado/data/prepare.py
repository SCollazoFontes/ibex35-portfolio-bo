import pandas as pd
from .tickers import actualizar_tickers


def prepare_data(raw: dict, cambios_tickers: dict) -> dict:
    """
    Procesa los datos en bruto devueltos por load_excel() y devuelve
    un dict con todos los DataFrames listos para usar en el backtest:

        - componentes_actualizados : composición histórica con tickers unificados
        - indice                   : precios del índice indexados por fecha
        - precios_componentes      : precios de componentes indexados por fecha
        - risk_free_alineado       : tipo libre de riesgo alineado con el índice
        - rendimientos_ibex        : rendimientos diarios del índice
        - rendimientos_componentes : rendimientos diarios de los componentes
        - correlaciones_historicas : matriz de correlaciones históricas (para VE)
    """

    # ------------------------------------------------------------------
    # Componentes: cabeceras con fechas en formato YYYY-MM-DD
    # ------------------------------------------------------------------
    componentes = raw["componentes"].copy()
    componentes.columns = (
        ["Ticker"]
        + pd.to_datetime(componentes.iloc[0, 1:], dayfirst=True)
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )
    componentes = componentes.iloc[1:]

    # Unificar tickers que han cambiado de nombre
    componentes_actualizados = actualizar_tickers(componentes, cambios_tickers)

    # ------------------------------------------------------------------
    # Índice: indexar por fecha
    # ------------------------------------------------------------------
    indice = raw["indice"].copy()
    indice.set_index("Date", inplace=True)
    indice.index = pd.to_datetime(indice.index)

    # ------------------------------------------------------------------
    # Precios de componentes: indexar por fecha
    # ------------------------------------------------------------------
    precios_componentes = raw["precios"].copy()
    precios_componentes.set_index(precios_componentes.columns[0], inplace=True)
    precios_componentes.index = pd.to_datetime(precios_componentes.index)

    # ------------------------------------------------------------------
    # Risk free: alinear con el calendario del índice
    # ------------------------------------------------------------------
    datos_risk_free = raw["risk_free"].copy()
    datos_risk_free.set_index("Date", inplace=True)
    datos_risk_free.index = pd.to_datetime(datos_risk_free.index)

    risk_free = datos_risk_free[["Yield"]].dropna() / 100
    risk_free.index = pd.to_datetime(risk_free.index)
    risk_free_alineado = risk_free.reindex(indice.index, method="ffill")

    # ------------------------------------------------------------------
    # Rendimientos diarios
    # ------------------------------------------------------------------
    rendimientos_ibex = pd.DataFrame(indice["Price"].pct_change())
    rendimientos_ibex.fillna(0, inplace=True)

    rendimientos_componentes = precios_componentes.pct_change(fill_method=None)

    # ------------------------------------------------------------------
    # Correlaciones históricas (usadas por el modelo VE)
    # ------------------------------------------------------------------
    correlaciones_historicas = rendimientos_componentes.corr()

    return {
        "componentes_actualizados":  componentes_actualizados,
        "indice":                    indice,
        "precios_componentes":       precios_componentes,
        "risk_free_alineado":        risk_free_alineado,
        "rendimientos_ibex":         rendimientos_ibex,
        "rendimientos_componentes":  rendimientos_componentes,
        "correlaciones_historicas":  correlaciones_historicas,
    }
