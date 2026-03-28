import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from config import LIM_INF


# ---------------------------------------------------------------------------
# Transformación de resultados
# ---------------------------------------------------------------------------

def transformar_pesos(resultados):
    """Convierte los resultados de pesos de las optimizaciones en una tabla por año."""
    lista_df = []
    for año, datos in resultados.items():
        df_pesos    = datos["Pesos"].copy()
        df_filtrado = df_pesos[df_pesos["Pesos"] > LIM_INF].copy()
        df_filtrado = df_filtrado.set_index("Ticker")[["Pesos"]]
        df_filtrado.columns = [año]
        lista_df.append(df_filtrado)

    df_final = pd.concat(lista_df, axis=1)
    df_final = df_final.map(lambda x: f"{x * 100:.2f}%" if x > 0 else "-")
    return df_final


def extraer_sharpe(resultados):
    """Extrae los Ratios de Sharpe Ex Ante y Ex Post de los resultados de optimización."""
    df_sharpe = pd.DataFrame.from_dict(
        {
            año: {
                "Sharpe Ratio Ex Ante": datos.get("Sharpe Ratio Ex Ante", np.nan),
                "Sharpe Ratio Ex Post": datos.get("Sharpe Ratio Ex Post", np.nan),
            }
            for año, datos in resultados.items()
        },
        orient="index",
    )
    df_sharpe.sort_index(inplace=True)
    return df_sharpe


def detalle_pesos(datos_resultados, metodo_seleccionado):
    """Genera una tabla de pesos por año para el modelo seleccionado."""
    lista_años      = []
    pesos_por_ticker = {}

    for año_sgte, valores in sorted(datos_resultados.items()):
        if metodo_seleccionado not in valores:
            continue
        for año_base, info_metodo in sorted(valores[metodo_seleccionado].items()):
            año_mostrado = año_base + 1
            lista_años.append(año_mostrado)
            for _, fila in info_metodo["Pesos"].iterrows():
                peso_valor = fila["Pesos"]
                if peso_valor > LIM_INF:
                    ticker = fila["Ticker"]
                    pesos_por_ticker.setdefault(ticker, {})[año_mostrado] = f"{peso_valor * 100:.2f}%"

    if not lista_años:
        print(f"No hay datos para el método {metodo_seleccionado}.")
        return ""

    lista_años  = sorted(set(lista_años))
    encabezados = ["Ticker"] + lista_años
    filas_tabla = [
        [ticker] + [pesos_por_ticker[ticker].get(año, "-") for año in lista_años]
        for ticker in sorted(pesos_por_ticker)
    ]

    tabla_resultado = tabulate(filas_tabla, headers=encabezados, tablefmt="rounded_outline")
    print(tabla_resultado)
    return tabla_resultado


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def visualizar(df_input):
    """Representa gráficamente los rendimientos anuales de la cartera vs IBEX 35."""
    df = df_input.copy()
    for col in df.columns:
        if col != "Año":
            df[col] = df[col].replace({"N/A": None}).str.rstrip("%").astype(float) / 100

    plt.figure(figsize=(12, 7))
    plt.plot(df["Año"], df["Rendimiento de la Cartera"], label="Rendimiento de la Cartera", linewidth=2)
    plt.plot(df["Año"], df["Rendimiento del IBEX 35"],   label="Rendimiento del IBEX 35",
             color="black", linestyle="--", linewidth=3)

    plt.title("Comparación de Rendimientos Anuales: Portfolio vs IBEX 35", fontsize=16)
    plt.xlabel("Año", fontsize=14)
    plt.ylabel("Rendimiento (%)", fontsize=14)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xticks(df["Año"], fontsize=12)
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def visualizar_escenarios(df_rendimientos, df_evolucion):
    """Dibuja rendimientos anuales y evolución acumulada del capital."""
    df_ret = df_rendimientos.copy()
    for col in df_ret.columns:
        df_ret[col] = df_ret[col].replace({"N/A": None}).str.rstrip("%").astype(float) / 100

    plt.figure(figsize=(14, 6))
    for col in df_ret.columns:
        if col == "IBEX 35":
            plt.plot(df_ret.index, df_ret[col], label=col, color="black", linestyle="--", linewidth=3)
        else:
            plt.plot(df_ret.index, df_ret[col], label=col, linewidth=2, alpha=0.8)
    plt.axhline(0, linestyle="--", linewidth=1.5, alpha=0.7)
    plt.title("Rendimientos Anuales vs IBEX 35", fontsize=16)
    plt.ylabel("Rendimiento (%)", fontsize=14)
    plt.xticks(df_ret.index, fontsize=12)
    plt.legend(fontsize=12, loc="upper right", frameon=True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    for col in df_evolucion.columns:
        if col == "IBEX 35":
            plt.plot(df_evolucion.index, df_evolucion[col], label=col, color="black", linestyle="--", linewidth=3)
        else:
            plt.plot(df_evolucion.index, df_evolucion[col], label=col, linewidth=2, alpha=0.8)
    plt.title("Evolución del Valor de la Cartera", fontsize=16)
    plt.ylabel("Valor de la Cartera (€)", fontsize=14)
    plt.xticks(df_evolucion.index, fontsize=12)
    plt.legend(fontsize=12, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.show()


def visualizar_evolucion_OB(evol_bayesiana):
    """Dibuja la evolución de la función objetivo en cada iteración de la OB."""
    iters       = np.arange(1, len(evol_bayesiana) + 1)
    obj_vals    = np.array(evol_bayesiana)
    best_so_far = np.minimum.accumulate(obj_vals)

    plt.figure()
    plt.plot(iters, obj_vals,    label="valor observado")
    plt.plot(iters, best_so_far, label="mejor hasta ahora", linewidth=2)
    plt.xlabel("iteración")
    plt.ylabel("valor objetivo (–Sharpe)")
    plt.title("Evolución de Optimización Bayesiana")
    plt.legend()
    plt.tight_layout()
    plt.show()
