import pandas as pd

from config import AÑO_INICIO, AÑO_FINAL, AÑOS_ATRAS, CAMBIOS_TICKERS, PATH_EXCEL
from data   import load_excel, prepare_data
from model.mmv      import MMV
from model.capm     import CAPM
from model.ve       import VE
from model.bayesian import OB
from data.returns   import rendimientos_anuales_ibex
from evaluation     import rendimiento_año_siguiente
from utils          import extraer_sharpe


def comparativa_modelos(año_inicio, años_atras, año_final, data):
    """
    Ejecuta los 4 modelos (MMV, CAPM, VE, OB) año a año y compara sus resultados.
    Devuelve: df_ex_ante, df_ex_post, df_comparativa, df_rendimientos, pesos, df_evolucion
    """
    resultados_ex_ante = []
    resultados_ex_post = []
    resultados_dif     = []
    rendimientos_list  = []
    pesos              = {}

    indice = data["indice"]
    precios_componentes = data["precios_componentes"]

    for año in range(año_inicio, año_final):
        año_siguiente = año + 1
        try:
            resultados_MMV,  detalles_MMV,  ex_ante_MMV          = MMV(año,  años_atras, año_siguiente, data)
            resultados_CAPM, detalles_CAPM, ex_ante_CAPM         = CAPM(año, años_atras, año_siguiente, data)
            resultados_VE,   detalles_VE,   ex_ante_VE           = VE(año,   años_atras, año_siguiente, data)
            resultados_OB,   detalles_OB,   ex_ante_OB,  evol    = OB(año,   años_atras, año_siguiente, data)

            rend_IBEX        = rendimientos_anuales_ibex(indice, [año_siguiente])
            rentabilidad_IBEX = rend_IBEX.get(año_siguiente, None)

            # Alfas
            alfa_MMV  = float(resultados_MMV.loc[0,  "Diferencia"].strip("%"))
            alfa_CAPM = float(resultados_CAPM.loc[0, "Diferencia"].strip("%"))
            alfa_ve   = float(resultados_VE.loc[0,   "Diferencia"].strip("%"))
            alfa_ob   = float(resultados_OB.loc[0,   "Diferencia"].strip("%"))

            # Sharpes
            Sharpe_Ante_MMV,  Sharpe_Post_MMV  = extraer_sharpe(detalles_MMV).loc[año,  ["Sharpe Ratio Ex Ante", "Sharpe Ratio Ex Post"]]
            Sharpe_Ante_CAPM, Sharpe_Post_CAPM = extraer_sharpe(detalles_CAPM).loc[año, ["Sharpe Ratio Ex Ante", "Sharpe Ratio Ex Post"]]
            Sharpe_Ante_VE,   Sharpe_Post_VE   = extraer_sharpe(detalles_VE).loc[año,   ["Sharpe Ratio Ex Ante", "Sharpe Ratio Ex Post"]]
            Sharpe_Ante_OB,   Sharpe_Post_OB   = extraer_sharpe(detalles_OB).loc[año,   ["Sharpe Ratio Ex Ante", "Sharpe Ratio Ex Post"]]

            # Rendimientos esperados
            Rend_EA_MMV  = ex_ante_MMV[0]["Rendimiento Ex Ante"]
            Rend_EA_CAPM = ex_ante_CAPM[0]["Rendimiento Ex Ante"]
            Rend_EA_VE   = ex_ante_VE[0]["Rendimiento Ex Ante"]
            Rend_EA_OB   = ex_ante_OB[0]["Rendimiento Ex Ante"]

            # Rendimientos reales
            pesos_MMV  = detalles_MMV[año]["Pesos"]
            pesos_CAPM = detalles_CAPM[año]["Pesos"]
            pesos_VE   = detalles_VE[año]["Pesos"]
            pesos_OB   = detalles_OB[año]["Pesos"]

            Rend_MMV  = rendimiento_año_siguiente(año, pesos_MMV,  precios_componentes)
            Rend_CAPM = rendimiento_año_siguiente(año, pesos_CAPM, precios_componentes)
            Rend_VE   = rendimiento_año_siguiente(año, pesos_VE,   precios_componentes)
            Rend_OB   = rendimiento_año_siguiente(año, pesos_OB,   precios_componentes)

            resultados_ex_ante.append({
                ("MMV",  "Rendimiento Esperado"): f"{Rend_EA_MMV  * 100:.4f}%",
                ("MMV",  "Sharpe Ex Ante"):        Sharpe_Ante_MMV,
                ("CAPM", "Rendimiento Esperado"): f"{Rend_EA_CAPM * 100:.4f}%",
                ("CAPM", "Sharpe Ex Ante"):        Sharpe_Ante_CAPM,
                ("VE",   "Rendimiento Esperado"): f"{Rend_EA_VE   * 100:.4f}%",
                ("VE",   "Sharpe Ex Ante"):        Sharpe_Ante_VE,
                ("OB",   "Rendimiento Esperado"): f"{Rend_EA_OB   * 100:.4f}%",
                ("OB",   "Sharpe Ex Ante"):        Sharpe_Ante_OB,
                "Año": año_siguiente,
            })

            resultados_ex_post.append({
                ("MMV",  "Rendimiento Real"): f"{Rend_MMV  * 100:.4f}%",
                ("MMV",  "Sharpe Ex Post"):   Sharpe_Post_MMV,
                ("CAPM", "Rendimiento Real"): f"{Rend_CAPM * 100:.4f}%",
                ("CAPM", "Sharpe Ex Post"):   Sharpe_Post_CAPM,
                ("VE",   "Rendimiento Real"): f"{Rend_VE   * 100:.4f}%",
                ("VE",   "Sharpe Ex Post"):   Sharpe_Post_VE,
                ("OB",   "Rendimiento Real"): f"{Rend_OB   * 100:.4f}%",
                ("OB",   "Sharpe Ex Post"):   Sharpe_Post_OB,
                "Año": año_siguiente,
            })

            resultados_dif.append({
                ("MMV",  "Alfa"):     alfa_MMV,
                ("MMV",  "Δ Sharpe"): round(Sharpe_Post_MMV  - Sharpe_Ante_MMV,  6),
                ("CAPM", "Alfa"):     alfa_CAPM,
                ("CAPM", "Δ Sharpe"): round(Sharpe_Post_CAPM - Sharpe_Ante_CAPM, 6),
                ("VE",   "Alfa"):     alfa_ve,
                ("VE",   "Δ Sharpe"): round(Sharpe_Post_VE   - Sharpe_Ante_VE,   6),
                ("OB",   "Alfa"):     alfa_ob,
                ("OB",   "Δ Sharpe"): round(Sharpe_Post_OB   - Sharpe_Ante_OB,   6),
                "Año": año_siguiente,
            })

            rendimientos_list.append({
                "Año":                    año_siguiente,
                "Media-Varianza":         Rend_MMV,
                "CAPM":                   Rend_CAPM,
                "Volatilidad Estocástica": Rend_VE,
                "Optimización Bayesiana": Rend_OB,
                "IBEX 35": f"{rentabilidad_IBEX * 100:.4f}%" if rentabilidad_IBEX is not None else None,
            })

            pesos[año_siguiente] = {
                "MMV":  detalles_MMV,
                "CAPM": detalles_CAPM,
                "VE":   detalles_VE,
                "OB":   detalles_OB,
            }

        except Exception as e:
            print(f"Error en el año {año}: {e}")

    # DataFrames de resultados
    df_ex_ante    = pd.DataFrame(resultados_ex_ante).set_index("Año")
    df_ex_post    = pd.DataFrame(resultados_ex_post).set_index("Año")
    df_comparativa = pd.DataFrame(resultados_dif).set_index("Año")

    for df in [df_ex_ante, df_ex_post, df_comparativa]:
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df.index.name = "Año"

    # Rendimientos anuales formateados
    df_rendimientos = (
        pd.DataFrame(rendimientos_list)
        .set_index("Año")
        .apply(lambda col: col.map(lambda x: f"{x * 100:.4f}%" if isinstance(x, (int, float)) else x))
    )

    # Evolución del capital (base 100)
    df_clean     = pd.DataFrame(rendimientos_list).set_index("Año")
    df_evolucion = df_clean.apply(
        lambda col: col.map(lambda x: float(x) if isinstance(x, (int, float)) else float(x.strip("%")) / 100)
    )
    primer_año   = df_evolucion.index.min()
    df_inicio    = pd.DataFrame([[0.0] * df_evolucion.shape[1]], index=[primer_año - 1], columns=df_evolucion.columns)
    df_evolucion = pd.concat([df_inicio, df_evolucion]).sort_index()
    df_evolucion = (1 + df_evolucion).cumprod() * 100
    df_evolucion = df_evolucion.round(2)

    # Promedio de alfas
    promedio = df_comparativa.mean(numeric_only=True)
    promedio.name = "Promedio"
    df_comparativa = pd.concat([df_comparativa, promedio.to_frame().T])

    for modelo in ["MMV", "CAPM", "VE", "OB"]:
        df_comparativa[(modelo, "Alfa")] = df_comparativa[(modelo, "Alfa")].map(lambda x: f"{x:.2f}%")

    return df_ex_ante, df_ex_post, df_comparativa, df_rendimientos, pesos, df_evolucion


def main():
    """Carga los datos y ejecuta la comparativa de los 3 escenarios del TFG."""
    if not PATH_EXCEL.exists():
        print(f"Error: no se encuentra el archivo de datos: {PATH_EXCEL}")
        return

    raw  = load_excel(PATH_EXCEL)
    data = prepare_data(raw, CAMBIOS_TICKERS)

    print("Ejecutando Escenario 1: Sin restricciones...")
    Ex_ante_1, Ex_post_1, Diferencias_1, Rendimientos_1, Pesos_1, Evolucion_1 = comparativa_modelos(
        AÑO_INICIO, AÑOS_ATRAS, AÑO_FINAL, data
    )
    print(Ex_ante_1)
    print(Ex_post_1)


if __name__ == "__main__":
    main()
