"""
Punto de entrada: carga datos, ejecuta backtest (cvxpy o bayesian) y muestra resultados.
Ejecutar desde la carpeta Documents con: python -m IBEX.run
Para comparar optimizador antiguo vs nuevo: python -m IBEX.run --compare
"""
from __future__ import annotations

import argparse
import warnings
from typing import Optional

import pandas as pd

from .config import (
    PATH_EXCEL,
    AÑOS_ATRAS,
    CAMBIOS_TICKERS,
    MIN_OBS_RATIO,
)
from .data import load_excel, prepare_data, filtrar_risk_free, rendimientos_diarios_filtrados, rendimientos_anuales_ibex
from .model import modelo_OB, modelo_OB_bayesian, modelo_OB_cvxpy  # modelo_OB_cvxpy puede ser None si no hay cvxpy
from .evaluation import rendimiento_año_siguiente, sharpe_año_siguiente


def OB(
    año_inicio: int,
    años_atras: int,
    año_final: int,
    data: dict,
    optimizer: Optional[str] = None,
) -> tuple:
    """
    Backtest de la cartera optimizada por Optimización Bayesiana año a año.
    Compara rendimiento y Sharpe ex post con el IBEX 35.

    Parameters
    ----------
    año_inicio, años_atras, año_final : int
    data : dict
        Debe contener: componentes_actualizados, precios_componentes,
        rendimientos_componentes, risk_free_alineado, indice

    Returns
    -------
    resultados_df : pd.DataFrame
    ob : dict por año con Pesos, Sharpe Ex Ante, Sharpe Ex Post
    rendimiento_cartera_ex_ante : list
    evolucion_optimizacion : dict año -> lista de valores de la función objetivo
    """
    ob = {}
    resultados = []
    rendimiento_cartera_ex_ante = []
    evolucion_optimizacion = {}

    componentes_actualizados = data["componentes_actualizados"]
    precios_componentes = data["precios_componentes"]
    rendimientos_componentes = data["rendimientos_componentes"]
    risk_free_alineado = data["risk_free_alineado"]
    indice = data["indice"]

    # Una sola llamada vectorizada para rendimientos anuales del IBEX (todos los años del backtest)
    años_evaluar = list(range(año_inicio + 1, año_final + 1))
    rendimientos_ibex_precomp = rendimientos_anuales_ibex(indice, años_evaluar)

    # Elegir optimizador: el pasado por argumento o el de config
    if optimizer == "cvxpy" and modelo_OB_cvxpy is not None:
        optimizador = modelo_OB_cvxpy
    elif optimizer == "bayesian":
        optimizador = modelo_OB_bayesian
    else:
        optimizador = modelo_OB  # usa config.OPTIMIZER y fallback a bayesian si cvxpy no instalado

    for año in range(año_inicio, año_final):
        try:
            rendimientos_diarios = rendimientos_diarios_filtrados(
                año,
                años_atras,
                componentes_actualizados,
                precios_componentes,
                rendimientos_componentes,
            )
            min_obs = int(MIN_OBS_RATIO * len(rendimientos_diarios))
            rendimientos_diarios_relevantes = rendimientos_diarios.dropna(axis=1, thresh=min_obs)

            rf_filtrado = filtrar_risk_free(risk_free_alineado, año, años_atras)
            risk_free_real = float(rf_filtrado["Yield"].mean())

            pesos_optimos, sharpe_ratio_opt, rendimiento_ex_ante, evolucion_bayesiana = optimizador(
                rendimientos_diarios_relevantes, risk_free_real
            )

            rendimiento_cartera = rendimiento_año_siguiente(año, pesos_optimos, precios_componentes)

            rendimiento_ibex = rendimientos_ibex_precomp.get(año + 1)
            sharpe_ex_post = sharpe_año_siguiente(
                año, pesos_optimos, precios_componentes, risk_free_alineado
            )

            # Alfa = rendimiento cartera - rendimiento benchmark (IBEX)
            alfa = (rendimiento_cartera - rendimiento_ibex) if rendimiento_ibex is not None else None

            resultados.append({
                "Año": año + 1,
                "Rendimiento real (Cartera)": rendimiento_cartera,
                "Rendimiento IBEX 35": rendimiento_ibex,
                "Alfa": alfa,
                "Sharpe Ex Ante": sharpe_ratio_opt,
                "Sharpe Ex Post": sharpe_ex_post,
            })

            ob[año] = {
                "Pesos": pesos_optimos,
                "Sharpe Ratio Ex Ante": sharpe_ratio_opt,
                "Sharpe Ratio Ex Post": sharpe_ex_post,
            }

            rendimiento_cartera_ex_ante.append({
                "Año": año + 1,
                "Rendimiento Ex Ante": rendimiento_ex_ante,
            })

            evolucion_optimizacion[año + 1] = evolucion_bayesiana

        except Exception as e:
            print(f"Error procesando el año {año}: {e}")

    resultados_df = pd.DataFrame(resultados)
    if not resultados_df.empty:
        # Formatear para salida legible: % para rendimientos y alfa, 4 decimales para Sharpe
        def fmt_pct(x):
            if pd.isna(x):
                return "-"
            try:
                return f"{float(x) * 100:.4f}%"
            except (TypeError, ValueError):
                return "-"

        def fmt_sharpe(x):
            if pd.isna(x):
                return "-"
            try:
                return f"{float(x):.4f}"
            except (TypeError, ValueError):
                return "-"

        resultados_df["Rendimiento real (Cartera)"] = resultados_df["Rendimiento real (Cartera)"].map(fmt_pct)
        resultados_df["Rendimiento IBEX 35"] = resultados_df["Rendimiento IBEX 35"].map(fmt_pct)
        resultados_df["Alfa"] = resultados_df["Alfa"].map(fmt_pct)
        resultados_df["Sharpe Ex Ante"] = resultados_df["Sharpe Ex Ante"].map(fmt_sharpe)
        resultados_df["Sharpe Ex Post"] = resultados_df["Sharpe Ex Post"].map(fmt_sharpe)
    else:
        print("No se generaron resultados válidos para los años seleccionados.")

    return resultados_df, ob, rendimiento_cartera_ex_ante, evolucion_optimizacion


def main():
    """Carga datos, ejecuta OB (por defecto cvxpy) y muestra resultados. Con --compare ejecuta ambos."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser(description="Backtest IBEX: cvxpy (rápido) o bayesian (lento).")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Ejecutar ambos optimizadores y mostrar resultados lado a lado.",
    )
    parser.add_argument(
        "--bayesian",
        action="store_true",
        help="Usar solo optimizador bayesiano (gp_minimize).",
    )
    args = parser.parse_args()

    if not PATH_EXCEL.exists():
        print(f"Error: No se encuentra el archivo de datos: {PATH_EXCEL}")
        print("Copie 'Evolucion IBEX.xlsx' en la carpeta inputs/ del proyecto.")
        return
    raw = load_excel(PATH_EXCEL)
    data = prepare_data(raw, CAMBIOS_TICKERS)
    año_inicio = 2005
    año_final = 2023

    if args.compare:
        if modelo_OB_cvxpy is None:
            print("AVISO: cvxpy no está instalado. Instale con: pip install cvxpy")
            print("Se ejecutará solo el optimizador bayesian dos veces para comparar.\n")
        print("Ejecutando optimizador cvxpy (rápido)...")
        res_cvxpy, ob_cvxpy, ex_ante_cvxpy, evol_cvxpy = OB(
            año_inicio, AÑOS_ATRAS, año_final, data, optimizer="cvxpy"
        )
        print("Ejecutando optimizador bayesian (puede tardar varios minutos)...")
        res_bayesian, ob_bayesian, ex_ante_bayesian, evol_bayesian = OB(
            año_inicio, AÑOS_ATRAS, año_final, data, optimizer="bayesian"
        )
        print("\n" + "=" * 60)
        print("Resultados CVXPY (nuevo, rápido)")
        print("=" * 60)
        print(res_cvxpy.to_string(index=False))
        print("\n" + "=" * 60)
        print("Resultados BAYESIAN (antiguo)")
        print("=" * 60)
        print(res_bayesian.to_string(index=False))
        print("\nPesos último año (cvxpy), peso > 1%:")
        ultimo = ob_cvxpy.get(año_final - 1, {})
        if ultimo:
            pesos = ultimo["Pesos"]
            print(pesos[pesos["Pesos"] > 0.01].to_string(index=False))
        return

    optimizer = "bayesian" if args.bayesian else "cvxpy"
    resultados_df, ob, ex_ante, evol = OB(año_inicio, AÑOS_ATRAS, año_final, data, optimizer=optimizer)

    titulo = "Optimización Bayesiana" if optimizer == "bayesian" else "Optimización cvxpy (QP)"
    print(f"Resultados {titulo} vs IBEX 35")
    print("(Rendimiento real, Alfa, Sharpe ex ante, Sharpe ex post)")
    print("=" * 80)
    print(resultados_df.to_string(index=False))
    print("\nPesos óptimos (último año, peso > 1%):")
    ultimo = ob.get(año_final - 1, {})
    if ultimo:
        pesos = ultimo["Pesos"]
        pesos_visible = pesos[pesos["Pesos"] > 0.01]
        print(pesos_visible.to_string(index=False))


if __name__ == "__main__":
    main()
