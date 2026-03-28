BASELINE RESULTADOS TFG ORIGINAL
=================================
Fuente: TFG Collazo Fontes, Santiago.ipynb (codigo_original/)
Extraído: 2026-03-25
ESTOS ARCHIVOS SON DE SOLO LECTURA — referencia permanente del TFG original.

PARÁMETROS COMUNES
------------------
- Período backtest: 2011-2023 (13 años)
- Lookback: 5 años
- Optimizador BO: n_calls=400, n_initial_points=70, patience=100, tol=1e-4
- Modelos: MMV (Media-Varianza), CAPM, VE (Volatilidad Estocástica), OB (Optimización Bayesiana)

ESCENARIOS
----------
scenario_1_sin_restricciones.csv  → peso_min=0, peso_max=1.0 (sin límite por activo)
scenario_2_ucits_10pct.csv        → peso_max=0.10, concentration_max=0.40 (UCITS 5/10/40)
scenario_3_ric_25pct.csv          → peso_max=0.25, concentration_max=0.50 (RIC 25/50)

COLUMNAS
--------
Year              : Año de evaluación ex post
Model             : MMV / CAPM / VE / OB
ExAnte_Return_pct : Retorno esperado anualizado (%) calculado con datos históricos
ExAnte_Sharpe     : Sharpe ratio ex ante (con datos de entrenamiento)
ExPost_Return_pct : Retorno real obtenido en el año de evaluación (%)
ExPost_Sharpe     : Sharpe ratio calculado con retornos reales del año siguiente
Alpha_pct         : ExPost_Return - IBEX_Return ese año (%)

BENCHMARK
---------
ibex_benchmark.csv → Retornos anuales del IBEX 35 (índice de PRECIOS, sin dividendos)

ADVERTENCIAS IMPORTANTES
------------------------
1. El benchmark IBEX es precio puro, NO total return. Los dividendos (~3-4% anual)
   no están incluidos. El alfa real es menor de lo que muestran estos datos.
2. Los ExPost del Escenario 3 necesitan verificación — pueden coincidir con Escenario 2
   en algunos años (posible error en la extracción automática).
3. El lookback de 5 años difiere de las modificaciones posteriores (20 años).
   No son directamente comparables sin ajustar este parámetro.
