# IBEX – Optimización de cartera (Sharpe)

Proyecto modular: backtest año a año de carteras sobre componentes del IBEX 35 maximizando el ratio de Sharpe. Incluye **optimizador rápido (cvxpy, QP convexo)** y **optimizador bayesiano (gp_minimize)** para comparar resultados.

## Contenido de la carpeta

- **inputs/** – Copia del Excel `Evolucion IBEX.xlsx` (datos del modelo).
- **codigo_original/** – Notebook del TFG.
- **config.py** – Ruta del Excel, `OPTIMIZER` (`"cvxpy"` | `"bayesian"`), `USE_LEDOIT_WOLF`, y parámetros.
- **data/** – Carga, preparación, tickers y rendimientos (incl. cálculo vectorizado de rendimientos anuales).
- **model/** – Ratio de Sharpe, estimación media/covarianza (Ledoit-Wolf opcional), optimizador cvxpy y bayesian.
- **evaluation.py** – Rendimiento y Sharpe ex post (año siguiente).
- **run.py** – Punto de entrada: `OB()` y `main()`.

## Uso

**Un solo entorno:** el proyecto usa un único entorno virtual dentro de la carpeta del proyecto (`.venv`).

Primera vez (crear entorno e instalar dependencias):

```bash
cd IBEX
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Ejecutar el backtest:** el paquete se llama `IBEX`, así que hay que lanzar desde la carpeta **padre** de `IBEX` (p. ej. `Documents`), no desde dentro de `IBEX`.

**Si estás en la carpeta IBEX** (prompt termina en `IBEX %`):

```bash
source .venv/bin/activate
cd ..
python -m IBEX.run
```

**Si estás en la carpeta que contiene IBEX** (p. ej. Documents):

```bash
source IBEX/.venv/bin/activate
python -m IBEX.run
```

Por defecto se usa el optimizador **cvxpy** (rápido). Para usar solo el bayesiano (lento):

```bash
python -m IBEX.run --bayesian
```

Para **comparar resultados** de ambos optimizadores (cvxpy vs bayesian):

```bash
python -m IBEX.run --compare
```

En otro script, con optimizador por defecto (config) o forzado:

```python
from IBEX.config import PATH_EXCEL, AÑOS_ATRAS, CAMBIOS_TICKERS
from IBEX.data import load_excel, prepare_data, filtrar_risk_free, rendimientos_diarios_filtrados, rendimientos_anuales_ibex
from IBEX.model import modelo_OB, modelo_OB_bayesian, modelo_OB_cvxpy
from IBEX.evaluation import rendimiento_año_siguiente, sharpe_año_siguiente
from IBEX.run import OB

raw = load_excel(PATH_EXCEL)
data = prepare_data(raw, CAMBIOS_TICKERS)
# Por defecto usa config.OPTIMIZER (cvxpy si está instalado)
resultados_df, ob, ex_ante, evol = OB(2005, AÑOS_ATRAS, 2023, data)
# Forzar bayesian para comparar con el código antiguo
resultados_bayesian, ob_b, ex_ante_b, evol_b = OB(2005, AÑOS_ATRAS, 2023, data, optimizer="bayesian")
```

## Configuración

En `config.py`:

- `PATH_EXCEL`: ruta del Excel (p. ej. `inputs/Evolucion IBEX.xlsx`).
- `OPTIMIZER`: `"cvxpy"` (rápido, recomendado) o `"bayesian"`.
- `USE_LEDOIT_WOLF`: `True` para estimar covarianza con Ledoit-Wolf (más estable).
- `AÑOS_ATRAS`, `CAMBIOS_TICKERS`, `PESO_*`, `OB_N_CALLS`, etc.

## Si siguen apareciendo avisos o errores

1. **Avisos (FutureWarning, UserWarning)**  
   Al ejecutar `python -m IBEX.run` ya se ignoran por defecto. Si usas el código importando desde otro script, añade al inicio:
   ```python
   import warnings
   warnings.filterwarnings("ignore", category=FutureWarning)
   warnings.filterwarnings("ignore", category=UserWarning)
   ```

2. **"The solver ECOS is not installed"**  
   No hace falta ECOS; cvxpy usa otro solver. Si aun así falla, instala uno explícito:
   ```bash
   pip install ecos
   ```
   o
   ```bash
   pip install scs osqp
   ```

3. **"No se encuentra el archivo de datos"**  
   Copia `Evolucion IBEX.xlsx` en `IBEX/inputs/` y ejecuta desde la carpeta que contiene `IBEX/` (p. ej. `Documents`):
   ```bash
   cd /ruta/a/Documents
   python -m IBEX.run
   ```

4. **Errores del linter / IDE (p. ej. tipos en pandas)**  
   Son advertencias de tipo, no errores de ejecución. Puedes ignorarlas o, si usas Pylance/Pyright, añadir en la raíz del proyecto un `pyrightconfig.json` con `"reportGeneralTypeIssues": "none"` para reducir avisos.
