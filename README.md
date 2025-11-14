# QFT Phase Solver Toolkit

Conjunto de herramientas en Python para caracterizar transformadas cuánticas de Fourier (QFT) a partir de las ecuaciones 27 y 28 descritas en el manuscrito `fourier_bosonica.pdf`. El repositorio incluye solvers numéricos, verificadores y utilidades para construir matrices QFT a partir de los parámetros hallados.

## Estructura del repositorio

- `qft_equation_27.py`: buscador de parámetros `λ_k` que satisfacen la ecuación 27 (condición de matriz de Hadamard compleja) usando optimización local/global, con soporte multi-run y verificación automática.
- `qft_equation_28.py`: solver para la ecuación 28 (condición específica QFT). Puede reutilizar soluciones de la ecuación 27 como semillas, admite Monte Carlo y métodos de optimización global.
- `verify_equation.py`: utilitario CLI para validar archivos JSON generados por los solvers y comprobar residuos frente a las ecuaciones 27 o 28.
- `qft_matrix_generator.py`: genera las matrices \(U^{(N)}\), \(u^{(N)}_{m,n}\), \(\Phi^{\text{in}}\), \(\Phi^{\text{out}}\) y la matriz QFT final \(D^{(N)}\) a partir de soluciones verificadas; incluye pruebas de unitariedad y coincidencia con la QFT estándar.
- `fourier_bosonica.pdf`: referencia teórica del proyecto.

## Requisitos

- Python 3.9 o superior.
- Dependencias: `numpy`, `scipy`, `matplotlib`. Instálalas con:

  ```bash
  pip install numpy scipy matplotlib
  ```

Algunos scripts usan `matplotlib` solo para depuración/visualización; si no necesitas gráficos puedes omitirla, pero mantenerla instalada evita errores de importación.

## Flujo de trabajo recomendado

1. **Resolver la ecuación 27**: ejecuta `qft_equation_27.py` para un tamaño N dado y guarda las soluciones verificadas en JSON.
2. **Refinar con la ecuación 28**: usa `qft_equation_28.py`, opcionalmente cargando el JSON anterior mediante `--eq27-file` para mejores puntos de partida.
3. **Verificar resultados**: ejecuta `verify_equation.py` sobre los archivos JSON producidos; permite revisar corridas individuales en modo multi-run.
4. **Construir matrices QFT**: con `qft_matrix_generator.py` transforma los parámetros `λ_k` en las matrices físicas y valida unitariedad/diferencias con la QFT estándar.

## Uso de los scripts

### `qft_equation_27.py`

```bash
python qft_equation_27.py N \
  [--method {bfgs,lbfgs,nelder-mead,differential-evolution,all}] \
  [--tolerance 1e-12] [--max-iter 2000] [--seed 123] \
  [--multi-run 20] [--output solutions/eq27_nN.json]
```

- `N`: dimensión de la QFT.
- `--method`: selecciona algoritmo (o `all` para ejecutar varios consecutivamente).
- `--multi-run`: disponible solo con `differential-evolution`; repite corridas con distintas semillas y guarda únicamente soluciones verificadas.
- `--output`: exporta resultados y metadatos en JSON; incluye soluciones estándar QFT como referencia.

El solver impone por defecto la simetría \(λ_k = λ_{N-k}\) y calcula gradientes analíticos para métodos basados en derivadas. Cada resultado almacena residuos, iteraciones y tipo de inicialización usada.

### `qft_equation_28.py`

```bash
python qft_equation_28.py N \
  [--method {nelder-mead,differential-evolution,basin-hopping,multi-start}] \
  [--eq27-file solutions/eq27_nN.json] \
  [--random-starts 5] [--monte-carlo 0.05] [--no-symmetry] \
  [--tolerance 1e-8] [--max-iter 1000] [--output solutions/eq28_nN.json]
```

- Habilita muestreo Monte Carlo cuando \(N^4\) términos vuelven prohibitivo el cálculo completo.
- `--eq27-file` carga múltiples soluciones válidas como semillas; el script las valida y documenta cuántas se usan.
- `--no-symmetry` desactiva la restricción \(λ_k = λ_{N-k}\).
- Modo `multi-start` combina arranques aleatorios con semillas de Eq.27 y reporta el origen de la mejor solución.
- La verificación comprueba residuos sobre un subconjunto de pares \((m,n)\) configurable con `--tolerance` y `sample_ratio`.

### `verify_equation.py`

```bash
python verify_equation.py resultados.json \
  --equation {27,28} [--tolerance 1e-12] \
  [--verify-all-runs] [--brief]
```

- Carga el JSON, imprime residuos por ecuación y resume la validez.
- `--verify-all-runs` revalida cada corrida guardada en multi-run, comparando con los datos almacenados.
- Reporta también la referencia de la QFT estándar si el archivo la incluye.

### `qft_matrix_generator.py`

```bash
python qft_matrix_generator.py soluciones.json \
  --output matrices.json [--equation {27,28}] [--tolerance 1e-12] [--verbose]
```

- Auto-detecta el formato (equación 27 o 28) y extrae `λ_k`. 
- Calcula todas las matrices intermedias, la QFT resultante y la QFT estándar.
- Incluye verificaciones de unitariedad (`max_deviation`) y diferencia respecto a la QFT canónica (`max_difference`, `relative_error`).
- La salida JSON contiene matrices complejas en forma de listas anidadas y metadatos del archivo origen.

## Formatos de salida

Los JSON generados por los solvers siguen estas claves comunes:

- `N`: dimensión resolvida.
- `tolerance`: tolerancia de verificación usada.
- `results`: diccionario por método con:
  - `lambda_full`: vector completo de parámetros.
  - `objective_value`, `success`, `iterations`, `solve_time`.
  - `verification`: residuos máximos/promedio y metadatos de comprobación.
  - En modo multi-run (`differential-evolution` en Eq.27): `is_multi_run`, `all_runs` (lista con semillas, duplicados y errores), `unique_solutions`, etc.
- `standard_qft`: parámetros y verificación para la QFT canónica (típicamente todos los `λ_k = 0`).

`qft_equation_28.py` añade datos específicos (`equation`, `use_symmetry`, `monte_carlo_ratio`, `eq27_solutions_used`, origen de la mejor solución, etc.).

## Consejos prácticos

- Ajusta `--tolerance` en consonancia con la precisión deseada; tolerancias demasiado estrictas pueden descartar soluciones numéricamente correctas.
- Usa `--multi-run` en Eq.27 para explorar múltiples mínimos y aumentar la probabilidad de hallar soluciones útiles para Eq.28.
- Para dimensiones grandes, prioriza métodos globales (`differential-evolution`, `basin-hopping`) y reduce `--monte-carlo` si el cálculo se vuelve lento.
- Verifica siempre las soluciones antes de generar matrices; `verify_equation.py` puede detectar degradaciones numéricas al volver a cargar archivos antiguos.
- `qft_matrix_generator.py` espera `λ_k` finitos y razonables; si se detectan desviaciones grandes el script fallará con un mensaje explicativo.

## Referencias

- `fourier_bosonica.pdf`: documento de base que introduce las ecuaciones 27 y 28, las definiciones de las matrices intermedias y la interpretación física de los parámetros `λ_k`. Revisa este texto para entender el contexto matemático completo de las herramientas.
