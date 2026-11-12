# Producto analítico para la aprobación de créditos vehiculares

Reto *Inteligencia Empresarial en la Era de los Datos* · Universidad de los Andes · Noviembre de 2026
Equipo: Romel, Gianella, Marcela e Itzabel

> **En una frase:** de cada 100 créditos, casi 22 no pagan la primera cuota. Construimos un modelo que estima,
> antes de desembolsar, la probabilidad de que un cliente pague, y un tablero en Dash para que las gerencias de
> Riesgo y Ventas decidan a quién aprobar. Aprobando el 88 % de las solicitudes, el impago de los aprobados baja de
> 23,5 % a 21,4 %, y con el mismo límite de impago se aprueban 2,6 veces más clientes que con la regla actual del buró.

- **Tablero (en la nube):** [URL de Render]
- **Reporte final (2 páginas):** [`reporte/Reporte_creditos_vehiculares.pdf`](reporte/Reporte_creditos_vehiculares.pdf)
- **Enunciado del reto:** [`reto_uandina_20261109.pdf`](reto_uandina_20261109.pdf)

---

## ¿Qué le toca a cada quién?

| Integrante | Rol del reto | Carpetas que le corresponden | Lo que presenta |
|---|---|---|---|
| **Romel** | 2 · Ingeniería de datos (apoya 3, 4 y 5) | `limpieza_datos/`, despliegue (`requirements.txt`, `render.yaml`) | Cómo se pasó de 41 columnas a 15 variables limpias |
| **Gianella** | 3 · Análisis de datos (Riesgo) | `eda/` (partes 2, 3 y 5) | Quiénes no pagan y por qué la regla del buró no alcanza |
| **Marcela** | 4 · Ciencia de datos | **`modelado/`** | Qué modelos se probaron, cuál ganó y qué tan confiable es |
| **Itzabel** | 3 · Análisis de datos (Ventas) y 5 · Tablero | `eda/` (sucursales), **`tablero/`** | Sucursales con buenos pagadores y demo del tablero |
| **Todos** | 1 · Análisis de negocio | `reporte/`, documento de la Fase 1 | Problema, preguntas y evaluación final |

---

## Cómo se conecta todo

```
train.csv ──► limpieza_datos.py ──► train_limpio.csv ──► modelado.py ──► modelos .pkl + predicciones ──► tablero/app.py ──► Render
 test.csv          (Fase 2)               │                 (Fase 3)                                         (Tarea 5)
                                          └──► eda.ipynb (gráficos y hallazgos, Tarea 2)
```

Cada etapa usa lo que deja la anterior. La función de preparación (`limpieza_datos/preparacion.py`) es la misma
en la limpieza, el modelo y el tablero, para que un cliente nuevo se procese igual que los datos de entrenamiento.

---

## Qué hay en cada carpeta

### Raíz del proyecto

| Archivo | Para qué sirve |
|---|---|
| `train.csv` | Datos originales: 233 154 créditos de agosto a octubre de 2018, con la respuesta (`LOAN_DEFAULT`: 1 = no pagó). **No se modifica.** |
| `test.csv` | 112 392 créditos de noviembre, **sin** la respuesta. Solo sirve para generar predicciones. |
| `diccionario_datos.xlsx` | Qué significa cada columna. |
| `reto_uandina_20261109.pdf` | Enunciado del reto. |
| `REFERENCIA/` | Material del curso (talleres de EDA, transformaciones, visualización y Dash). Solo consulta. |
| `requirements.txt` | Librerías y versiones que necesita el proyecto (las instala Render). |
| `.python-version` · `render.yaml` | Versión de Python y configuración del despliegue en Render. |
| `.gitignore` | Archivos que git no sube (cachés de Python). |

### `limpieza_datos/` · Fase 2 · Soporte 2 · Romel

| Archivo | Para qué sirve |
|---|---|
| `preparacion.py` | **Caja de herramientas** que usan todos: calcula la edad a la fecha de solicitud, convierte "1yrs 11mon" a meses, crea la categoría del buró (`CNS_GRUPO`), pone "Sin dato" en los vacíos y agrupa sucursales raras o nuevas en "OTRA". |
| `limpieza_datos.py` | Script que aplica esas herramientas paso a paso, verifica cada paso y guarda la versión limpia. Se corre con `python limpieza_datos.py --datos ..` |
| `datos_v1/train_limpio.csv` · `test_limpio.csv` | Datos limpios con las **15 variables** del modelo. |
| `datos_v1/decisiones_variables.csv` | Por qué se quitó o se conservó cada una de las 41 columnas originales. |
| `datos_v1/reporte_calidad.csv` | Problemas encontrados (saldos negativos, valores extremos) y qué se decidió. |
| `datos_v1/bitacora.csv` | Filas y columnas después de cada paso. |
| `datos_v1/comparacion_conjuntos.csv` | Compara entrenamiento, validación y prueba (que se parezcan). |
| `datos_v1/prep_*.pkl` | Transformaciones ajustadas solo con agosto–septiembre (versión de la Fase 2). |

### `eda/` · Tarea 2 · Soporte 3 · Gianella e Itzabel

| Archivo | Para qué sirve |
|---|---|
| `eda.ipynb` | **Notebook del análisis exploratorio**, contado como historia en 5 partes: cómo son los créditos, quiénes no pagan, si sirve la regla del buró y dónde está el riesgo, si basta una variable y si el riesgo cambia por mes. Cada gráfico tiene un título con el hallazgo y una explicación. Termina con los 3 hallazgos. |
| `graficos/01` a `09` | Los gráficos en imagen (histogramas, caja, violín, barras, dispersión, correlaciones, pairplot, impago por mes) para el reporte y las diapositivas. |
| `graficos/08_sucursales_interactivo.html` | Gráfico interactivo de sucursales (se abre con doble clic en el navegador). |

### `modelado/` · Tarea 4 · Soporte 4 · Marcela

| Archivo | Para qué sirve |
|---|---|
| **`modelado.ipynb`** | **Empieza por aquí.** El entrenamiento paso a paso, ya ejecutado, con gráficos y una explicación en lenguaje de negocio después de cada sección: referencia del buró, elección de variables, 29 configuraciones probadas, prueba en octubre, robustez (bootstrap y sin sucursal/edad), explicabilidad y equidad. Termina con las conclusiones. |
| `modelado.py` | El mismo proceso como script. **Es el que genera los archivos que usa el tablero.** Se corre con `python modelado.py` (≈ 6 minutos). |
| `ingenieria.py` | Piezas que usa el modelo: variables nuevas que se probaron, agrupador de categorías y transformación WoE. |
| `salida_modelado.log` | Registro completo de la última corrida de `modelado.py` (cada modelo entrenado y su resultado). |
| `salidas/modelo_ventas_boosting.pkl` | **Modelo final de Ventas** (gradient boosting): el que decide aprobar, revisar o rechazar. |
| `salidas/modelo_riesgo_logistica.pkl` | **Modelo final de Riesgo** (regresión logística): el que explica por qué. |
| `salidas/config_modelos.json` | Variables de entrada, cortes de aprobar/revisar/rechazar y métricas. Lo lee el tablero. |
| `salidas/predicciones_octubre.csv` | Probabilidad de cada crédito de octubre según cada modelo (con la respuesta real). Alimenta las pestañas Riesgo y Ventas. |
| `salidas/predicciones_noviembre.csv` | Probabilidad y decisión para cada crédito de `test.csv`. |
| `salidas/graficos/` | Curva ROC, curva de aprobación vs. impago (**la más importante**), calibración, importancia de variables y coeficientes. |
| `salidas/tablas/` | Todas las cifras en CSV. Las más útiles: `comparacion_modelos_octubre.csv` (métricas de cada modelo), `robustez_bootstrap.csv` (88 % vs 90 %), `sensibilidad_variables.csv` (sin sucursal / sin edad), `coeficientes_logistica.csv` y `desempeno_por_segmento.csv` (equidad). |

### `tablero/` · Tarea 5 · Soporte 5 · Itzabel

| Archivo | Para qué sirve |
|---|---|
| `app.py` | **El tablero en Dash.** Pestañas: **Riesgo** (control del % de aprobación y si se cumple la meta), **Ventas** (sucursales, fabricantes y descarga de noviembre), **Evaluar cliente** (formulario que devuelve la probabilidad de pago y la decisión) y **Resumen ejecutivo**. Local: `python app.py` → http://127.0.0.1:8050 |
| `prueba_tablero.ipynb` | Registro de las pruebas del tablero (cada pestaña probada sin servidor). |

### `reporte/` · Resultado 1 · Todos

| Archivo | Para qué sirve |
|---|---|
| `Reporte_creditos_vehiculares.pdf` · `.docx` | **Reporte final de 2 páginas** (el Word es para editar). |
| `figura1.png` · `figura_reporte.py` | La figura del reporte y el script que la genera. |

---

## Guía rápida para Marcela (modelado)

1. Abre **`modelado/modelado.ipynb`** y léelo de arriba abajo: no hace falta correrlo, ya tiene todos los resultados.
2. Las cifras que tienes que manejar para la presentación:

   | Qué | Valor |
   |---|---|
   | Modelos comparados | Regresión logística (L1, L2, WoE), gradient boosting, Random Forest y la regla del buró como referencia |
   | Cómo se validó | Entrena con agosto–septiembre y se prueba **una sola vez** en octubre, un mes que el modelo no vio |
   | Mejor modelo | Gradient boosting: AUC 0,648, KS 0,218 (la regla del buró: AUC 0,554) |
   | Resultado clave | Aprobando el 88 %, el impago baja de 23,5 % a 21,4 % (se cumple la meta en 98 de cada 100 repeticiones) |
   | Para Ventas | Con el mismo límite de impago: 88 663 créditos aprobables frente a 34 467 de la regla (+54 196) |
   | Por qué el AUC no es más alto | La mitad de los clientes no tiene historial y no hay datos de ingreso; 4 algoritmos distintos llegan al mismo techo |
   | Debilidad | Ordena bien a los clientes, pero en octubre se quedó corto en el nivel de riesgo (estimó 19,8 %, fue 23,5 %) |
   | Dato de equidad | Sin la edad el modelo casi no pierde (AUC 0,646): se puede retirar |

3. El gráfico para tu parte de la presentación: `modelado/salidas/graficos/curva_aprobacion_impago.png`.

---

## Cómo volver a correr todo

```bash
pip install -r requirements.txt jupyter seaborn matplotlib      # una sola vez
cd limpieza_datos && python limpieza_datos.py --datos ..         # 1. limpieza (≈ 1 min)
# 2. abrir eda/eda.ipynb y ejecutar todo                         #    EDA (≈ 1 min)
cd ../modelado && python modelado.py                             # 3. modelado (≈ 6 min)
cd ../tablero && python app.py                                   # 4. tablero → http://127.0.0.1:8050
```

Los resultados son reproducibles: se usa la semilla 42 en todo el proyecto.
