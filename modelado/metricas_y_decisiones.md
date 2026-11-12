# Métricas y decisiones del modelado

**Proyecto:** aprobación de créditos vehiculares · **Rol:** ciencia de datos · **Soporte 4 del reto**

Este documento explica **con qué métricas medimos los modelos, por qué elegimos esas y no otras**, y qué decisiones de
método tomamos en el camino. Todas las cifras son de **octubre de 2018**, el mes que los modelos nunca vieron al
entrenar, y salen de `modelado.ipynb`, `salida_modelado.log` y `salidas/tablas/`.

---

## 1. Resumen en una tabla

| Pregunta | Métrica | Gradient boosting | Regla del buró | ¿Por qué esta métrica? |
|---|---|---|---|---|
| ¿Ordena bien a los clientes por riesgo? | **AUC-ROC** | **0,648** | 0,554 | No depende de un umbral ni del desbalance |
| ¿Cuánto mejor que el azar? | **Gini** (= 2·AUC − 1) | **0,296** | 0,109 | Es la métrica estándar en banca |
| ¿Separa a buenos y malos pagadores? | **KS** | **0,218** | 0,074 | Muy usada en crédito; fácil de explicar |
| ¿Encuentra a los que no pagan? | **PR-AUC** | **0,345** | 0,263 | Se enfoca en la clase minoritaria (azar = 0,235) |
| ¿Sus probabilidades son realistas? | **Brier** y **curva de calibración** | 0,1726 | 0,1796 | El tablero muestra probabilidades |
| ¿Cumple la meta de Riesgo? (CN2) | **Impago de los aprobados** | **21,40 %** al 88 % | 22,71 % aprobando 91 % | Es la meta del negocio |
| ¿Cumple la meta de Ventas? (CN1 y PE1) | **% aprobado** y **créditos aprobables con el mismo impago** | **88 663** | 34 467 | Mide "aprobar más clientes viables" |
| ¿Cuánto dinero protege? (CE1) | **Monto evitado** | **278 millones** al 88 % | — | Traduce el modelo a dinero |
| ¿El resultado es confiable? | **Intervalo de confianza (bootstrap)** | 98 % de seguridad al 88 % | — | Evita confundir suerte con resultado |
| ¿Trata igual a todos los grupos? | **AUC y % aprobado por segmento** | Ver sección 6 | — | Ética y riesgo regulatorio |

**No usamos accuracy (exactitud) como métrica principal.** La razón está en la sección 7.

---

## 2. Por qué este problema necesita métricas especiales

Tres características del problema determinan qué métricas sirven:

1. **Las clases están desbalanceadas.** En octubre pagaron 76,5 de cada 100 clientes y no pagaron 23,5. Cualquier
   métrica que premie "acertar en general" queda dominada por los que pagan.
2. **Los errores no cuestan lo mismo.** Aprobar a alguien que no paga significa perder dinero; rechazar a alguien que sí
   pagaba significa perder una venta. Una métrica que trate ambos errores igual no le sirve a ninguna de las dos gerencias.
3. **La decisión es por orden, no por un umbral fijo.** El negocio no pregunta "¿este cliente tiene más de 50 % de
   probabilidad de no pagar?" (casi nadie la tiene). Pregunta "¿quiénes son el 12 % más riesgoso?". Por eso lo más
   importante es que el modelo **ordene bien**.

Por eso el criterio **CML2** de la Fase 1 dice: *"el desempeño se mide con AUC-ROC, PR-AUC, KS y la calibración de las
probabilidades; la exactitud no es la métrica principal"*.

---

## 3. Métricas de discriminación: ¿el modelo ordena bien?

### AUC-ROC (métrica principal)
- **Qué mide:** si tomamos al azar un cliente que no pagó y uno que sí pagó, el AUC es la probabilidad de que el modelo
  le asigne más riesgo al que no pagó. Un AUC de 0,5 es tirar una moneda; 1 es perfecto.
- **Nuestro valor:** 0,648 con gradient boosting. De cada 100 parejas (un cliente que paga y uno que no), el modelo
  ordena bien a unas 65; la regla del buró, a 55.
- **Por qué la usamos:** no depende de ningún umbral ni del desbalance de clases, y mide justo lo que necesita el negocio,
  que es ordenar. Es la métrica más usada para comparar modelos de crédito (Lessmann et al., 2015).
- **Dónde se usó:** para elegir variables y configuración (en septiembre) y para comparar modelos (en octubre).

### Gini
- **Qué mide:** lo mismo que el AUC, en la escala que usan los bancos: Gini = 2 × AUC − 1 (0 es azar, 1 es perfecto).
- **Nuestro valor:** 0,296 (boosting) frente a 0,109 (buró). **El modelo tiene 2,7 veces el poder de la regla actual.**
- **Por qué la reportamos:** es el lenguaje de las áreas de riesgo. Como referencia general, un Gini cercano a 0,3 se
  considera modesto pero utilizable, sobre todo con muchos clientes sin historial.

### KS (Kolmogorov–Smirnov)
- **Qué mide:** la máxima distancia entre la distribución de puntajes de quienes pagan y de quienes no pagan. Cuanto más
  alto, mejor separa el modelo los dos grupos.
- **Nuestro valor:** 0,218 frente a 0,074 de la regla: **3 veces más separación**.
- **Por qué la usamos:** es tradicional en evaluación de crédito y se explica en una frase a la gerencia.

### PR-AUC (área bajo la curva precisión–recall)
- **Qué mide:** qué tan bien encuentra el modelo a los clientes que **no pagan** (la clase minoritaria), sin dejarse
  impresionar por los que pagan.
- **Nuestro valor:** 0,345. El azar daría 0,235, que es la proporción de impago de octubre.
- **Por qué la usamos:** con clases desbalanceadas complementa al AUC, que a veces se ve optimista.

| Modelo | AUC | Gini | KS | PR-AUC |
|---|---|---|---|---|
| Clasificador trivial | 0,500 | 0,000 | 0,000 | 0,235 |
| Regla del buró | 0,554 | 0,109 | 0,074 | 0,263 |
| Regresión logística WoE | 0,631 | 0,262 | 0,190 | 0,328 |
| Random Forest | 0,635 | 0,270 | 0,197 | 0,334 |
| Regresión logística (Ridge) | 0,639 | 0,278 | 0,204 | 0,338 |
| **Gradient boosting** | **0,648** | **0,296** | **0,218** | **0,345** |

---

## 4. Métricas de calibración: ¿las probabilidades son realistas?

### Brier score
- **Qué mide:** el error cuadrático promedio entre la probabilidad predicha y lo que pasó (0 = perfecto). Combina
  calibración y discriminación.
- **Nuestro valor:** 0,1726 (boosting) frente a 0,1796 (buró) y 0,1808 (trivial).
- **Por qué la usamos:** el tablero muestra la probabilidad de pago de cada cliente; necesitamos saber si ese número es
  creíble.

### Curva de calibración y probabilidad media
- **Qué mide:** agrupa a los clientes por probabilidad predicha y compara con el impago real de cada grupo. Si los
  puntos caen sobre la diagonal, el modelo está bien calibrado.
- **Lo que encontramos (y lo decimos tal cual):** **el modelo se quedó corto en octubre.** Esperaba que no pagaran 19,8 de
  cada 100 clientes y no pagaron 23,5; el riesgo real fue cerca de 1,2 veces el estimado. La curva es recta y paralela
  a la diagonal: **el orden es correcto, el nivel no**. Octubre fue un mes peor que agosto y septiembre.
- **Qué hicimos:** probamos calibrar (Platt / sigmoide) y no mejoró (Brier 0,1728), porque el problema es el cambio de
  nivel del mes, no la forma de la curva. Por eso **las decisiones se toman por orden** (aprobar el X % de menor riesgo)
  y el tablero advierte que la probabilidad sirve para comparar clientes, no como cifra exacta. Recomendamos recalibrar
  cada mes con datos recientes (Niculescu-Mizil y Caruana, 2005).

---

## 5. Métricas de negocio: ¿cumple las metas?

Las métricas técnicas dicen si el modelo es bueno; estas dicen **si le sirve a la empresa**. Todas se calculan en
octubre aprobando el porcentaje de solicitudes de menor riesgo.

| Métrica | Criterio | Qué mide | Gradient boosting (aprobando 88 %) |
|---|---|---|---|
| **Impago de los aprobados** | CN2 · Riesgo | De cada 100 aprobados, cuántos no pagan | **21,40 %** (meta < 21,71 %; sin modelo 23,51 %) |
| **% de solicitudes aprobadas** | CN1 · Ventas | Cuántas aprobaciones se conservan | 88 % (meta ≥ 90 %, ver sección 6) |
| **Créditos aprobables con impago < 21,71 %** | PE1 · Ventas | Cuántos clientes se pueden aprobar sin pasar la meta | **88 663** frente a **34 467** de la regla (+54 196, 2,6 veces) |
| **Impagos evitados** | CML3 | Clientes que no iban a pagar y se rechazan | 4 605 |
| **Buenos pagadores rechazados** | CML3 | El costo para Ventas | 7 199 |
| **Monto evitado** | CE1 | Dinero que no se desembolsa a quien no iba a pagar | **278 millones** (exposición bruta, no pérdida neta) |

**Precisión y recall con el corte del negocio** (rechazar el 12 % más riesgoso): de cada 100 rechazados, **39 no habrían
pagado** (sin modelo serían 23,5, es decir 1,7 veces más puntería), y el modelo atrapa a **1 de cada 5** clientes que no
pagan rechazando solo a 12 de cada 100. Las reportamos como apoyo, no como métricas para elegir modelo.

**Riesgo promedio frente a riesgo del cliente adicional.** El 21,40 % es el promedio de todos los aprobados. Los clientes
que se suman en el borde son mucho más riesgosos: entre el 88 % y el 90 % de aprobación, **no paga 1 de cada 3**
(34,7 %). Por eso cada punto de aprobación adicional cuesta cada vez más impago.

---

## 6. Métricas de confiabilidad, robustez y equidad

### Intervalo de confianza por bootstrap
- **Qué es:** repetimos la medición 1 000 veces sobre muestras distintas de octubre (con reemplazo) para ver cuánto
  cambia el resultado por azar.
- **Por qué lo usamos:** la meta es 21,71 % y el modelo dio 21,70 % al 90 %. Con una diferencia tan pequeña había que saber
  si era un resultado real o suerte.

| % aprobado | Impago de los aprobados | Intervalo 95 % | Veces que cumple la meta |
|---|---|---|---|
| 85 % | 20,93 % | 20,65 % – 21,22 % | 100 de 100 |
| **88 %** | **21,40 %** | **21,13 % – 21,69 %** | **98 de 100** |
| 90 % | 21,70 % | 21,43 % – 21,97 % | 52 de 100 (una moneda al aire) |

**Conclusión:** CN1 (aprobar al menos el 90 %) y CN2 (impago < 21,71 %) no se pueden garantizar a la vez. Recomendamos
aprobar alrededor del 88 % y que las gerencias decidan.

### Sensibilidad: ¿de qué depende el modelo?
| Variante (gradient boosting) | AUC | Impago de los aprobados al 90 % |
|---|---|---|
| Modelo completo (15 variables) | 0,648 | 21,70 % |
| Sin sucursal | 0,629 | 21,81 % (no cumple) |
| Sin edad | 0,646 | 21,65 % |

La sucursal aporta información que nada reemplaza; **la edad casi no aporta y se puede retirar** sin perder desempeño.

### Equidad por segmento
Medimos AUC, % aprobado e impago de los aprobados por edad y tipo de empleo. Los jóvenes de 18 a 25 años y los
independientes tienen más impago real y se les aprueba algo menos (88 % frente a 92 % de los mayores de 46); el modelo
distingue algo peor a los jóvenes (AUC 0,63 frente a 0,67). Es una diferencia moderada que se declara y se vigila.

### Importancia de variables y coeficientes (explicabilidad)
- **Importancia por permutación** (boosting): cuánto cae el AUC si se desordena una variable. LTV, sucursal y buró son
  las que más pesan. La usamos en lugar de la importancia interna de los árboles porque se mide en datos nuevos y no
  favorece a las variables con muchas categorías.
- **Coeficientes y *odds ratios*** (regresión logística): cuánto multiplica cada variable las probabilidades relativas
  de no pagar. Por ejemplo, la sucursal 251 las multiplica por 2,2.

---

## 7. Métricas que NO usamos (o no como principales) y por qué

| Métrica | ¿Se usó? | Por qué no |
|---|---|---|
| **Accuracy (exactitud)** | Solo como referencia | Decir que "todos pagan" ya da **76,5 %** sin detectar ningún impago. Con el umbral de 0,5 los modelos también dan 76,5 % (casi nadie supera 50 % de riesgo), y con el corte del negocio dan **73,9 %**, *menos* que no hacer nada, aunque bajan el impago de 23,5 % a 21,4 %. Trata igual dos errores que cuestan distinto. |
| **F1-score** | No | Exige un umbral fijo y mezcla precisión y recall con el mismo peso, cuando para el negocio pesan distinto. El umbral lo define el negocio (88 %), no la maximización del F1. |
| **Precisión y recall como criterio de selección** | Solo como apoyo | Dependen del umbral; para elegir modelos usamos AUC y KS, que no dependen de él. Se reportan en el corte elegido (sección 5). |
| **Matriz de confusión con umbral 0,5** | No | Con 0,5 casi no hay predicciones de impago y la matriz no dice nada útil. En el tablero se muestra con el corte del negocio. |
| **Log-loss** | Indirectamente | El gradient boosting la usa internamente para entrenar; para reportar elegimos Brier, que se interpreta más fácil. |
| **MSE, MAE, R²** | No | Son métricas de **regresión** (predecir un número continuo); este es un problema de **clasificación**. |
| **Métricas de validación cruzada aleatoria (k-fold)** | No para el resultado final | Mezclar meses al azar dejaría al modelo "ver el futuro" y daría cifras optimistas. Validamos en el tiempo (ver sección 8). |

---

## 8. Otras decisiones de método y por qué

| Decisión | Lo que hicimos | Lo que no hicimos y por qué |
|---|---|---|
| **Validación** | Temporal: aprender con agosto, ajustar con septiembre, probar **una sola vez** en octubre | Validación cruzada aleatoria: mezcla meses y el impago cambia de 19,2 % a 23,5 % entre meses. Usar octubre para ajustar y para probar habría inflado los resultados. |
| **Desbalance de clases** | Se mantuvo la proporción real; el desbalance se maneja con las métricas y el corte | **SMOTE, sobremuestreo o `class_weight="balanced"`**: alteran las probabilidades (las vuelven poco realistas) y no mejoran el orden de los clientes, que es lo que importa. |
| **Umbral de decisión** | Por orden: aprobar el X % de menor riesgo, con zonas aprobar / revisar / rechazar | **Umbral de 0,5**: casi ningún cliente lo supera y no responde a ninguna meta del negocio. |
| **Selección de variables** | Valor de información (IV), correlación ≥ 0,8, V de Cramér, disponibilidad antes de aprobar y ética; se probaron variables nuevas y el concesionario | **PCA**: mezcla variables y Riesgo no podría explicar por qué se rechaza a un cliente; además solo hay 15 variables. |
| **Algoritmos** | Regresión logística (Lasso, Ridge y WoE), Random Forest y gradient boosting | **SVM, KNN y redes neuronales**: lentos con 233 mil registros, sin probabilidades directas o difíciles de explicar; en datos tabulares no superan a los árboles (Grinsztajn et al., 2022). **XGBoost / LightGBM**: el mismo método que nuestro gradient boosting. |
| **Hiperparámetros** | 29 configuraciones evaluadas en septiembre por AUC | Búsquedas más grandes: las configuraciones difieren en milésimas; el límite está en los datos. |
| **Calibración** | Se probó la calibración sigmoide | No se adoptó porque no mejoró; se recomienda recalibrar mensualmente en producción. |
| **Combinar modelos (stacking)** | Dos modelos independientes: uno decide, otro explica | No combinamos: ordenan de forma muy parecida y un modelo combinado sería más difícil de explicar. |

---

## 9. Cómo explicarlo en una frase

> "Medimos los modelos con AUC, KS y PR-AUC porque las clases están desbalanceadas y lo que importa es ordenar bien a los
> clientes; el accuracy engaña, porque decir que todos pagan ya da 76,5 %. El mejor modelo tiene un AUC de 0,648, 2,7
> veces el poder de la regla del buró, y en la métrica del negocio baja el impago de los aprobados de 23,5 % a 21,4 %
> aprobando el 88 %, con 98 % de confianza."

---

## Referencias

- Lessmann, S., Baesens, B., Seow, H.-V. y Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*, 247(1), 124–136.
- Siddiqi, N. (2017). *Intelligent credit scoring: Building and implementing better credit risk scorecards* (2.ª ed.). Wiley.
- Niculescu-Mizil, A. y Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625–632.
- Grinsztajn, L., Oyallon, E. y Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 35*.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
