# Guía para la revisión del proyecto

Para leer antes de presentar o de que revisen el proyecto. Va de lo más importante a lo más detallado.
Si solo tienes 2 minutos, lee la parte 1 y la chuleta del final.

---

## 1 · Lo esencial en 1 minuto

> "Una fintech presta para comprar carros y **22 de cada 100 clientes no pagan la primera cuota**. Hoy deciden con la
> calificación del buró, pero **la mitad de los clientes no tiene historial**, así que el buró no sirve con ellos.
> Construimos un **modelo que combina 15 datos del cliente** y estima su probabilidad de pagar. Lo probamos con
> octubre, un mes que no vio al aprender: **aprobando el 88 % de las solicitudes, el impago baja de 23,5 % a 21,4 %**,
> por debajo de la meta de 21,71 %, y con el mismo límite de impago **aprueba 2,6 veces más clientes que la regla del
> buró**. Todo está en un **tablero web** para las gerencias de Riesgo y Ventas."

---

## 2 · Las metas del proyecto (CN y CML)

**CN = criterio de éxito del negocio.** Una meta por gerencia:

| Código | Para quién | Qué exige | Resultado |
|---|---|---|---|
| **CN1** | Ventas | Conservar al menos el **90 %** de las aprobaciones | Se cumple al 90 %, pero entonces CN2 queda al límite |
| **CN2** | Riesgo | Impago de los aprobados **menor a 21,71 %** | ✅ Al 88 %: 21,40 % (98 de cada 100 repeticiones) |
| CN3 | Ambas | Que cada gerencia responda su pregunta con el tablero | Falta validarlo con usuarios |

**El dilema:** CN1 pide aprobar mucho y CN2 pide poco impago. Al 90 % se cumple CN1, pero CN2 queda como una moneda al
aire (21,70 % frente a 21,71 %). Recomendamos el **88 %** y que decidan las gerencias.

**CML = criterio del modelo (machine learning):** CML1 ganarle al azar y a la regla del buró · CML2 medir con AUC, KS,
PR-AUC y calibración, no con accuracy · CML3 encontrar un corte que cumpla CN1 y CN2 · CML4 desempeño mínimo.

---

## 3 · Los tipos de variables

La tabla completa está en `eda/tipos_de_variables.csv` y en `eda/eda.ipynb` (parte 0).

| Tipo | Qué es | Ejemplos |
|---|---|---|
| **Numérica continua** | Cantidad con decimales | LTV (73,3 %), monto |
| **Numérica discreta** | Conteo | Número de cuentas, edad en años |
| **Categórica nominal** | Grupos sin orden | Sucursal, fabricante, tipo de empleo |
| **Categórica ordinal** | Grupos con orden | Buró: muy bajo < bajo < medio < alto < muy alto |
| **Binaria** | Dos valores | **Pagó o no** (`LOAN_DEFAULT`, lo que se predice) |

- **"Pagó o no" es binaria**, por eso es un problema de **clasificación**. El modelo entrega una **probabilidad
  continua** (por ejemplo 0,117) que se convierte en una decisión (aprobar, revisar o rechazar).
- **Trampa 1:** la sucursal parece un número, pero es una etiqueta (la 152 no es "el doble" que la 76). Se trata como texto.
- **Trampa 2:** el buró es ordinal solo en parte: "sin historial" y "sin puntaje" no caben en la escala, por eso se
  trata como nominal.

**Qué medida de relación usar según el tipo:**

| Par | Medida | Por qué |
|---|---|---|
| Numérica – numérica | **Spearman** | Los montos tienen colas largas y muchos ceros; Spearman compara el orden |
| Categórica – categórica | **V de Cramér** | Sirve para grupos sin orden |
| Numérica – categórica | **η (eta)** | Mide cuánto cambia el promedio entre grupos |

---

## 4 · ¿Por qué esas 15 variables?

De **41 columnas** quedaron **15**. Cada columna pasó 4 filtros:

| Filtro | Pregunta | Ejemplo de lo que se quitó |
|---|---|---|
| 1 | ¿Se conoce **antes de aprobar**? | Código del empleado que desembolsa (se registra después: sería trampa) |
| 2 | ¿**Ayuda a predecir**? (valor de información, IV) | 8 columnas de cuentas secundarias (el 97 % tiene 0) |
| 3 | ¿**Repite** a otra? (correlación ≥ 0,8) | Monto desembolsado y saldo (se parecen 0,92–0,99 al monto aprobado) |
| 4 | ¿Es **justa y estable**? | Código postal (discrimina por zona, 6 698 valores) |

| Grupo | Variables | Evidencia |
|---|---|---|
| Préstamo | **LTV**, monto | **La más importante.** Impago de 14 % (LTV ≤ 60 %) a 26 % (LTV 80–90 %) |
| Buró | **Categoría del buró** | De 16 % ("muy bajo") a 30 % ("muy alto"). La tercera más importante |
| Canal | **Sucursal**, fabricante | De 13 % a 34 % según la sucursal. La segunda más importante |
| Historial | Cuentas, cuentas activas, en mora, deuda aprobada, meses de historial | Con cuentas en mora: 27 % frente a 21 % |
| Comportamiento reciente | Consultas al buró, cuentas morosas y nuevas en 6 meses | Con consultas: 26 % frente a 21 % |
| Perfil | Edad, tipo de empleo | Independientes 23 %, asalariados 20 %. La edad se revisó por equidad |

Además se probó agregar 3 variables nuevas y el concesionario: **no mejoraron** el modelo.

---

## 5 · Los gráficos, paso a paso

**Cómo leer cualquier gráfico:** (1) ¿qué hay en los ejes? (2) ¿qué patrón se ve? (3) ¿qué significa para el negocio?
El título de cada gráfico ya dice el hallazgo.

### Del EDA (`eda/graficos/`)

| Gráfico | Qué mirar | Qué decir |
|---|---|---|
| **03 buró y sucursal** ⭐ | Barras de % de impago; línea roja = meta; **gris = sin historial** | "El buró ordena el riesgo, pero la barra gris es la mitad de los clientes, está sobre la meta y el buró no los distingue. La sucursal 251 tiene 34 % de impago; la 152, 13 %." |
| **02 caja y violín** | La caja roja (no paga) está más arriba en LTV; los violines de edad son casi iguales | "El LTV separa a los grupos; la edad casi no." |
| **04 barras 100 %** | El número de la parte roja sube con el LTV | "Financiar más se paga en riesgo: de 14 a 26 de cada 100." |
| **05 dispersión** | Puntos rojos mezclados con verdes en todas partes | "Ninguna variable sola separa; hay que combinarlas." |
| **06 correlaciones** | Cuadros rojo oscuro = variables casi iguales | "Por eso se dejó una variable de cada grupo." |
| **00 asociación por tipo** | La última fila: ninguna pasa de 0,11 | "Ninguna variable sola explica el impago." |
| **09 impago por mes** | 21,6 %, 19,2 %, 23,5 % | "El riesgo cambia; por eso probamos con un mes que el modelo no vio." |

### Del modelado (`modelado/salidas/graficos/`)

| Gráfico | Qué mirar | Qué decir |
|---|---|---|
| **curva_aprobacion_impago** ⭐⭐ | Eje X = % aprobado, eje Y = % que no paga. Más abajo es mejor. Línea roja = meta; negra = 88 % | "Para cualquier cantidad de aprobaciones, el modelo tiene menos impago que la regla. La regla llega a la meta aprobando el 35 %; el modelo, el 88 %." |
| **roc_octubre** | Más arriba y a la izquierda es mejor; la diagonal es el azar | "Los modelos (AUC 0,635–0,648) superan a la regla (0,554), que casi coincide con el azar." |
| **importancia_boosting** | Barras más largas = más importantes | "LTV, sucursal y buró son lo que más pesa." |
| **coeficientes_logistica** | Rojo sube el riesgo; verde lo baja | "La sucursal 251 multiplica el riesgo por 2,2. Son asociaciones, no causas." |
| **calibracion_octubre** | Puntos por encima de la diagonal | "El modelo se quedó corto: esperaba 19,8 de cada 100 impagos y fueron 23,5. Acierta el orden, no el nivel; por eso decidimos por orden." |

---

## 6 · Preguntas probables y respuestas cortas

| Pregunta | Respuesta |
|---|---|
| ¿Qué modelos usaron? | Probamos 4: regresión logística (Lasso, Ridge y WoE), Random Forest y gradient boosting. Usamos gradient boosting para decidir y regresión logística para explicar. Son **independientes**: ninguno alimenta al otro. |
| ¿Por qué esos? | Son el estándar en crédito (Lessmann et al., 2015). El boosting ordena mejor; la logística se explica con coeficientes. |
| ¿Por qué no redes neuronales, SVM o KNN? | Lentos o difíciles de explicar, y en datos de tabla no superan a los árboles. Los 4 modelos llegaron al mismo techo: el límite está en los datos. |
| ¿Cómo entrenaron? | Aprende con agosto, ajustamos con septiembre y probamos **una sola vez** con octubre, que nunca vio. |
| ¿Qué accuracy tiene? | 73,9 %, pero no es la métrica correcta: decir "todos pagan" ya da 76,5 % sin detectar a nadie. Usamos AUC 0,648, KS 0,218 y el impago de los aprobados (21,4 %). |
| ¿Qué es el AUC? | Si tomo un cliente que pagó y uno que no, es la probabilidad de que el modelo le ponga más riesgo al que no pagó. 0,5 es azar. |
| ¿No es bajo 0,65? | Es moderado: la mitad no tiene historial y no tenemos el ingreso. Aun así es 2,7 veces mejor que la regla y cumple la meta del negocio. |
| ¿Cómo clasifica a un cliente? | Calcula su probabilidad de no pagar y lo ordena: el 85 % más seguro se aprueba, el siguiente 5 % se revisa y el 10 % más riesgoso se rechaza. |
| ¿Cuánto puede aprobar Ventas? | Hasta el 88 % con seguridad (21,4 %). Entre el 88 % y el 90 %, 1 de cada 3 clientes adicionales no paga. |
| ¿Por qué no balancearon las clases (SMOTE)? | Distorsiona las probabilidades y no mejora el orden de los clientes. |
| ¿Es justo el modelo? | Revisamos por edad y empleo: diferencias moderadas. Sin la edad el modelo casi no pierde (AUC 0,646), así que se puede retirar. |
| ¿Limitaciones? | Solo hay clientes aprobados, solo se mira la primera cuota, el modelo se queda corto en meses malos y la sucursal es asociación, no causa. |
| ¿En qué moneda están los montos? | El caso no lo dice. Los documentos (Aadhaar, PAN) sugieren India, pero no está confirmado; usamos las unidades del archivo. |

**Si no sabes algo:** "Eso lo trabajó más a fondo [nombre]. Lo que sí puedo decir es que [idea general]. Está
documentado en el repositorio." Admitir un límite con calma da mejor impresión que inventar.

---

## Chuleta (si solo recuerdas 6 cosas)

1. **22 de cada 100** no pagan la primera cuota.
2. La **mitad no tiene historial**, por eso el buró no alcanza.
3. **LTV, sucursal y buró** son lo que más pesa.
4. Aprobando el **88 %**, el impago baja de **23,5 % a 21,4 %**.
5. **AUC 0,648** (la regla del buró: 0,554).
6. **2,6 veces** más clientes que la regla del buró.

Más detalle: `README.md` (qué hay en cada archivo) · `modelado/metricas_y_decisiones.md` (métricas y decisiones) ·
`eda/eda.ipynb` y `modelado/modelado.ipynb` (la historia completa con gráficos).
