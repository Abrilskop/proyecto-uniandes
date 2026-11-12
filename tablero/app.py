"""
app.py
Tablero en Dash · Aprobación de créditos vehiculares (Tarea 5 del reto · Soporte 5).

Estructura de las sesiones 6 y 7: estilos -> datos -> funciones (KPI y figuras) -> layout -> callbacks -> run.

Usuarios y preguntas
| Usuario            | Pregunta                                                          | KPI                                    | Gráfico                               |
|--------------------|-------------------------------------------------------------------|----------------------------------------|---------------------------------------|
| Gerencia de Riesgo | ¿Desde qué probabilidad rechazar para bajar el impago < 21,71 %?  | Impago de los aprobados, monto evitado | Curva aprobación vs impago, confusión |
| Gerencia de Riesgo | ¿Por qué se rechaza a un cliente?                                 | Coeficientes / aporte por variable     | Barras de coeficientes                |
| Gerencia de Ventas | ¿Qué sucursales y fabricantes concentran buenos pagadores?        | Probabilidad de pago media             | Burbujas por sucursal, barras         |
| Gerencia de Ventas | ¿Cuántos clientes viables más aprueba el modelo que la regla?     | Aprobados con el mismo límite          | Barras modelo vs regla del buró       |
| Ambas              | ¿Aprobar a este cliente?                                          | Probabilidad de pago y decisión        | Formulario                            |

Uso local:   python app.py      -> abrir http://127.0.0.1:8050
Nube:        gunicorn --chdir tablero app:server
Datos: salidas de modelado/modelado.py (predicciones de octubre, modelos .pkl y config_modelos.json).
"""
import json
import os
import sys
import warnings
from datetime import date

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, os.path.join(RAIZ, "limpieza_datos"))
sys.path.insert(0, os.path.join(RAIZ, "modelado"))
import preparacion as prep   # noqa: E402  (necesario para cargar los .pkl)
import ingenieria            # noqa: E402,F401  (necesario para cargar los .pkl)

SALIDAS = os.path.join(RAIZ, "modelado", "salidas")

# ---------------------------------------------------------------------------
# Estilos reutilizables (los mismos de las sesiones 6 y 7)
# ---------------------------------------------------------------------------
COLOR = {"fondo": "#F5F7FA", "texto": "#1F2937", "primario": "#1F4E79", "suave": "#6B7280",
         "bien": "#1E8449", "mal": "#C0392B", "alerta": "#B9770E"}
ESTILO_PAGINA = {"fontFamily": "Arial, sans-serif", "backgroundColor": COLOR["fondo"],
                 "padding": "20px", "color": COLOR["texto"]}
ESTILO_TARJETA = {"backgroundColor": "white", "borderRadius": "10px", "padding": "14px 18px",
                  "boxShadow": "0 1px 4px rgba(0,0,0,.08)", "flex": "1", "minWidth": "160px",
                  "boxSizing": "border-box"}
ESTILO_FILA = {"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"}
ESTILO_CAJA = {"backgroundColor": "white", "borderRadius": "10px", "padding": "8px",
               "boxShadow": "0 1px 4px rgba(0,0,0,.08)", "flex": "1", "minWidth": "380px"}
ESTILO_AYUDA = {"color": COLOR["suave"], "fontSize": "14px", "marginTop": "0"}
MARGEN = dict(t=50, l=10, r=10, b=10)
COLOR_MODELO = {"Gradient boosting": "#D62728", "Regresión logística": "#1F77B4",
                "Random Forest": "#2CA02C", "Regla del buró": "#7F7F7F"}


def tarjeta(titulo, valor, id_valor=None):
    """Tarjeta KPI: título pequeño + valor grande. Con id_valor, el valor se actualiza con un callback."""
    props = {"id": id_valor} if id_valor else {}
    return html.Div([
        html.Div(titulo, style={"fontSize": "13px", "color": COLOR["suave"]}),
        html.Div(valor, style={"fontSize": "26px", "fontWeight": "700", "color": COLOR["primario"]}, **props),
    ], style=ESTILO_TARJETA)


def caja(id_=None, figura=None):
    grafico = dcc.Graph(id=id_) if id_ else dcc.Graph(figure=figura)
    return html.Div(grafico, style=ESTILO_CAJA)


def pct(x, dec=2):
    return "—" if x is None or pd.isna(x) else f"{100 * x:.{dec}f} %".replace(".", ",")


def num(x):
    return "—" if x is None or pd.isna(x) else f"{x:,.0f}".replace(",", " ")


def millones(x):
    return f"{x / 1e6:,.1f} M".replace(",", " ").replace(".", ",")


# ---------------------------------------------------------------------------
# Datos (salidas del modelado)
# ---------------------------------------------------------------------------
with open(os.path.join(SALIDAS, "config_modelos.json"), encoding="utf-8") as fh:
    CONFIG = json.load(fh)
LIMITE = CONFIG["limite_impago"]
APROBACION = CONFIG["tasa_aprobacion"]
VARIABLES = CONFIG["variables_entrada"]
TIPOS_TEXTO = {c: str for c in CONFIG["tipos_texto"]}

# Solo las columnas que usa el tablero y tipos livianos (el plan gratis de Render tiene 512 MB de RAM)
pred = pd.read_csv(os.path.join(SALIDAS, "predicciones_octubre.csv"),
                   usecols=["BRANCH_ID", "MANUFACTURER_ID", "DISBURSED_AMOUNT", "modelo", "y_true", "proba"],
                   dtype={**TIPOS_TEXTO, "DISBURSED_AMOUNT": "float32", "y_true": "int8", "proba": "float32"})
MODELOS = [m for m in COLOR_MODELO if m in set(pred["modelo"])]
POR_MODELO = {}
for _m in MODELOS:
    _d = pred[pred["modelo"] == _m].drop(columns="modelo").reset_index(drop=True)
    _d["rango"] = _d["proba"].rank(method="first", pct=True).astype("float32")   # 0 = menor riesgo
    POR_MODELO[_m] = _d
del pred, _d, _m

MODELO_VENTAS = joblib.load(os.path.join(SALIDAS, CONFIG["modelos"]["ventas"]["archivo"]))
MODELO_RIESGO = joblib.load(os.path.join(SALIDAS, CONFIG["modelos"]["riesgo"]["archivo"]))
NOVIEMBRE = pd.read_csv(os.path.join(SALIDAS, "predicciones_noviembre.csv"))
COEF = pd.read_csv(os.path.join(SALIDAS, "tablas", "coeficientes_logistica.csv"))
IMPORTANCIA = pd.read_csv(os.path.join(SALIDAS, "tablas", "importancia_boosting.csv"))

NOMBRES = {
    "DISBURSED_AMOUNT": "Monto del préstamo", "PRI_SANCTIONED_AMOUNT": "Deuda aprobada en otras cuentas",
    "LTV": "LTV (% financiado)", "PRI_NO_OF_ACCTS": "N.º de cuentas", "PRI_ACTIVE_ACCTS": "Cuentas activas",
    "PRI_OVERDUE_ACCTS": "Cuentas en mora", "HISTORIAL_CRED_M": "Historial crediticio (meses)",
    "NO_OF_INQUIRIES": "Consultas al buró", "DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS": "Cuentas morosas (6 meses)",
    "NEW_ACCTS_IN_LAST_SIX_MONTHS": "Cuentas nuevas (6 meses)", "EDAD": "Edad", "CNS_GRUPO": "Categoría del buró",
    "EMPLOYMENT_TYPE": "Tipo de empleo", "BRANCH_ID": "Sucursal", "MANUFACTURER_ID": "Fabricante",
}
ORDEN_BURO = ["Muy bajo", "Bajo", "Sin puntaje", "Sin historial", "Medio", "Alto", "Muy alto"]
SUCURSALES = sorted(POR_MODELO[MODELOS[0]]["BRANCH_ID"].unique(), key=lambda s: int(s) if s.isdigit() else 10**9)
FABRICANTES = sorted(POR_MODELO[MODELOS[0]]["MANUFACTURER_ID"].unique(), key=lambda s: int(s) if s.isdigit() else 10**9)


# ---------------------------------------------------------------------------
# Funciones de KPI (como metricas(df, umbral) en la sesión 7)
# ---------------------------------------------------------------------------
def aplicar_corte(modelo, tasa):
    """Aprueba el `tasa` % de solicitudes de menor riesgo (el corte se calcula con todo octubre)."""
    return POR_MODELO[modelo].assign(aprobado=lambda d: d["rango"] <= tasa)


def kpis(d):
    if len(d) == 0:
        return dict(n=0, pct_aprob=np.nan, impago_aprob=np.nan, impago_real=np.nan, evitados=0,
                    buenos_rech=0, monto_evitado=0.0, prob_pago=np.nan)
    ap, re = d[d["aprobado"]], d[~d["aprobado"]]
    return dict(n=len(d), pct_aprob=len(ap) / len(d),
                impago_aprob=ap["y_true"].mean() if len(ap) else np.nan,
                impago_real=d["y_true"].mean(), evitados=int(re["y_true"].sum()),
                buenos_rech=int((1 - re["y_true"]).sum()),
                monto_evitado=float(re.loc[re["y_true"] == 1, "DISBURSED_AMOUNT"].sum()),
                prob_pago=1 - ap["proba"].mean() if len(ap) else np.nan)


def max_aprobados(d, modelo):
    """Máximo de créditos aprobables con impago de los aprobados < LIMITE.
    Regla del buró: solo categorías completas (de menor a mayor riesgo)."""
    if modelo == "Regla del buró":
        g = d.groupby("proba")["y_true"].agg(["size", "sum"]).sort_index().cumsum()
        ok = g[g["sum"] / g["size"] < LIMITE]
        return int(ok["size"].max()) if len(ok) else 0
    y = d.sort_values("proba", kind="stable")["y_true"].to_numpy()
    tasa = np.cumsum(y) / np.arange(1, len(y) + 1)
    idx = np.where(tasa < LIMITE)[0]
    return int(idx.max() + 1) if len(idx) else 0


def curva(d, tasas):
    y = d.sort_values("rango")["y_true"].to_numpy()
    acum = np.cumsum(y)
    return [acum[max(int(round(t * len(y))), 1) - 1] / max(int(round(t * len(y))), 1) for t in tasas]


def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float((tpr - fpr).max())


TASAS = np.round(np.arange(0.30, 1.0001, 0.01), 2)
CURVAS = {m: curva(POR_MODELO[m], TASAS) for m in MODELOS}
MAXIMOS = {m: max_aprobados(POR_MODELO[m], m) for m in MODELOS}
METRICAS = pd.DataFrame({m: {"AUC": roc_auc_score(d["y_true"], d["proba"]),
                             "KS": ks(d["y_true"], d["proba"]),
                             "PR-AUC": average_precision_score(d["y_true"], d["proba"])}
                         for m, d in POR_MODELO.items()}).T
N_OCT = len(POR_MODELO[MODELOS[0]])
IMPAGO_OCT = POR_MODELO[MODELOS[0]]["y_true"].mean()


# ---------------------------------------------------------------------------
# Figuras (cada una es una función que devuelve una figura, como en S6 y S7)
# ---------------------------------------------------------------------------
def fig_curva(tasa):
    fig = go.Figure()
    for m in MODELOS:
        fig.add_trace(go.Scatter(x=TASAS * 100, y=np.array(CURVAS[m]) * 100, mode="lines", name=m,
                                 line=dict(color=COLOR_MODELO[m], width=3 if m != "Regla del buró" else 2)))
    fig.add_hline(y=LIMITE * 100, line_dash="dash", line_color=COLOR["mal"],
                  annotation_text=f"Meta CN2: {pct(LIMITE)}", annotation_position="bottom left")
    fig.add_vline(x=tasa * 100, line_dash="dot", line_color=COLOR["primario"],
                  annotation_text=f"Corte: {tasa:.0%}", annotation_position="top")
    return fig.update_layout(title="Aprobar menos baja el impago: el modelo lo logra con menos rechazos",
                             xaxis_title="% de solicitudes aprobadas", yaxis_title="% de impago de los aprobados",
                             legend=dict(orientation="h", y=-0.2), margin=MARGEN)


def fig_confusion(d):
    tabla = pd.crosstab(d["y_true"].map({0: "Pagó", 1: "No pagó"}),
                        d["aprobado"].map({True: "Aprobar", False: "Rechazar"}))
    tabla = tabla.reindex(index=["Pagó", "No pagó"], columns=["Aprobar", "Rechazar"], fill_value=0)
    fig = px.imshow(tabla.values, x=["Decisión: aprobar", "Decisión: rechazar"], y=["Real: pagó", "Real: no pagó"],
                    text_auto=True, color_continuous_scale="Blues",
                    title="Rechazados que no pagaban = impagos evitados")
    return fig.update_layout(coloraxis_showscale=False, margin=MARGEN)


def fig_probas(d, umbral):
    bordes = np.linspace(0, max(0.8, d["proba"].max()), 41)
    fig = go.Figure()
    for valor, nombre, color in [(0, "Pagó", COLOR["bien"]), (1, "No pagó", COLOR["mal"])]:
        conteo, _ = np.histogram(d.loc[d["y_true"] == valor, "proba"], bins=bordes)
        fig.add_trace(go.Bar(x=(bordes[:-1] + bordes[1:]) / 2, y=conteo, name=nombre, marker_color=color,
                             opacity=0.65))
    fig.add_vline(x=umbral, line_dash="dash", line_color=COLOR["primario"],
                  annotation_text=f"Corte P(impago) = {umbral:.3f}")
    return fig.update_layout(barmode="overlay", title="Quienes no pagan tienen probabilidades más altas",
                             xaxis_title="Probabilidad de impago estimada", yaxis_title="Créditos", margin=MARGEN)


def fig_coeficientes():
    c = COEF.head(15).copy()
    c["variable"] = (c["variable"].str.replace(r"^(cat|num|log)__", "", regex=True)
                     .str.replace("BRANCH_ID_", "Sucursal ").str.replace("CNS_GRUPO_", "Buró: ")
                     .str.replace("MANUFACTURER_ID_", "Fabricante ").replace(NOMBRES))
    c = c.iloc[::-1]
    fig = go.Figure(go.Bar(x=c["coeficiente"], y=c["variable"], orientation="h",
                           marker_color=np.where(c["coeficiente"] > 0, COLOR["mal"], COLOR["bien"])))
    return fig.update_layout(title="Regresión logística: qué sube (rojo) o baja (verde) el riesgo",
                             xaxis_title="Coeficiente (log-odds)", margin=MARGEN, height=480)


def fig_comparacion():
    largo = METRICAS.reset_index(names="modelo").melt(id_vars="modelo", var_name="métrica")
    fig = px.bar(largo, x="métrica", y="value", color="modelo", barmode="group", text_auto=".3f",
                 color_discrete_map=COLOR_MODELO,
                 title="Los 3 modelos superan a la regla del buró (octubre)")
    return fig.update_layout(yaxis_title="", margin=MARGEN, legend=dict(orientation="h", y=-0.2))


def resumen_canal(d, col, minimo=200):
    t = (d.groupby(col).agg(creditos=("y_true", "size"), impago_real=("y_true", "mean"),
                            pct_aprobado=("aprobado", "mean"), p_impago=("proba", "mean"))
         .reset_index())
    t["prob_pago"] = 1 - t["p_impago"]
    return t[t["creditos"] >= minimo].sort_values("prob_pago", ascending=False)


def fig_sucursales(t):
    fig = px.scatter(t, x="creditos", y=t["prob_pago"] * 100, size="creditos", color=t["pct_aprobado"] * 100,
                     hover_data={"BRANCH_ID": True, "creditos": True},
                     color_continuous_scale="RdYlGn", labels={"x": "Créditos", "y": "Prob. de pago media (%)",
                                                              "color": "% aprobado"},
                     title="Sucursales: arriba están las que concentran buenos pagadores")
    fig.update_traces(text=t["BRANCH_ID"], hovertemplate="Sucursal %{text}<br>Créditos %{x}<br>"
                                                         "Prob. de pago %{y:.1f} %<extra></extra>")
    return fig.update_layout(xaxis_title="Créditos en octubre", yaxis_title="Probabilidad de pago media (%)",
                             margin=MARGEN)


def fig_fabricantes(t):
    t = t.sort_values("prob_pago")
    fig = go.Figure(go.Bar(x=t["prob_pago"] * 100, y="Fabricante " + t["MANUFACTURER_ID"], orientation="h",
                           marker_color=COLOR["primario"],
                           text=[f"impago real {pct(v, 1)}" for v in t["impago_real"]], textposition="inside"))
    return fig.update_layout(title="Fabricantes según probabilidad de pago media", margin=MARGEN,
                             xaxis_title="Probabilidad de pago media (%)",
                             xaxis_range=[min(70, t["prob_pago"].min() * 100 - 2), 100])


def fig_modelo_vs_regla():
    datos = pd.DataFrame({"Estrategia": ["Regla del buró", "Gradient boosting"],
                          "Aprobados": [MAXIMOS["Regla del buró"], MAXIMOS["Gradient boosting"]]})
    fig = px.bar(datos, x="Estrategia", y="Aprobados", text_auto=",", color="Estrategia",
                 color_discrete_map=COLOR_MODELO,
                 title=f"Con impago < {pct(LIMITE)}, el modelo aprueba "
                       f"{MAXIMOS['Gradient boosting'] / max(MAXIMOS['Regla del buró'], 1):.1f} veces más créditos")
    return fig.update_layout(showlegend=False, margin=MARGEN, yaxis_title="Créditos aprobables en octubre")


def fig_aporte(aporte):
    a = aporte.reindex(aporte.abs().sort_values(ascending=False).index).head(8).iloc[::-1]
    fig = go.Figure(go.Bar(x=a.values, y=[NOMBRES.get(v, v) for v in a.index], orientation="h",
                           marker_color=np.where(a.values > 0, COLOR["mal"], COLOR["bien"])))
    return fig.update_layout(title="¿Por qué? Variables que más suben (rojo) o bajan (verde) el riesgo",
                             xaxis_title="Aporte al riesgo (regresión logística)", margin=MARGEN, height=380)


# ---------------------------------------------------------------------------
# Resumen ejecutivo: conclusiones calculadas a partir de los datos (como en S7)
# ---------------------------------------------------------------------------
k90 = kpis(aplicar_corte("Gradient boosting", APROBACION))
k88 = kpis(aplicar_corte("Gradient boosting", 0.88))
k90_lr = kpis(aplicar_corte("Regresión logística", APROBACION))
top_imp = [NOMBRES.get(v, v) for v in IMPORTANCIA["variable"].head(3)]
dist_nov = NOVIEMBRE["DECISION_VENTAS"].value_counts(normalize=True)
CONCLUSIONES = [
    f"Aprobando el {APROBACION:.0%} de las solicitudes de octubre, el impago de los aprobados baja de "
    f"{pct(IMPAGO_OCT)} a {pct(k90['impago_aprob'])} con gradient boosting (meta: menos de {pct(LIMITE)}).",
    f"Con el mismo límite de impago, el modelo puede aprobar {num(MAXIMOS['Gradient boosting'])} créditos y la regla "
    f"del buró solo {num(MAXIMOS['Regla del buró'])}: {num(MAXIMOS['Gradient boosting'] - MAXIMOS['Regla del buró'])} "
    f"clientes viables más.",
    f"Rechazar el {1 - APROBACION:.0%} más riesgoso evita {num(k90['evitados'])} impagos y "
    f"{millones(k90['monto_evitado'])} desembolsados a clientes que no pagaron; el costo es rechazar "
    f"{num(k90['buenos_rech'])} buenos pagadores.",
    f"Las variables que más pesan son: {', '.join(top_imp)}.",
]
pestana_resumen = html.Div([
    html.H2("Conclusiones"), html.Ol([html.Li(c, style={"marginBottom": "8px"}) for c in CONCLUSIONES]),
    html.H3("Recomendación"),
    html.P(f"Usar gradient boosting para decidir y la regresión logística para explicar. El margen frente a la meta "
           f"es estrecho al {APROBACION:.0%} ({pct(k90['impago_aprob'])}); aprobando el 88 % el impago baja a "
           f"{pct(k88['impago_aprob'])}, con más holgura. Revisar manualmente la franja 'Revisar'."),
    html.H3("Noviembre (test.csv, sin etiqueta)"),
    html.P(f"Con los cortes del modelo final: {pct(dist_nov.get('Aprobar', 0), 0)} aprobar, "
           f"{pct(dist_nov.get('Revisar', 0), 0)} revisar y {pct(dist_nov.get('Rechazar', 0), 0)} rechazar."),
    html.H3("Limitaciones"),
    html.Ul([html.Li("Solo hay créditos aprobados: no se sabe cómo habrían pagado los solicitantes rechazados."),
             html.Li("La regresión logística queda al límite de la meta al 90 % "
                     f"({pct(k90_lr['impago_aprob'])}); conviene un corte algo menor si se usa para decidir."),
             html.Li(f"El modelo subestima el nivel de riesgo en meses malos: predijo en promedio "
                     f"{pct(POR_MODELO['Gradient boosting']['proba'].mean())} y el impago real fue {pct(IMPAGO_OCT)}."),
             html.Li("Las diferencias entre sucursales son asociaciones, no prueban que una sucursal gestione peor."),
             html.Li("El modelo apoya la decisión de crédito; no la reemplaza.")]),
], style={**ESTILO_TARJETA, "maxWidth": "950px", "marginTop": "16px"})


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
instrucciones = html.Details([
    html.Summary("¿Cómo usar este tablero? (clic para ver u ocultar)", style={"fontWeight": "700", "cursor": "pointer"}),
    html.Ul([
        html.Li([html.B("Riesgo: "), "mueve el control '% de solicitudes a aprobar'. Las tarjetas muestran si el "
                 "impago de los aprobados queda bajo la meta de 21,71 % (verde) y cuánto se evita."]),
        html.Li([html.B("Ventas: "), "identifica sucursales y fabricantes con buenos pagadores y compara cuántos "
                 "clientes aprueba el modelo frente a la regla del buró. Puedes descargar las decisiones de noviembre."]),
        html.Li([html.B("Evaluar cliente: "), "ingresa los datos de la solicitud y pulsa 'Calcular' para ver la "
                 "probabilidad de pago, la decisión sugerida y las variables que la explican."]),
        html.Li([html.B("Resumen ejecutivo: "), "conclusiones, recomendación y limitaciones."]),
        html.Li("Los resultados de Riesgo y Ventas se calculan con octubre de 2018, un mes que el modelo no vio al entrenar."),
    ]),
], open=True, style={**ESTILO_TARJETA, "marginBottom": "16px"})

marcas_tasa = {v: f"{v} %" for v in range(50, 101, 10)}

pestana_riesgo = html.Div([
    html.P("Pregunta (PE2): ¿desde qué probabilidad de impago revisar o rechazar para que el impago de los "
           "aprobados quede por debajo de 21,71 %?", style=ESTILO_AYUDA),
    html.Div([
        html.Div([html.Label("Modelo"), dcc.Dropdown(id="r-modelo", options=MODELOS, value="Regresión logística",
                                                     clearable=False)], style={"flex": "1"}),
        html.Div([html.Label("% de solicitudes a aprobar"),
                  dcc.Slider(id="r-tasa", min=50, max=100, step=1, value=int(APROBACION * 100), marks=marcas_tasa,
                             tooltip={"placement": "bottom", "always_visible": True})], style={"flex": "2"}),
        html.Div([html.Label("Sucursal"), dcc.Dropdown(id="r-sucursal", options=["Todas"] + SUCURSALES,
                                                       value="Todas", clearable=False)], style={"flex": "1"}),
    ], style={**ESTILO_TARJETA, "display": "flex", "gap": "20px", "alignItems": "end", "marginBottom": "16px",
              "flexWrap": "wrap"}),
    html.Div([tarjeta("% aprobado", "", "r-k-aprob"), tarjeta("Impago de los aprobados", "", "r-k-impago"),
              tarjeta("Impagos evitados", "", "r-k-evitados"), tarjeta("Buenos pagadores rechazados", "", "r-k-buenos"),
              tarjeta("Monto evitado", "", "r-k-monto"), tarjeta("AUC del modelo", "", "r-k-auc")], style=ESTILO_FILA),
    html.Div([caja("r-g-curva"), caja("r-g-conf")], style=ESTILO_FILA),
    html.Div([caja("r-g-prob"), caja(figura=fig_coeficientes())], style=ESTILO_FILA),
    html.Div([caja(figura=fig_comparacion())], style=ESTILO_FILA),
], style={"paddingTop": "16px"})

pestana_ventas = html.Div([
    html.P("Pregunta (PE1): ¿qué clientes, sucursales y fabricantes concentran buenos pagadores y cuántos clientes "
           "viables más se pueden aprobar? Modelo: gradient boosting.", style=ESTILO_AYUDA),
    html.Div([
        html.Div([html.Label("% de solicitudes a aprobar"),
                  dcc.Slider(id="v-tasa", min=50, max=100, step=1, value=int(APROBACION * 100), marks=marcas_tasa,
                             tooltip={"placement": "bottom", "always_visible": True})], style={"flex": "3"}),
        html.Div([html.Button("Descargar decisiones de noviembre (CSV)", id="v-btn-csv",
                              style={"height": "38px"}), dcc.Download(id="v-descarga")], style={"flex": "1"}),
    ], style={**ESTILO_TARJETA, "display": "flex", "gap": "20px", "alignItems": "end", "marginBottom": "16px",
              "flexWrap": "wrap"}),
    html.Div([tarjeta("Créditos aprobados (octubre)", "", "v-k-aprob"),
              tarjeta("Prob. de pago media de los aprobados", "", "v-k-prob"),
              tarjeta("Impago de los aprobados", "", "v-k-impago"),
              tarjeta("Clientes viables más que la regla del buró",
                      "+" + num(MAXIMOS["Gradient boosting"] - MAXIMOS["Regla del buró"]))], style=ESTILO_FILA),
    html.Div([caja("v-g-suc"), caja(figura=fig_modelo_vs_regla())], style=ESTILO_FILA),
    html.Div([caja("v-g-fab")], style=ESTILO_FILA),
    html.Div([html.H4("Sucursales con 200 o más créditos (ordena con clic en la columna)"),
              dash_table.DataTable(id="v-tabla", page_size=10, sort_action="native",
                                   columns=[{"name": n, "id": i} for i, n in [
                                       ("BRANCH_ID", "Sucursal"), ("creditos", "Créditos"),
                                       ("prob_pago", "Prob. pago media (%)"), ("impago_real", "Impago real (%)"),
                                       ("pct_aprobado", "% aprobado")]],
                                   style_cell={"fontFamily": "Arial", "padding": "6px"})],
             style={**ESTILO_TARJETA, "marginBottom": "16px"}),
], style={"paddingTop": "16px"})


def campo(etiqueta, componente):
    return html.Div([html.Label(etiqueta, style={"fontSize": "13px"}), componente],
                    style={"flex": "1", "minWidth": "200px"})


def numero(id_, valor, minimo=0):
    return dcc.Input(id=id_, type="number", value=valor, min=minimo, style={"width": "100%"})


pestana_cliente = html.Div([
    html.P("Ingresa los datos de la solicitud. La edad se calcula a la fecha de hoy y el LTV como "
           "monto del préstamo / valor del vehículo.", style=ESTILO_AYUDA),
    html.Div([
        html.H4("Cliente y préstamo"),
        html.Div([
            campo("Fecha de nacimiento", dcc.DatePickerSingle(id="c-nac", date="1990-01-01",
                                                              display_format="DD/MM/YYYY")),
            campo("Tipo de empleo", dcc.Dropdown(id="c-empleo", value="Salaried", clearable=False, options=[
                {"label": "Asalariado", "value": "Salaried"}, {"label": "Independiente", "value": "Self employed"},
                {"label": "Sin dato", "value": "Sin dato"}])),
            campo("Monto del préstamo", numero("c-monto", 55000, 1)),
            campo("Valor del vehículo", numero("c-valor", 75000, 1)),
        ], style=ESTILO_FILA),
        html.Div([
            campo("Sucursal", dcc.Dropdown(id="c-sucursal", value=SUCURSALES[0], clearable=False,
                                           options=SUCURSALES + [{"label": "Otra / nueva", "value": "OTRA"}])),
            campo("Fabricante", dcc.Dropdown(id="c-fabricante", value=FABRICANTES[0], clearable=False,
                                             options=FABRICANTES + [{"label": "Otro / nuevo", "value": "OTRA"}])),
            campo("Categoría del buró", dcc.Dropdown(id="c-buro", options=ORDEN_BURO, value="Sin historial",
                                                     clearable=False)),
        ], style=ESTILO_FILA),
        html.H4("Historial crediticio (buró)"),
        html.Div([
            campo("N.º de cuentas", numero("c-cuentas", 0)),
            campo("Cuentas activas", numero("c-activas", 0)),
            campo("Cuentas en mora", numero("c-mora", 0)),
            campo("Deuda aprobada en otras cuentas", numero("c-deuda", 0)),
        ], style=ESTILO_FILA),
        html.Div([
            campo("Historial: años", numero("c-hist-a", 0)),
            campo("Historial: meses", numero("c-hist-m", 0)),
            campo("Consultas al buró", numero("c-consultas", 0)),
            campo("Cuentas nuevas (6 meses)", numero("c-nuevas", 0)),
            campo("Cuentas morosas (6 meses)", numero("c-morosas", 0)),
        ], style=ESTILO_FILA),
        html.Button("Calcular", id="c-calcular", n_clicks=0,
                    style={"height": "40px", "width": "180px", "backgroundColor": COLOR["primario"],
                           "color": "white", "border": "none", "borderRadius": "6px", "fontWeight": "700"}),
    ], style={**ESTILO_TARJETA, "marginBottom": "16px"}),
    html.Div([html.Div(id="c-resultado", style={**ESTILO_TARJETA, "flex": "1"}),
              html.Div(dcc.Graph(id="c-g-aporte"), style={**ESTILO_CAJA, "flex": "2"})], style=ESTILO_FILA),
], style={"paddingTop": "16px"})

app = Dash(__name__, title="Créditos vehiculares")
server = app.server
app.layout = html.Div([
    html.H1("Aprobación de créditos vehiculares · Tablero de Riesgo y Ventas"),
    html.P(f"Validación: octubre de 2018 ({num(N_OCT)} créditos, impago real {pct(IMPAGO_OCT)}) · "
           f"Meta: impago de los aprobados < {pct(LIMITE)} aprobando al menos el {APROBACION:.0%}",
           style={"color": COLOR["suave"]}),
    instrucciones,
    dcc.Tabs([dcc.Tab(label="Riesgo", children=pestana_riesgo),
              dcc.Tab(label="Ventas", children=pestana_ventas),
              dcc.Tab(label="Evaluar cliente", children=pestana_cliente),
              dcc.Tab(label="Resumen ejecutivo", children=pestana_resumen)]),
], style=ESTILO_PAGINA)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def semaforo(valor, bueno):
    return html.Span(pct(valor), style={"color": COLOR["bien"] if bueno else COLOR["mal"]})


@app.callback(
    Output("r-k-aprob", "children"), Output("r-k-impago", "children"), Output("r-k-evitados", "children"),
    Output("r-k-buenos", "children"), Output("r-k-monto", "children"), Output("r-k-auc", "children"),
    Output("r-g-curva", "figure"), Output("r-g-conf", "figure"), Output("r-g-prob", "figure"),
    Input("r-modelo", "value"), Input("r-tasa", "value"), Input("r-sucursal", "value"),
)
def actualizar_riesgo(modelo, tasa_pct, sucursal):
    tasa = tasa_pct / 100
    todos = aplicar_corte(modelo, tasa)
    umbral = todos["proba"].quantile(tasa) if tasa < 1 else 1.0
    d = todos if sucursal == "Todas" else todos[todos["BRANCH_ID"] == sucursal]
    k = kpis(d)
    auc = roc_auc_score(d["y_true"], d["proba"]) if d["y_true"].nunique() > 1 else np.nan
    return (pct(k["pct_aprob"], 1), semaforo(k["impago_aprob"], k["impago_aprob"] < LIMITE), num(k["evitados"]),
            num(k["buenos_rech"]), millones(k["monto_evitado"]), "—" if pd.isna(auc) else f"{auc:.3f}",
            fig_curva(tasa), fig_confusion(d), fig_probas(d, umbral))


@app.callback(
    Output("v-k-aprob", "children"), Output("v-k-prob", "children"), Output("v-k-impago", "children"),
    Output("v-g-suc", "figure"), Output("v-g-fab", "figure"), Output("v-tabla", "data"),
    Input("v-tasa", "value"),
)
def actualizar_ventas(tasa_pct):
    d = aplicar_corte("Gradient boosting", tasa_pct / 100)
    k = kpis(d)
    suc = resumen_canal(d, "BRANCH_ID")
    tabla = suc.assign(prob_pago=(suc["prob_pago"] * 100).round(1), impago_real=(suc["impago_real"] * 100).round(1),
                       pct_aprobado=(suc["pct_aprobado"] * 100).round(1))
    return (f"{num(d['aprobado'].sum())} ({pct(k['pct_aprob'], 0)})", pct(k["prob_pago"], 1),
            semaforo(k["impago_aprob"], k["impago_aprob"] < LIMITE), fig_sucursales(suc),
            fig_fabricantes(resumen_canal(d, "MANUFACTURER_ID")),
            tabla[["BRANCH_ID", "creditos", "prob_pago", "impago_real", "pct_aprobado"]].to_dict("records"))


@app.callback(Output("v-descarga", "data"), Input("v-btn-csv", "n_clicks"), prevent_initial_call=True)
def descargar(_):
    return dcc.send_data_frame(NOVIEMBRE.to_csv, "decisiones_noviembre.csv", index=False)


def construir_cliente(nac, empleo, monto, valor, sucursal, fabricante, buro, cuentas, activas, mora, deuda,
                      hist_a, hist_m, consultas, nuevas, morosas, hoy=None):
    """Arma una fila con las 15 variables del modelo (mismo formato que los datos limpios) y la lista de alertas."""
    hoy = pd.Timestamp(hoy or date.today())
    v = lambda x: 0 if x is None else x   # noqa: E731
    edad = int(prep.edad_cumplida(pd.Series([hoy]), pd.Series([pd.Timestamp(nac)])).iloc[0])
    ltv = 100 * v(monto) / v(valor) if v(valor) > 0 else np.nan
    alertas = []
    if edad < 18:
        alertas.append(f"El cliente tiene {edad} años: es menor de edad.")
    if pd.isna(ltv) or ltv <= 0 or ltv > 100:
        alertas.append("El LTV debe estar entre 0 y 100 %: revisa el monto y el valor del vehículo.")
    if v(activas) > v(cuentas) or v(mora) > v(cuentas):
        alertas.append("Las cuentas activas o en mora no pueden superar el número de cuentas.")
    fila = pd.DataFrame([{
        "DISBURSED_AMOUNT": v(monto), "PRI_SANCTIONED_AMOUNT": v(deuda), "LTV": ltv,
        "PRI_NO_OF_ACCTS": v(cuentas), "PRI_ACTIVE_ACCTS": v(activas), "PRI_OVERDUE_ACCTS": v(mora),
        "HISTORIAL_CRED_M": 12 * v(hist_a) + v(hist_m), "NO_OF_INQUIRIES": v(consultas),
        "DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS": v(morosas), "NEW_ACCTS_IN_LAST_SIX_MONTHS": v(nuevas),
        "EDAD": edad, "CNS_GRUPO": buro, "EMPLOYMENT_TYPE": empleo,
        "BRANCH_ID": str(sucursal), "MANUFACTURER_ID": str(fabricante),
    }])
    return fila[VARIABLES], edad, ltv, alertas


def decision(p, clave):
    c = CONFIG["modelos"][clave]
    if p >= c["corte_rechazar"]:
        return "Rechazar", COLOR["mal"]
    if p >= c["corte_revisar"]:
        return "Revisar", COLOR["alerta"]
    return "Aprobar", COLOR["bien"]


def aporte_logistica(fila):
    """Aporte de cada variable original al log-odds de impago (coeficiente × valor transformado)."""
    xt = MODELO_RIESGO[:-1].transform(fila).iloc[0]
    contrib = xt * MODELO_RIESGO[-1].coef_[0]
    por_variable = {}
    for nombre, valor in contrib.items():
        resto = nombre.split("__", 1)[1]
        var = next((v for v in VARIABLES if resto == v or resto.startswith(v + "_")), resto)
        por_variable[var] = por_variable.get(var, 0.0) + valor
    return pd.Series(por_variable)


@app.callback(
    Output("c-resultado", "children"), Output("c-g-aporte", "figure"),
    Input("c-calcular", "n_clicks"),
    State("c-nac", "date"), State("c-empleo", "value"), State("c-monto", "value"), State("c-valor", "value"),
    State("c-sucursal", "value"), State("c-fabricante", "value"), State("c-buro", "value"),
    State("c-cuentas", "value"), State("c-activas", "value"), State("c-mora", "value"), State("c-deuda", "value"),
    State("c-hist-a", "value"), State("c-hist-m", "value"), State("c-consultas", "value"),
    State("c-nuevas", "value"), State("c-morosas", "value"),
)
def evaluar_cliente(n_clicks, *valores):
    if not n_clicks:
        vacio = go.Figure().update_layout(title="Pulsa 'Calcular' para ver la explicación", margin=MARGEN, height=380)
        return html.Div("Completa el formulario y pulsa 'Calcular'.", style={"color": COLOR["suave"]}), vacio
    fila, edad, ltv, alertas = construir_cliente(*valores)
    if any("LTV" in a for a in alertas):
        return html.Div([html.B("No se puede calcular:"), html.Ul([html.Li(a) for a in alertas])],
                        style={"color": COLOR["mal"]}), go.Figure().update_layout(margin=MARGEN, height=380)
    p_v = float(MODELO_VENTAS.predict_proba(fila)[:, 1][0])
    p_r = float(MODELO_RIESGO.predict_proba(fila)[:, 1][0])
    dec_v, col_v = decision(p_v, "ventas")
    dec_r, col_r = decision(p_r, "riesgo")
    resultado = html.Div([
        html.Div("Probabilidad de pago", style={"fontSize": "13px", "color": COLOR["suave"]}),
        html.Div(pct(1 - p_v, 1), style={"fontSize": "40px", "fontWeight": "700", "color": col_v}),
        html.Div(["Decisión sugerida: ", html.B(dec_v, style={"color": col_v})], style={"fontSize": "20px"}),
        html.Div("(gradient boosting · cumple las metas de octubre)", style={"fontSize": "12px",
                                                                            "color": COLOR["suave"]}),
        html.Hr(),
        html.Div([f"Regresión logística (Riesgo): {pct(1 - p_r, 1)} de pago · ",
                  html.B(dec_r, style={"color": col_r})]),
        html.Div(f"Edad calculada: {edad} años · LTV: {ltv:.1f} %".replace(".", ","),
                 style={"marginTop": "6px", "color": COLOR["suave"]}),
        html.Ul([html.Li(a, style={"color": COLOR["alerta"]}) for a in alertas]) if alertas else None,
        html.Div("Rechazar = está entre el 10 % de solicitudes más riesgosas; Revisar = el siguiente 5 %.",
                 style={"fontSize": "12px", "color": COLOR["suave"], "marginTop": "8px"}),
        html.Div("Ojo: el modelo acierta al ordenar a los clientes, pero en octubre el impago real fue cerca de "
                 "1,2 veces el que estimó. Usa la probabilidad para comparar clientes, no como cifra exacta.",
                 style={"fontSize": "12px", "color": COLOR["alerta"], "marginTop": "6px"}),
    ])
    return resultado, fig_aporte(aporte_logistica(fila))


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 8050)))
