"""
modelado.py
Fase 3 (CRISP-ML(Q)): modelado - créditos vehiculares.

Uso (desde la carpeta modelado/):
    python modelado.py
    python modelado.py --datos ".." --limpios "../limpieza_datos/datos_v1" --salida salidas

Requisitos: pip install pandas numpy scikit-learn joblib matplotlib
Antes debe haberse corrido limpieza_datos.py (genera datos_v1/).

Esquema de entrenamiento, prueba y evaluación (sin fuga de información):
    A) Ajuste    : entrena con AGOSTO, mide en SEPTIEMBRE.
                   Aquí se decide qué variables usar y los hiperparámetros.
    B) Prueba    : entrena con AGOSTO+SEPTIEMBRE con lo decidido en A, mide en OCTUBRE.
                   Octubre se usa una sola vez: son los resultados que se reportan.
    C) Producción: reentrena con AGOSTO-OCTUBRE y predice NOVIEMBRE (test.csv, sin etiqueta).

Modelos:
    PE1 Ventas  -> HistGradientBoostingClassifier (mejor ordenamiento)
    PE2 Riesgo  -> Regresión logística (one-hot o WoE, se elige en A) (explicable)
    Comparación -> Random Forest
    Referencias -> clasificador trivial y regla del buró (CNS_GRUPO)

Salidas en --salida (por defecto salidas/):
    tablas/*.csv, graficos/*.png
    modelo_ventas_boosting.pkl, modelo_riesgo_logistica.pkl  (pipeline completo)
    config_modelos.json      variables, hiperparámetros y puntos de corte para el tablero
    predicciones_noviembre.csv
"""
import argparse
import itertools
import json
import os
import sys
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, os.path.join(RAIZ, "limpieza_datos"))
sys.path.insert(0, AQUI)
import preparacion as prep  # noqa: E402
import ingenieria as ing    # noqa: E402

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 30)

# ---------------------------------------------------------------------------
# Parámetros del negocio (Fase 1)
# ---------------------------------------------------------------------------
SEMILLA = 42
LIMITE = prep.LIMITE_IMPAGO          # CN2: impago de los aprobados < 21,71 %
APROBACION = 0.90                    # CN1: aprobar al menos el 90 %
FRANJA_REVISION = 0.05               # dentro del 90 % aprobado, el 5 % más riesgoso pasa a "revisar"
MEJORA_MINIMA = 0.002                # una variable extra se acepta si sube el AUC al menos 0,002
NUM_LOG = prep.NUM_LOG
NUM_BASE = prep.NUM
CAT_BASE = prep.CAT
ORDEN_BURO = ["Muy bajo", "Bajo", "Sin puntaje", "Sin historial", "Medio", "Alto", "Muy alto"]

np.random.seed(SEMILLA)
INICIO = time.perf_counter()


def titulo(texto):
    print("\n" + "=" * 90 + f"\n{texto}   [{time.perf_counter() - INICIO:,.0f} s]\n" + "=" * 90, flush=True)


# ---------------------------------------------------------------------------
# Construcción de modelos (pipeline completo: datos limpios -> probabilidad)
# ---------------------------------------------------------------------------
def columnas(nuevas, proveedor):
    """Columnas de entrada (datos limpios) y columnas numéricas/categóricas que ve el modelo."""
    cat = CAT_BASE + (["SUPPLIER_ID"] if proveedor else [])
    num = NUM_BASE + (ing.NUEVAS if nuevas else [])
    entrada = NUM_LOG + NUM_BASE + cat
    return entrada, num, cat


def construir(tipo, nuevas=False, proveedor=False, **params):
    entrada, num, cat = columnas(nuevas, proveedor)
    agrupar = [c for c in ["BRANCH_ID", "MANUFACTURER_ID", "SUPPLIER_ID"] if c in cat]
    pasos = [("nuevas", ing.AgregarVariables(activar=nuevas)),
             ("raros", ing.AgrupadorTop(agrupar, prep.MIN_REGISTROS_CATEGORIA))]

    if tipo == "logistica":          # one-hot + escalado
        ct = ColumnTransformer([
            ("log", Pipeline([("tope", prep.TopeLog()), ("escala", StandardScaler())]), NUM_LOG),
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ]).set_output(transform="pandas")
        pasos += [("columnas", ct), ("modelo", LogisticRegression(max_iter=3000, **params))]
    elif tipo == "logistica_woe":    # scorecard: todas las variables en WoE
        pasos += [("woe", ing.WoE(NUM_LOG + num, cat)),
                  ("modelo", LogisticRegression(max_iter=3000, **params))]
    elif tipo in ("boosting", "bosque"):
        ct = ColumnTransformer([
            ("log", prep.TopeLog(), NUM_LOG),
            ("num", "passthrough", num),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
        ]).set_output(transform="pandas")
        if tipo == "boosting":
            modelo = HistGradientBoostingClassifier(
                categorical_features=[f"cat__{c}" for c in cat], max_iter=500,
                early_stopping=True, random_state=SEMILLA, **params)
        else:
            modelo = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEMILLA, **params)
        pasos += [("columnas", ct), ("modelo", modelo)]
    else:
        raise ValueError(tipo)
    return Pipeline(pasos), entrada


def probar(tipo, X_tr, y_tr, X_ev, nuevas=False, proveedor=False, **params):
    modelo, entrada = construir(tipo, nuevas, proveedor, **params)
    modelo.fit(X_tr[entrada], y_tr)
    return modelo, modelo.predict_proba(X_ev[entrada])[:, 1]


# ---------------------------------------------------------------------------
# Métricas (CML2, CML3, CN1, CN2, CE1)
# ---------------------------------------------------------------------------
def max_aprobados(y, p):
    """Máximo de créditos aprobables (de menor a mayor riesgo) con impago de los aprobados < LIMITE."""
    orden = np.argsort(p, kind="stable")
    tasa = np.cumsum(y[orden]) / np.arange(1, len(y) + 1)
    ok = np.where(tasa < LIMITE)[0]
    return int(ok.max() + 1) if len(ok) else 0


def aprobados_al(p, tasa):
    """Máscara de aprobados: el `tasa` % de menor probabilidad de impago."""
    orden = np.argsort(p, kind="stable")
    aprob = np.zeros(len(p), dtype=bool)
    aprob[orden[: int(round(len(p) * tasa))]] = True
    return aprob


def metricas(y, p, monto, tasa=APROBACION):
    y, p, monto = np.asarray(y), np.asarray(p, dtype=float), np.asarray(monto, dtype=float)
    fpr, tpr, _ = roc_curve(y, p)
    aprob = aprobados_al(p, tasa)
    rech = ~aprob
    n_max = max_aprobados(y, p)
    return {
        "AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "KS": float((tpr - fpr).max()),
        "Brier": brier_score_loss(y, p),
        "Prob_media": p.mean(),
        "Impago_real": y.mean(),
        f"Impago_aprobados_{int(tasa*100)}": y[aprob].mean(),
        "Cumple_CN2": bool(y[aprob].mean() < LIMITE),
        "Impagos_evitados": int(y[rech].sum()),
        "Buenos_rechazados": int((1 - y[rech]).sum()),
        "Monto_evitado_CE1": float(monto[rech & (y == 1)].sum()),
        "Max_aprobados_CN2": n_max,
        "Max_aprob_pct_CN2": n_max / len(y),
    }


def regla_buro(cat_tr, y_tr, cat_ev, y_ev):
    """Probabilidad = impago de la categoría del buró en el entrenamiento.
    Máximo aprobable = categorías completas, de menor a mayor riesgo, mientras el impago sea < LIMITE."""
    tasas = pd.Series(y_tr.to_numpy()).groupby(cat_tr.to_numpy()).mean()
    p = cat_ev.map(tasas).fillna(y_tr.mean()).to_numpy()
    n, malos, n_max, puntos = 0, 0, 0, []
    for cat in tasas.sort_values().index:
        m = (cat_ev == cat).to_numpy()
        n, malos = n + m.sum(), malos + y_ev.to_numpy()[m].sum()
        puntos.append((cat, n / len(y_ev), malos / n))
        if malos / n < LIMITE:
            n_max = n
    return p, n_max, puntos


# ---------------------------------------------------------------------------
def main(dir_datos, dir_limpios, dir_salida):
    os.makedirs(os.path.join(dir_salida, "tablas"), exist_ok=True)
    os.makedirs(os.path.join(dir_salida, "graficos"), exist_ok=True)
    tabla = lambda nombre: os.path.join(dir_salida, "tablas", nombre)        # noqa: E731
    grafico = lambda nombre: os.path.join(dir_salida, "graficos", nombre)    # noqa: E731

    # ------------------------------------------------------------------ Paso 0
    titulo("Paso 0 · Carga de datos limpios (versión 1) y partición por tiempo")
    tipos = {"BRANCH_ID": str, "MANUFACTURER_ID": str}   # deben ser texto, como al ajustar la limpieza
    train = pd.read_csv(os.path.join(dir_limpios, "train_limpio.csv"), dtype=tipos)
    test = pd.read_csv(os.path.join(dir_limpios, "test_limpio.csv"), dtype=tipos)

    # SUPPLIER_ID no quedó en la versión 1; se trae del original solo para probar si aporta.
    def proveedor(archivo):
        crudo = prep.estandarizar_estructura(pd.read_csv(os.path.join(dir_datos, archivo)))
        return crudo[["UNIQUEID", "SUPPLIER_ID"]].astype({"SUPPLIER_ID": str})
    train = train.merge(proveedor("train.csv"), on="UNIQUEID", how="left", validate="1:1")
    test = test.merge(proveedor("test.csv"), on="UNIQUEID", how="left", validate="1:1")
    assert train["SUPPLIER_ID"].notna().all() and test["SUPPLIER_ID"].notna().all()

    y = train["LOAN_DEFAULT"]
    ago, sep, octu = (train["MES"] == 8), (train["MES"] == 9), (train["MES"] == 10)
    ago_sep = ago | sep
    for nombre, m in [("Agosto (ajuste-entrenamiento)", ago), ("Septiembre (ajuste-evaluación)", sep),
                      ("Ago+Sep (prueba-entrenamiento)", ago_sep), ("Octubre (prueba final)", octu)]:
        print(f"{nombre:34s} {m.sum():>8,} créditos · impago {y[m].mean():.2%}")
    print(f"{'Noviembre (predicción)':34s} {len(test):>8,} créditos · sin etiqueta")
    X_ago, y_ago = train[ago], y[ago]
    X_sep, y_sep = train[sep], y[sep]
    X_as, y_as = train[ago_sep], y[ago_sep]
    X_oct, y_oct = train[octu], y[octu]
    monto_oct = X_oct["DISBURSED_AMOUNT"]

    # ------------------------------------------------------------------ Paso 1
    titulo("Paso 1 · Referencias en octubre (CML1): clasificador trivial y regla del buró")
    resultados = {}
    p_trivial = np.full(len(y_oct), y_as.mean())
    resultados["Trivial"] = metricas(y_oct, p_trivial, monto_oct)
    p_buro, n_buro, puntos_buro = regla_buro(X_as["CNS_GRUPO"], y_as, X_oct["CNS_GRUPO"], y_oct)
    resultados["Regla buró"] = metricas(y_oct, p_buro, monto_oct)
    resultados["Regla buró"]["Max_aprobados_CN2"] = n_buro          # solo categorías completas
    resultados["Regla buró"]["Max_aprob_pct_CN2"] = n_buro / len(y_oct)
    print(pd.DataFrame(resultados).T[["AUC", "KS", "Max_aprobados_CN2", "Max_aprob_pct_CN2"]].round(4).to_string())

    # ------------------------------------------------------------------ Paso 2
    titulo("Paso 2 · Ingeniería de características (ajuste: agosto -> septiembre)")
    print("Variables base (Fase 2):", len(prep.VARIABLES_MODELO), "| nuevas candidatas:", ing.NUEVAS,
          "| SUPPLIER_ID agrupado (>=100 créditos, máx. 200)")
    filas = []
    for nuevas, prov in itertools.product([False, True], [False, True]):
        fila = {"nuevas": nuevas, "SUPPLIER_ID": prov}
        for tipo, params in [("logistica", {"C": 1.0}), ("boosting", {})]:
            _, p = probar(tipo, X_ago, y_ago, X_sep, nuevas, prov, **params)
            fila[f"AUC_sep_{tipo}"] = roc_auc_score(y_sep, p)
        fila["AUC_promedio"] = (fila["AUC_sep_logistica"] + fila["AUC_sep_boosting"]) / 2
        filas.append(fila)
        print(f"  nuevas={nuevas!s:5} SUPPLIER_ID={prov!s:5} -> AUC sep logística {fila['AUC_sep_logistica']:.4f}"
              f" · boosting {fila['AUC_sep_boosting']:.4f}", flush=True)
    variantes = pd.DataFrame(filas)
    variantes.to_csv(tabla("ingenieria_caracteristicas.csv"), index=False)
    base = variantes.iloc[0]
    mejor = variantes.loc[variantes["AUC_promedio"].idxmax()]
    if mejor["AUC_promedio"] - base["AUC_promedio"] < MEJORA_MINIMA:
        mejor = base
    NUEVAS, PROV = bool(mejor["nuevas"]), bool(mejor["SUPPLIER_ID"])
    entrada_final = columnas(NUEVAS, PROV)[0]
    print(f"\nDecisión: variables nuevas = {NUEVAS} · SUPPLIER_ID = {PROV} "
          f"(se acepta un cambio solo si el AUC promedio sube >= {MEJORA_MINIMA})")
    print("Variables de entrada del modelo final:", len(entrada_final), entrada_final,
          "+ nuevas: " + str(ing.NUEVAS) if NUEVAS else "")

    woe_iv = ing.WoE(NUM_LOG + columnas(NUEVAS, PROV)[1], columnas(NUEVAS, PROV)[2])
    woe_iv.fit(ing.AgregarVariables(NUEVAS).transform(X_as), y_as)
    iv = pd.Series(woe_iv.iv_, name="IV").sort_values(ascending=False)
    iv.to_csv(tabla("valor_informacion_ago_sep.csv"))
    print("\nValor de información (IV) con agosto-septiembre:\n", iv.round(4).to_string())

    # ------------------------------------------------------------------ Paso 3
    titulo("Paso 3 · Hiperparámetros (ajuste: agosto -> septiembre, métrica AUC)")
    rejillas = {
        "logistica": [{"C": c, "l1_ratio": 0.0} for c in (0.01, 0.1, 1.0)]
                     + [{"C": c, "l1_ratio": 1.0, "solver": "liblinear"} for c in (0.01, 0.1, 1.0)],
        "logistica_woe": [{"C": c, "l1_ratio": 0.0} for c in (0.01, 0.1, 1.0)],
        "boosting": [dict(learning_rate=lr, max_leaf_nodes=h, min_samples_leaf=m, l2_regularization=l2)
                     for lr, h, m, l2 in itertools.product((0.05, 0.1), (15, 31), (100, 400), (0.0, 1.0))],
        "bosque": [dict(min_samples_leaf=m, max_features=f) for m in (50, 200) for f in ("sqrt", 0.3)],
    }
    busqueda, mejores = [], {}
    for tipo, rejilla in rejillas.items():
        for params in rejilla:
            t0 = time.perf_counter()
            _, p = probar(tipo, X_ago, y_ago, X_sep, NUEVAS, PROV, **params)
            auc = roc_auc_score(y_sep, p)
            busqueda.append({"modelo": tipo, "params": json.dumps(params), "AUC_sep": auc})
            print(f"  {tipo:14s} {json.dumps(params):95s} AUC {auc:.4f}  ({time.perf_counter()-t0:.0f} s)",
                  flush=True)
        mejor_fila = max((b for b in busqueda if b["modelo"] == tipo), key=lambda b: b["AUC_sep"])
        mejores[tipo] = json.loads(mejor_fila["params"])
    pd.DataFrame(busqueda).to_csv(tabla("busqueda_hiperparametros.csv"), index=False)
    auc_tipo = pd.DataFrame(busqueda).groupby("modelo")["AUC_sep"].max()
    RIESGO = "logistica_woe" if auc_tipo["logistica_woe"] > auc_tipo["logistica"] else "logistica"
    print("\nMejores hiperparámetros:", json.dumps(mejores, indent=1))
    print(f"Variante de regresión logística para Riesgo: {RIESGO} "
          f"(AUC sep one-hot {auc_tipo['logistica']:.4f} vs WoE {auc_tipo['logistica_woe']:.4f})")

    # ------------------------------------------------------------------ Paso 4
    titulo("Paso 4 · Prueba final: entrena ago+sep, evalúa en OCTUBRE (CML1, CML2)")
    modelos_oct, probs = {}, {"Regla buró": p_buro}
    nombres = {"logistica": "Logística one-hot", "logistica_woe": "Logística WoE",
               "boosting": "Gradient boosting", "bosque": "Random Forest"}
    for tipo, nombre in nombres.items():
        modelos_oct[tipo], probs[nombre] = probar(tipo, X_as, y_as, X_oct, NUEVAS, PROV, **mejores[tipo])
        resultados[nombre] = metricas(y_oct, probs[nombre], monto_oct)
        print(f"  {nombre:18s} AUC {resultados[nombre]['AUC']:.4f}", flush=True)

    # Calibración (CML2): sigmoide con validación cruzada de 3 pliegues sobre ago+sep
    for tipo in ["boosting", RIESGO]:
        nombre = nombres[tipo] + " calibrada"
        base_modelo, entrada = construir(tipo, NUEVAS, PROV, **mejores[tipo])
        cal = CalibratedClassifierCV(base_modelo, method="sigmoid", cv=3).fit(X_as[entrada], y_as)
        probs[nombre] = cal.predict_proba(X_oct[entrada])[:, 1]
        resultados[nombre] = metricas(y_oct, probs[nombre], monto_oct)

    comparacion = pd.DataFrame(resultados).T
    comparacion.to_csv(tabla("comparacion_modelos_octubre.csv"))
    print("\n", comparacion.drop(columns=["Impago_real"]).round(4).to_string())

    # ------------------------------------------------------------------ Paso 5
    titulo("Paso 5 · Punto de corte (CML3, CN1, CN2, CE1) en octubre")
    tasas = np.round(np.arange(0.50, 1.0001, 0.01), 2)
    curvas = {}
    for nombre in ["Gradient boosting", nombres[RIESGO], "Random Forest"]:
        p = probs[nombre]
        curvas[nombre] = [y_oct.to_numpy()[aprobados_al(p, t)].mean() for t in tasas]
    curva = pd.DataFrame(curvas, index=tasas)
    curva.index.name = "tasa_aprobacion"
    curva.to_csv(tabla("curva_aprobacion_impago_octubre.csv"))
    print("Impago de los aprobados según % aprobado:\n",
          (curva.loc[[0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 1.00]] * 100).round(2).to_string())

    zonas = {}
    for nombre, clave in [("Gradient boosting", "ventas"), (nombres[RIESGO], "riesgo")]:
        p = probs[nombre]
        r = resultados[nombre]
        cumple = r["Cumple_CN2"]
        print(f"\n{nombre}: aprobando {APROBACION:.0%} -> impago aprobados "
              f"{r['Impago_aprobados_90']:.2%} (meta < {LIMITE:.2%}) -> CN1 y CN2 {'SE CUMPLEN' if cumple else 'NO se cumplen a la vez'}")
        print(f"   impagos evitados {r['Impagos_evitados']:,} · buenos pagadores rechazados {r['Buenos_rechazados']:,}"
              f" · monto evitado {r['Monto_evitado_CE1']:,.0f}")
        print(f"   máximo aprobable con impago < {LIMITE:.2%}: {r['Max_aprobados_CN2']:,} "
              f"({r['Max_aprob_pct_CN2']:.2%}) frente a {n_buro:,} de la regla del buró "
              f"(+{r['Max_aprobados_CN2'] - n_buro:,} créditos)")
        zonas[clave] = {"corte_revisar": float(np.quantile(p, APROBACION - FRANJA_REVISION)),
                        "corte_rechazar": float(np.quantile(p, APROBACION))}
    if not resultados["Gradient boosting"]["Cumple_CN2"]:
        print("\nNingún corte al 90 % cumple CN2 con boosting: revisar la curva y decidir con las gerencias (R2).")

    # PE1 Ventas: sucursales y fabricantes que concentran buenos pagadores (octubre)
    aprob_gb = aprobados_al(probs["Gradient boosting"], APROBACION)
    canal = X_oct.assign(aprobado=aprob_gb, impago=y_oct.to_numpy(), p_pago=1 - probs["Gradient boosting"])
    for col in ["BRANCH_ID", "MANUFACTURER_ID"]:
        t = canal.groupby(col).agg(creditos=("impago", "size"), impago_real=("impago", "mean"),
                                   pct_aprobado=("aprobado", "mean"), prob_pago_media=("p_pago", "mean"))
        t = t[t["creditos"] >= 200].sort_values("prob_pago_media", ascending=False)
        t.to_csv(tabla(f"ventas_por_{col.lower()}_octubre.csv"))
        print(f"\nVentas · {col} (>= 200 créditos), 5 con mayor probabilidad de pago:\n",
              t.head(5).round(4).to_string())

    # ------------------------------------------------------------------ Paso 6
    titulo("Paso 6 · Explicabilidad")
    lr = modelos_oct[RIESGO]
    nombres_lr = lr[:-1].transform(X_oct[entrada_final].head(5)).columns
    coef = pd.DataFrame({"variable": nombres_lr, "coeficiente": lr.named_steps["modelo"].coef_[0]})
    coef["odds_ratio"] = np.exp(coef["coeficiente"])
    coef = coef.reindex(coef["coeficiente"].abs().sort_values(ascending=False).index)
    coef.to_csv(tabla("coeficientes_logistica.csv"), index=False)
    print("Regresión logística (Riesgo), 15 coeficientes de mayor magnitud:\n",
          coef.head(15).round(4).to_string(index=False))

    muestra = X_oct.sample(n=min(20000, len(X_oct)), random_state=SEMILLA)
    imp = permutation_importance(modelos_oct["boosting"], muestra[entrada_final], y_oct.loc[muestra.index],
                                 scoring="roc_auc", n_repeats=5, random_state=SEMILLA, n_jobs=1)
    importancia = pd.DataFrame({"variable": entrada_final, "caida_AUC": imp.importances_mean,
                                "desv": imp.importances_std}).sort_values("caida_AUC", ascending=False)
    importancia.to_csv(tabla("importancia_boosting.csv"), index=False)
    print("\nGradient boosting (Ventas), importancia por permutación (caída del AUC):\n",
          importancia.round(4).to_string(index=False))

    # ------------------------------------------------------------------ Paso 7
    titulo("Paso 7 · Desempeño por segmento (equidad, R11)")
    segmentos = X_oct.assign(
        GRUPO_EDAD=pd.cut(X_oct["EDAD"], [0, 25, 30, 35, 45, 120], labels=["18-25", "26-30", "31-35", "36-45", "46+"]))
    filas = []
    for nombre in ["Gradient boosting", nombres[RIESGO]]:
        aprob = aprobados_al(probs[nombre], APROBACION)
        for col in ["GRUPO_EDAD", "EMPLOYMENT_TYPE"]:
            for g, idx in segmentos.groupby(col, observed=True).indices.items():
                yy, pp, aa = y_oct.to_numpy()[idx], probs[nombre][idx], aprob[idx]
                filas.append({"modelo": nombre, "segmento": col, "grupo": g, "creditos": len(idx),
                              "impago_real": yy.mean(),
                              "AUC": roc_auc_score(yy, pp) if len(set(yy)) > 1 else np.nan,
                              "pct_aprobado": aa.mean(), "impago_aprobados": yy[aa].mean()})
    equidad = pd.DataFrame(filas)
    equidad.to_csv(tabla("desempeno_por_segmento.csv"), index=False)
    print(equidad.round(4).to_string(index=False))

    # ------------------------------------------------------------------ Paso 8
    titulo("Paso 8 · Gráficos")
    fig, ax = plt.subplots(figsize=(6, 5))
    for nombre in ["Gradient boosting", nombres[RIESGO], "Random Forest", "Regla buró"]:
        fpr, tpr, _ = roc_curve(y_oct, probs[nombre])
        ax.plot(fpr, tpr, label=f"{nombre} (AUC {resultados[nombre]['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Trivial (AUC 0,5)")
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="Curva ROC · octubre")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(grafico("roc_octubre.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for nombre in curva.columns:
        ax.plot(curva.index * 100, curva[nombre] * 100, label=nombre)
    ax.plot([x * 100 for _, x, _ in puntos_buro], [z * 100 for _, _, z in puntos_buro], "o--", color="gray",
            label="Regla buró (categorías completas)")
    ax.axhline(LIMITE * 100, color="red", ls=":", label=f"Meta CN2 ({LIMITE:.2%})")
    ax.axvline(APROBACION * 100, color="green", ls=":", label=f"Meta CN1 ({APROBACION:.0%})")
    ax.set(xlabel="% de solicitudes aprobadas", ylabel="% de impago de los aprobados",
           title="Aprobación vs. impago · octubre", xlim=(30, 100))
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(grafico("curva_aprobacion_impago.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for nombre in ["Gradient boosting", "Gradient boosting calibrada", nombres[RIESGO], nombres[RIESGO] + " calibrada"]:
        frac, media = calibration_curve(y_oct, probs[nombre], n_bins=10, strategy="quantile")
        ax.plot(media, frac, "o-", label=f"{nombre} (Brier {resultados[nombre]['Brier']:.4f})")
    ax.plot([0, 0.6], [0, 0.6], "k--", lw=1)
    ax.set(xlabel="Probabilidad predicha", ylabel="Impago observado", title="Calibración · octubre")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(grafico("calibracion_octubre.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    imp_g = importancia.sort_values("caida_AUC")
    ax.barh(imp_g["variable"], imp_g["caida_AUC"], xerr=imp_g["desv"])
    ax.set(xlabel="Caída del AUC al permutar", title="Importancia de variables · gradient boosting")
    fig.tight_layout(); fig.savefig(grafico("importancia_boosting.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    top = coef.head(20).iloc[::-1]
    ax.barh(top["variable"], top["coeficiente"], color=np.where(top["coeficiente"] > 0, "#c0392b", "#27ae60"))
    ax.set(xlabel="Coeficiente (positivo = más riesgo de impago)", title="Regresión logística · 20 coeficientes")
    fig.tight_layout(); fig.savefig(grafico("coeficientes_logistica.png"), dpi=150); plt.close(fig)
    print("Gráficos guardados en", os.path.join(dir_salida, "graficos"))

    # ------------------------------------------------------------------ Paso 9
    titulo("Paso 9 · Modelo final: reentrena ago-oct y predice noviembre")
    finales = {}
    for clave, tipo in [("ventas", "boosting"), ("riesgo", RIESGO)]:
        modelo, entrada = construir(tipo, NUEVAS, PROV, **mejores[tipo])
        modelo.fit(train[entrada], y)
        p_nov = modelo.predict_proba(test[entrada])[:, 1]
        finales[clave] = {"modelo": modelo, "tipo": tipo, "p": p_nov,
                          "corte_revisar": float(np.quantile(p_nov, APROBACION - FRANJA_REVISION)),
                          "corte_rechazar": float(np.quantile(p_nov, APROBACION))}
    joblib.dump(finales["ventas"]["modelo"], os.path.join(dir_salida, "modelo_ventas_boosting.pkl"))
    joblib.dump(finales["riesgo"]["modelo"], os.path.join(dir_salida, "modelo_riesgo_logistica.pkl"))

    def decision(p, f):
        return np.select([p >= f["corte_rechazar"], p >= f["corte_revisar"]], ["Rechazar", "Revisar"], "Aprobar")

    pred = pd.DataFrame({"UNIQUEID": test["UNIQUEID"]})
    for clave in ["ventas", "riesgo"]:
        f = finales[clave]
        pred[f"P_IMPAGO_{clave.upper()}"] = f["p"].round(5)
        pred[f"P_PAGO_{clave.upper()}"] = (1 - f["p"]).round(5)
        pred[f"DECISION_{clave.upper()}"] = decision(f["p"], f)
    pred.to_csv(os.path.join(dir_salida, "predicciones_noviembre.csv"), index=False)
    print(pred["DECISION_VENTAS"].value_counts(normalize=True).round(4).to_string())

    config = {
        "variables_entrada": entrada_final,
        "variables_nuevas": ing.NUEVAS if NUEVAS else [],
        "usa_supplier_id": PROV,
        "tipos_texto": ["BRANCH_ID", "MANUFACTURER_ID"] + (["SUPPLIER_ID"] if PROV else []),
        "limite_impago": LIMITE, "tasa_aprobacion": APROBACION, "franja_revision": FRANJA_REVISION,
        "modelos": {clave: {"archivo": f"modelo_{clave}_{'boosting' if clave == 'ventas' else 'logistica'}.pkl",
                            "tipo": finales[clave]["tipo"], "hiperparametros": mejores[finales[clave]["tipo"]],
                            "corte_revisar": finales[clave]["corte_revisar"],
                            "corte_rechazar": finales[clave]["corte_rechazar"]} for clave in finales},
        "cortes_prueba_octubre": zonas,
        "metricas_octubre": {k: {m: (float(v) if isinstance(v, (int, float, np.floating, np.integer, bool)) else v)
                                 for m, v in r.items()} for k, r in resultados.items()},
        "uso_en_tablero": ("import preparacion, ingenieria; df = preparacion.preparar(datos, fecha_consulta=hoy); "
                           "p = modelo.predict_proba(df[variables_entrada])[:, 1]; "
                           "Rechazar si p >= corte_rechazar, Revisar si p >= corte_revisar, si no Aprobar"),
    }
    with open(os.path.join(dir_salida, "config_modelos.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)

    titulo("Resumen")
    resumen = comparacion[["AUC", "PR_AUC", "KS", "Brier", "Impago_aprobados_90", "Cumple_CN2",
                           "Monto_evitado_CE1", "Max_aprobados_CN2"]]
    print(resumen.round(4).to_string())
    print(f"\nTiempo total: {(time.perf_counter() - INICIO) / 60:.1f} min · archivos en {os.path.abspath(dir_salida)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modelado - créditos vehiculares (Fase 3 CRISP-ML(Q))")
    parser.add_argument("--datos", default=RAIZ, help="Carpeta con train.csv y test.csv originales")
    parser.add_argument("--limpios", default=os.path.join(RAIZ, "limpieza_datos", "datos_v1"),
                        help="Carpeta datos_v1 generada por limpieza_datos.py")
    parser.add_argument("--salida", default=os.path.join(AQUI, "salidas"), help="Carpeta de salida")
    args = parser.parse_args()
    main(args.datos, args.limpios, args.salida)
