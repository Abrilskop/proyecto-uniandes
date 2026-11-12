"""
figura_reporte.py
Figura 1 del reporte: (a) el problema (la regla del buró no distingue a la mitad de los clientes) y
(b) la solución (el modelo aprueba más con menos impago). También imprime la Tabla 1.

Uso (desde la carpeta reporte/):  python figura_reporte.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
LIMITE = 0.2171
CORTE = 0.88
ORDEN = ["Muy bajo", "Bajo", "Sin puntaje", "Sin historial", "Medio", "Alto", "Muy alto"]
COLOR = {"Gradient boosting": "#C0392B", "Regresión logística": "#1F77B4", "Random Forest": "#2CA02C",
         "Regla del buró": "#7F7F7F"}

train = pd.read_csv(os.path.join(RAIZ, "limpieza_datos", "datos_v1", "train_limpio.csv"),
                    usecols=["CNS_GRUPO", "LOAN_DEFAULT"])
pred = pd.read_csv(os.path.join(RAIZ, "modelado", "salidas", "predicciones_octubre.csv"),
                   usecols=["modelo", "y_true", "proba"])

fig, ax = plt.subplots(1, 2, figsize=(10, 3.1), gridspec_kw={"width_ratios": [1, 1.25]})

# (a) El problema
buro = train.groupby("CNS_GRUPO")["LOAN_DEFAULT"].agg(["mean", "size"]).reindex(ORDEN)
colores = ["#95A5A6" if c == "Sin historial" else ("#C0392B" if m > LIMITE else "#27AE60")
           for c, m in buro["mean"].items()]
ax[0].barh(buro.index, buro["mean"] * 100, color=colores)
ax[0].invert_yaxis()
ax[0].axvline(LIMITE * 100, color="red", ls="--", lw=1)
ax[0].text(LIMITE * 100 + 0.4, -0.62, "meta 21,71 %", color="red", fontsize=7.5)
for i, (m, n) in enumerate(zip(buro["mean"], buro["size"])):
    ax[0].text(m * 100 + 0.3, i, f"{m*100:.1f} %  ({n/len(train)*100:.0f} % de los clientes)".replace(".", ","),
               va="center", fontsize=7.5)
ax[0].set_ylim(6.6, -0.9)
ax[0].set_xlim(0, 52)
ax[0].set_xlabel("% de impago en cada categoría del buró", fontsize=8)
ax[0].set_title("(a) La regla del buró ordena el riesgo, pero\nla mitad de los clientes no tiene historial (gris)",
                fontsize=9, loc="left")
ax[0].tick_params(labelsize=8)

# (b) La solución
tasas = np.round(np.arange(0.20, 1.0001, 0.01), 2)
# la regla del buró solo puede aprobar categorías completas: un punto por categoría
rb = (pred[pred["modelo"] == "Regla del buró"].groupby("proba")["y_true"].agg(["size", "sum"])
      .sort_index().cumsum())
n_oct = rb["size"].iloc[-1]
ax[1].plot(rb["size"] / n_oct * 100, rb["sum"] / rb["size"] * 100, "o--", color=COLOR["Regla del buró"],
           lw=1.3, ms=4, label="Regla del buró (categorías completas)")
for modelo in ["Regresión logística", "Gradient boosting"]:
    d = pred[pred["modelo"] == modelo].sort_values("proba", kind="stable")
    y = d["y_true"].to_numpy()
    acum = np.cumsum(y)
    curva = [acum[int(round(t * len(y))) - 1] / int(round(t * len(y))) for t in tasas]
    ax[1].plot(tasas * 100, np.array(curva) * 100, color=COLOR[modelo], lw=2, label=modelo)
gb = pred[pred["modelo"] == "Gradient boosting"].sort_values("proba", kind="stable")["y_true"].to_numpy()
imp88 = gb[: int(round(CORTE * len(gb)))].mean() * 100
ax[1].axhline(LIMITE * 100, color="red", ls="--", lw=1)
ax[1].scatter([CORTE * 100], [imp88], color="black", zorder=5, s=25)
ax[1].annotate(f"Corte recomendado: aprobar el 88 %\nimpago {imp88:.1f} % (98 % de confianza)".replace(".", ","),
               (CORTE * 100, imp88), xytext=(52, 23.2), fontsize=7.5,
               arrowprops=dict(arrowstyle="->", lw=0.8))
ax[1].text(21, LIMITE * 100 - 0.5, "meta 21,71 %", color="red", fontsize=7.5)
ax[1].annotate("aprobar a todos:\n23,5 %", (100, 23.51), xytext=(91, 24.6), fontsize=7, ha="center",
               arrowprops=dict(arrowstyle="->", lw=0.6))
ax[1].set_xlim(20, 101)
ax[1].set_ylim(13, 25.5)
ax[1].set_xlabel("% de solicitudes aprobadas (octubre)", fontsize=8)
ax[1].set_ylabel("% de impago de los aprobados", fontsize=8)
ax[1].set_title("(b) Con el modelo se aprueban muchas más solicitudes\ncon el mismo impago que con la regla del buró",
                fontsize=9, loc="left")
ax[1].legend(fontsize=7.5, loc="lower right")
ax[1].tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(AQUI, "figura1.png"), dpi=220)
print("figura1.png guardada")

# Tabla 1
filas = []
for modelo in ["Regla del buró", "Random Forest", "Regresión logística", "Gradient boosting"]:
    d = pred[pred["modelo"] == modelo]
    y, p = d["y_true"].to_numpy(), d["proba"].to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    orden = np.argsort(p, kind="stable")
    tasa_acum = np.cumsum(y[orden]) / np.arange(1, len(y) + 1)
    ok = np.where(tasa_acum < LIMITE)[0]
    maximo = int(ok.max() + 1) if len(ok) else 0
    impago_88 = y[orden[: int(round(CORTE * len(y)))]].mean() * 100
    if modelo == "Regla del buró":   # la regla solo aprueba categorías completas del buró
        g = d.groupby("proba")["y_true"].agg(["size", "sum"]).sort_index().cumsum()
        maximo = int(g.loc[g["sum"] / g["size"] < LIMITE, "size"].max())
        impago_88 = np.nan          # no puede aprobar exactamente el 88 %
    filas.append({"modelo": modelo, "AUC": roc_auc_score(y, p), "KS": (tpr - fpr).max(),
                  "impago_88": impago_88, "max_aprobables": maximo})
print(pd.DataFrame(filas).round(3).to_string(index=False))
