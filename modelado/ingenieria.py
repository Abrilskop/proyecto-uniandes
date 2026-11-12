"""
ingenieria.py
Fase 3 (CRISP-ML(Q)): transformadores de ingeniería de características para el modelado.

Lo usan:
  - modelado.py (entrena y guarda los modelos)
  - el tablero en Dash: debe hacer `import preparacion` e `import ingenieria`
    ANTES de cargar los .pkl de los modelos, porque contienen estas clases.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Variables nuevas que crea AgregarVariables
NUEVAS = ["TIENE_MORA", "CONSULTO_BURO", "PROP_MORA"]


class AgregarVariables(BaseEstimator, TransformerMixin):
    """
    Crea variables a partir de las 15 de la Fase 2 (no usa la variable objetivo):
      TIENE_MORA    = 1 si tiene cuentas en mora o mora en los últimos 6 meses
      CONSULTO_BURO = 1 si tiene consultas al buró
      PROP_MORA     = cuentas en mora / cuentas principales (0 si no tiene cuentas)
    Con activar=False devuelve los datos sin cambios (sirve para comparar).
    """

    def __init__(self, activar=True):
        self.activar = activar

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.activar:
            X["TIENE_MORA"] = ((X["PRI_OVERDUE_ACCTS"] > 0)
                               | (X["DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS"] > 0)).astype(int)
            X["CONSULTO_BURO"] = (X["NO_OF_INQUIRIES"] > 0).astype(int)
            cuentas = X["PRI_NO_OF_ACCTS"]
            prop = X["PRI_OVERDUE_ACCTS"] / cuentas.where(cuentas > 0, 1)
            X["PROP_MORA"] = prop.where(cuentas > 0, 0.0).clip(0, 1)
        return X


class AgrupadorTop(BaseEstimator, TransformerMixin):
    """
    Agrupa en 'OTRA' las categorías con menos de `min_registros` en el entrenamiento,
    las que superan el máximo de `max_categorias` (límite de HistGradientBoosting: 255)
    y las que no se vieron al entrenar (sucursales o concesionarios nuevos).
    """

    def __init__(self, columnas, min_registros=100, max_categorias=200, etiqueta="OTRA"):
        self.columnas = columnas
        self.min_registros = min_registros
        self.max_categorias = max_categorias
        self.etiqueta = etiqueta

    def fit(self, X, y=None):
        self.frecuentes_ = {}
        for c in self.columnas:
            conteo = X[c].astype(str).value_counts()
            conteo = conteo[conteo >= self.min_registros]
            self.frecuentes_[c] = list(conteo.index[: self.max_categorias])
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columnas:
            valores = X[c].astype(str)
            X[c] = valores.where(valores.isin(self.frecuentes_[c]), self.etiqueta)
        return X


class WoE(BaseEstimator, TransformerMixin):
    """
    Peso de evidencia (Weight of Evidence), estándar de los scorecards de crédito (Siddiqi, 2017).
    - Numéricas: se cortan en `n_bins` cuantiles calculados con el entrenamiento.
    - Categóricas: cada categoría es un grupo.
    WoE = ln(% de malos del grupo / % de buenos del grupo): positivo = más riesgo.
    Grupos no vistos al entrenar reciben WoE = 0 (riesgo promedio).
    También calcula el valor de información (IV) de cada variable en `iv_`.
    """

    def __init__(self, columnas_num, columnas_cat, n_bins=10, suavizado=0.5):
        self.columnas_num = columnas_num
        self.columnas_cat = columnas_cat
        self.n_bins = n_bins
        self.suavizado = suavizado

    def _woe(self, grupos, y):
        t = pd.DataFrame({"g": grupos, "y": y}).groupby("g")["y"].agg(["sum", "count"])
        malos = t["sum"]
        buenos = t["count"] - t["sum"]
        k, a = len(t), self.suavizado
        p_malos = (malos + a) / (malos.sum() + a * k)
        p_buenos = (buenos + a) / (buenos.sum() + a * k)
        woe = np.log(p_malos / p_buenos)
        iv = float(((p_malos - p_buenos) * woe).sum())
        return woe.to_dict(), iv

    def _grupos_num(self, X, c):
        return np.searchsorted(self.cortes_[c], X[c].to_numpy(dtype=float), side="right")

    def fit(self, X, y):
        y = np.asarray(y)
        self.cortes_, self.mapas_, self.iv_ = {}, {}, {}
        cuantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        for c in self.columnas_num:
            self.cortes_[c] = np.unique(np.quantile(X[c].to_numpy(dtype=float), cuantiles))
            self.mapas_[c], self.iv_[c] = self._woe(self._grupos_num(X, c), y)
        for c in self.columnas_cat:
            self.mapas_[c], self.iv_[c] = self._woe(X[c].astype(str).to_numpy(), y)
        return self

    def transform(self, X):
        salida = {}
        for c in self.columnas_num:
            grupos = pd.Series(self._grupos_num(X, c), index=X.index)
            salida[c + "_WOE"] = grupos.map(self.mapas_[c]).fillna(0.0).astype(float)
        for c in self.columnas_cat:
            salida[c + "_WOE"] = X[c].astype(str).map(self.mapas_[c]).fillna(0.0).astype(float)
        return pd.DataFrame(salida, index=X.index)
