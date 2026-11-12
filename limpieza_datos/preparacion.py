"""
preparacion.py
Funciones y transformadores de la Fase 2 (CRISP-ML(Q)) para el proyecto de créditos vehiculares.

Lo usan:
  - limpieza_datos.py  (genera los datos limpios y los preprocesadores .pkl)
  - el tablero en Dash (para preparar los datos de un cliente igual que en el entrenamiento)

IMPORTANTE: el tablero debe hacer `import preparacion` (o `from preparacion import *`)
antes de cargar los archivos .pkl, porque contienen las clases AgrupadorRaros y TopeLog.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Parámetros del negocio (decididos en la Fase 1)
# ---------------------------------------------------------------------------
FORMATO_FECHA = "%d-%m-%Y"
DIAS_ANTES_DESEMBOLSO = 2      # fecha de solicitud aproximada = desembolso - 2 días
LIMITE_IMPAGO = 0.2171         # 21,71 % observado en train.csv
MESES_ENTRENAMIENTO = [8, 9]   # agosto y septiembre de 2018
MES_VALIDACION = 10            # octubre de 2018
MIN_REGISTROS_CATEGORIA = 100  # mínimo de registros para conservar una sucursal o fabricante
USAR_AADHAR = False            # variable sensible: el equipo decide si se justifica

# Variables del modelo (15)
NUM_LOG = ["DISBURSED_AMOUNT", "PRI_SANCTIONED_AMOUNT"]
NUM = ["LTV", "PRI_NO_OF_ACCTS", "PRI_ACTIVE_ACCTS", "PRI_OVERDUE_ACCTS", "HISTORIAL_CRED_M",
       "NO_OF_INQUIRIES", "DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS", "NEW_ACCTS_IN_LAST_SIX_MONTHS", "EDAD"]
if USAR_AADHAR:
    NUM = NUM + ["AADHAR_FLAG"]
CAT = ["CNS_GRUPO", "EMPLOYMENT_TYPE", "BRANCH_ID", "MANUFACTURER_ID"]
VARIABLES_MODELO = NUM_LOG + NUM + CAT

TRADUCCION_BURO = {"Very Low Risk": "Muy bajo", "Low Risk": "Bajo", "Medium Risk": "Medio",
                   "High Risk": "Alto", "Very High Risk": "Muy alto"}

COLUMNAS_ID = ["BRANCH_ID", "MANUFACTURER_ID", "SUPPLIER_ID", "STATE_ID",
               "EMPLOYEE_CODE_ID", "CURRENT_PINCODE_ID"]


# ---------------------------------------------------------------------------
# Paso 1: estructura
# ---------------------------------------------------------------------------
def estandarizar_estructura(df):
    """Nombres de columnas en mayúsculas y sin puntos (el diccionario usa p. ej. Date.of.Birth)."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.upper().str.replace(".", "_", regex=False)
    return df


# ---------------------------------------------------------------------------
# Paso 3: fechas
# ---------------------------------------------------------------------------
def edad_cumplida(ref, nac):
    """Edad en años cumplidos a la fecha de referencia."""
    no_cumplio = (ref.dt.month < nac.dt.month) | ((ref.dt.month == nac.dt.month) & (ref.dt.day < nac.dt.day))
    return ref.dt.year - nac.dt.year - no_cumplio.astype(int)


def tratar_fechas(df, fecha_consulta=None):
    """
    Datos históricos (fecha_consulta=None): edad a (desembolso - 2 días) y MES del desembolso.
    Tablero: edad a la fecha de la consulta.
    DISBURSAL_DATE no entra al modelo: solo hay 3 meses y test.csv es un mes nuevo.
    """
    df = df.copy()
    nac = pd.to_datetime(df["DATE_OF_BIRTH"], format=FORMATO_FECHA, errors="coerce")
    if fecha_consulta is None:
        desembolso = pd.to_datetime(df["DISBURSAL_DATE"], format=FORMATO_FECHA, errors="coerce")
        ref = desembolso - pd.Timedelta(days=DIAS_ANTES_DESEMBOLSO)
        df["MES"] = desembolso.dt.month
    else:
        ref = pd.Series(pd.Timestamp(fecha_consulta), index=df.index)
    df["EDAD"] = edad_cumplida(ref, nac)
    return df


# ---------------------------------------------------------------------------
# Paso 4: antigüedades de texto a meses
# ---------------------------------------------------------------------------
def texto_a_meses(s):
    """'1yrs 11mon' -> 23."""
    m = s.astype(str).str.extract(r"(\d+)\s*yrs\s*(\d+)\s*mon")
    return m[0].astype(float) * 12 + m[1].astype(float)


def tratar_antiguedades(df):
    df = df.copy()
    df["HISTORIAL_CRED_M"] = texto_a_meses(df["CREDIT_HISTORY_LENGTH"])
    df["ANTIG_PROM_CUENTAS_M"] = texto_a_meses(df["AVERAGE_ACCT_AGE"])
    return df


# ---------------------------------------------------------------------------
# Paso 5: categóricas
# ---------------------------------------------------------------------------
def tratar_categoricas(df):
    df = df.copy()
    df["EMPLOYMENT_TYPE"] = df["EMPLOYMENT_TYPE"].fillna("Sin dato")
    desc = df["PERFORM_CNS_SCORE_DESCRIPTION"].astype(str)
    grupo = desc.str.split("-", n=1).str[1].map(TRADUCCION_BURO)
    grupo = grupo.mask(desc.str.startswith("No Bureau"), "Sin historial")
    grupo = grupo.mask(desc.str.startswith("Not Scored"), "Sin puntaje")
    df["CNS_GRUPO"] = grupo
    for col in COLUMNAS_ID:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def preparar(df, fecha_consulta=None):
    """Pasos 1 a 5. La misma función se usa en el entrenamiento y en el tablero."""
    df = estandarizar_estructura(df)
    df = tratar_fechas(df, fecha_consulta)
    df = tratar_antiguedades(df)
    df = tratar_categoricas(df)
    return df


# ---------------------------------------------------------------------------
# Paso 9: transformadores ajustados solo con el entrenamiento
# ---------------------------------------------------------------------------
class AgrupadorRaros(BaseEstimator, TransformerMixin):
    """Agrupa en 'OTRA' las categorías poco frecuentes en el entrenamiento y las que no se vieron al entrenar."""

    def __init__(self, columnas, min_registros=100, etiqueta="OTRA"):
        self.columnas = columnas
        self.min_registros = min_registros
        self.etiqueta = etiqueta

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        self.frecuentes_ = {}
        for c in self.columnas:
            conteo = X[c].value_counts()
            self.frecuentes_[c] = set(conteo[conteo >= self.min_registros].index)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columnas:
            X[c] = X[c].where(X[c].isin(self.frecuentes_[c]), self.etiqueta)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class TopeLog(BaseEstimator, TransformerMixin):
    """Recorta a [0, percentil] calculado con el entrenamiento y aplica log(1 + x)."""

    def __init__(self, percentil=0.999):
        self.percentil = percentil

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        self.topes_ = X.quantile(self.percentil)
        return self

    def transform(self, X):
        return np.log1p(X.clip(lower=0, upper=self.topes_, axis=1))

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def crear_preprocesador(para="logistica", min_registros=MIN_REGISTROS_CATEGORIA):
    """
    para="logistica": tope+log, escalado y one-hot (PE2, Gerencia de Riesgo).
    para="boosting" : tope+log y codificación ordinal (PE1, Gerencia de Ventas).
                      En HistGradientBoostingClassifier, las columnas 'cat__*' son categóricas.
    """
    if para == "logistica":
        columnas = ColumnTransformer([
            ("log", Pipeline([("tope", TopeLog()), ("escala", StandardScaler())]), NUM_LOG),
            ("num", StandardScaler(), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ])
    elif para == "boosting":
        columnas = ColumnTransformer([
            ("log", TopeLog(), NUM_LOG),
            ("num", "passthrough", NUM),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
        ])
    else:
        raise ValueError("para debe ser 'logistica' o 'boosting'")
    return Pipeline([
        ("raros", AgrupadorRaros(["BRANCH_ID", "MANUFACTURER_ID"], min_registros)),
        ("columnas", columnas),
    ]).set_output(transform="pandas")
