"""
limpieza_datos.py
Fase 2 (CRISP-ML(Q)): preparación y limpieza de datos - créditos vehiculares.

Uso (desde la carpeta donde están este archivo y preparacion.py):
    python limpieza_datos.py                         # train.csv y test.csv en la misma carpeta
    python limpieza_datos.py --datos "C:/ruta/a/los/csv"
    python limpieza_datos.py --datos ./data --salida ./datos_v1

Requisitos: pip install pandas numpy scikit-learn joblib

Salidas en la carpeta --salida (por defecto datos_v1/):
    train_limpio.csv, test_limpio.csv      datos limpios antes de escalar (15 variables + UNIQUEID, MES)
    prep_logistica.pkl, prep_boosting.pkl  preprocesadores ajustados solo con agosto-septiembre
    bitacora.csv                           filas y columnas después de cada paso
    reporte_calidad.csv                    diagnóstico de valores inválidos y extremos
    decisiones_variables.csv               decisión tomada para cada columna original
    comparacion_conjuntos.csv              entrenamiento vs validación vs prueba
"""
import argparse
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

import preparacion as prep

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
np.random.seed(42)

bitacora = []


def registrar(paso, df, nota, conjunto="train"):
    bitacora.append({"paso": paso, "conjunto": conjunto, "filas": df.shape[0],
                     "columnas": df.shape[1], "nota": nota})


def titulo(texto):
    print("\n" + "=" * 78 + f"\n{texto}\n" + "=" * 78)


def main(dir_datos, dir_salida):
    # ------------------------------------------------------------------ Paso 0
    titulo("Paso 0 · Carga (versión 0, archivos originales sin modificar)")
    train_raw = pd.read_csv(os.path.join(dir_datos, "train.csv"))
    test_raw = pd.read_csv(os.path.join(dir_datos, "test.csv"))
    registrar("0 Carga", train_raw, "train.csv original")
    registrar("0 Carga", test_raw, "test.csv original", "test")
    print("train:", train_raw.shape, "| test:", test_raw.shape)

    # ------------------------------------------------------------------ Paso 1
    titulo("Paso 1 · Estructura")
    train = prep.estandarizar_estructura(train_raw)
    test = prep.estandarizar_estructura(test_raw)
    diferencia = set(train.columns) ^ set(test.columns)
    print("Columnas que no están en ambos archivos:", diferencia)
    assert diferencia == {"LOAN_DEFAULT"}, "train y test deben diferir solo en LOAN_DEFAULT"
    registrar("1 Estructura", train, "Nombres estandarizados")

    # ------------------------------------------------------------------ Paso 2
    titulo("Paso 2 · Integridad")
    controles = {
        "UNIQUEID duplicados (train)": train["UNIQUEID"].duplicated().sum(),
        "UNIQUEID duplicados (test)": test["UNIQUEID"].duplicated().sum(),
        "Filas completamente duplicadas (train)": train.duplicated().sum(),
        "IDs presentes en train y en test": len(set(train["UNIQUEID"]) & set(test["UNIQUEID"])),
        "LOAN_DEFAULT distinto de 0/1 o vacío": (~train["LOAN_DEFAULT"].isin([0, 1])).sum(),
    }
    print(pd.Series(controles, name="cantidad").to_string())
    assert all(v == 0 for v in controles.values()), "Revisar integridad"
    vacios = pd.DataFrame({"train": train.isna().sum(), "test": test.isna().sum()})
    print("\nColumnas con vacíos:\n", vacios[(vacios.fillna(0) > 0).any(axis=1)].to_string())
    print(f"\nImpago en train: {train['LOAN_DEFAULT'].mean():.2%} "
          f"({train['LOAN_DEFAULT'].sum():,} de {len(train):,})")

    # ------------------------------------------------------------------ Paso 3
    titulo("Paso 3 · Fechas: EDAD a la solicitud (desembolso - 2 días) y MES")
    for nombre, d in [("train", train), ("test", test)]:
        for col in ["DATE_OF_BIRTH", "DISBURSAL_DATE"]:
            invalidas = pd.to_datetime(d[col], format=prep.FORMATO_FECHA, errors="coerce").isna().sum()
            print(f"{nombre} · {col}: fechas inválidas = {invalidas}")
            assert invalidas == 0, f"Fechas inválidas en {nombre}.{col}"
    train = prep.tratar_fechas(train)
    test = prep.tratar_fechas(test)
    for nombre, d in [("train", train), ("test", test)]:
        fd = pd.to_datetime(d["DISBURSAL_DATE"], format=prep.FORMATO_FECHA)
        print(f"{nombre}: desembolsos del {fd.min():%d-%m-%Y} al {fd.max():%d-%m-%Y} · "
              f"edad de {d['EDAD'].min()} a {d['EDAD'].max()}")
    assert train["EDAD"].between(18, 100).all(), "Edades fuera de rango en train"
    menores = test[test["EDAD"] < 18]
    print(f"test: registros con edad < 18 a la fecha de solicitud aproximada = {len(menores)} "
          "(no se eliminan; consulta al negocio)")
    if len(menores):
        print(menores[["UNIQUEID", "DATE_OF_BIRTH", "DISBURSAL_DATE", "EDAD"]].to_string(index=False))
    edad_desembolso = prep.edad_cumplida(
        pd.to_datetime(train["DISBURSAL_DATE"], format=prep.FORMATO_FECHA),
        pd.to_datetime(train["DATE_OF_BIRTH"], format=prep.FORMATO_FECHA))
    print("Registros cuya edad cambia por los 2 días:", int((edad_desembolso != train["EDAD"]).sum()))
    print("\nImpago por mes:\n", train.groupby("MES")["LOAN_DEFAULT"].agg(creditos="size", impago="mean")
          .assign(impago=lambda t: (t["impago"] * 100).round(2)).to_string())
    registrar("3 Fechas", train, "EDAD y MES creados")

    # ------------------------------------------------------------------ Paso 4
    titulo("Paso 4 · Antigüedades de texto a meses")
    train = prep.tratar_antiguedades(train)
    test = prep.tratar_antiguedades(test)
    assert train[["HISTORIAL_CRED_M", "ANTIG_PROM_CUENTAS_M"]].notna().all().all()
    con_historial = train["PRI_NO_OF_ACCTS"] > 0
    rho = train.loc[con_historial, ["HISTORIAL_CRED_M", "ANTIG_PROM_CUENTAS_M"]].corr(method="spearman").iloc[0, 1]
    print(f"Correlación (Spearman) entre clientes con historial: {rho:.2f} -> se conserva HISTORIAL_CRED_M")
    registrar("4 Antigüedades", train, "Texto convertido a meses")

    # ------------------------------------------------------------------ Paso 5
    titulo("Paso 5 · Categóricas: 'Sin dato' en empleo y CNS_GRUPO")
    train = prep.tratar_categoricas(train)
    test = prep.tratar_categoricas(test)
    assert train["CNS_GRUPO"].notna().all() and test["CNS_GRUPO"].notna().all()
    print(train.groupby("CNS_GRUPO")["LOAN_DEFAULT"].agg(creditos="size", impago="mean")
          .sort_values("impago").assign(impago=lambda t: (t["impago"] * 100).round(2)).to_string())
    registrar("5 Categóricas", train, "Sin dato en empleo, CNS_GRUPO, IDs como texto")

    columnas_control = ["EDAD", "HISTORIAL_CRED_M", "CNS_GRUPO"]
    assert prep.preparar(train_raw)[columnas_control].equals(train[columnas_control])
    print("preparar() reproduce los pasos 1 a 5")

    # ------------------------------------------------------------------ Paso 6
    titulo("Paso 6 · Valores inválidos y extremos (diagnóstico)")
    reporte = pd.DataFrame([
        ["PRI_CURRENT_BALANCE < 0", (train["PRI_CURRENT_BALANCE"] < 0).sum(), "Variable eliminada por redundancia"],
        ["SEC_CURRENT_BALANCE < 0", (train["SEC_CURRENT_BALANCE"] < 0).sum(), "Variable eliminada: sin aporte"],
        ["PRI_SANCTIONED_AMOUNT >= 1e9", (train["PRI_SANCTIONED_AMOUNT"] >= 1e9).sum(), "Tope percentil 99,9 + log(1+x)"],
        ["PRI_SANCTIONED_AMOUNT < 0", (train["PRI_SANCTIONED_AMOUNT"] < 0).sum(), "Se recorta a 0"],
        ["PRI_OVERDUE_ACCTS > PRI_ACTIVE_ACCTS", (train["PRI_OVERDUE_ACCTS"] > train["PRI_ACTIVE_ACCTS"]).sum(),
         "Se conserva: posibles cuentas cerradas en mora (consulta al negocio)"],
        ["LTV fuera de (0, 100]", (~train["LTV"].between(0.01, 100)).sum(), "Debe ser 0"],
        ["DISBURSED_AMOUNT <= 0", (train["DISBURSED_AMOUNT"] <= 0).sum(), "Debe ser 0"],
    ], columns=["caso", "registros", "decision"])
    print(reporte.to_string(index=False))

    # ------------------------------------------------------------------ Paso 7
    titulo("Paso 7 · Selección de variables")
    decisiones = {
        "Identificador (se guarda aparte)": ["UNIQUEID"],
        "Constante": ["MOBILENO_AVL_FLAG"],
        "Redundante": ["PRI_DISBURSED_AMOUNT", "PRI_CURRENT_BALANCE", "ANTIG_PROM_CUENTAS_M", "PERFORM_CNS_SCORE",
                       "PERFORM_CNS_SCORE_DESCRIPTION", "STATE_ID", "VOTERID_FLAG", "ASSET_COST"],
        "Sin aporte": ["SEC_NO_OF_ACCTS", "SEC_ACTIVE_ACCTS", "SEC_OVERDUE_ACCTS", "SEC_CURRENT_BALANCE",
                       "SEC_SANCTIONED_AMOUNT", "SEC_DISBURSED_AMOUNT", "SEC_INSTAL_AMT",
                       "PAN_FLAG", "DRIVING_FLAG", "PASSPORT_FLAG", "PRIMARY_INSTAL_AMT"],
        "Fuga de información": ["EMPLOYEE_CODE_ID"],
        "Sensible / demasiadas categorías": ["CURRENT_PINCODE_ID", "SUPPLIER_ID"],
        "Ya transformada": ["DATE_OF_BIRTH", "DISBURSAL_DATE", "AVERAGE_ACCT_AGE", "CREDIT_HISTORY_LENGTH"],
        "A decidir por el equipo (sensible)": [] if prep.USAR_AADHAR else ["AADHAR_FLAG"],
        "Objetivo / partición": ["LOAN_DEFAULT", "MES"],
        "Variable del modelo": prep.VARIABLES_MODELO,
    }
    tabla_decisiones = pd.DataFrame([(v, k) for k, vs in decisiones.items() for v in vs],
                                    columns=["variable", "decision"])
    sin_decision = set(train.columns) - set(tabla_decisiones["variable"])
    assert not sin_decision, f"Columnas sin decisión: {sin_decision}"
    print(tabla_decisiones.groupby("decision")["variable"].count().to_string())
    print("Variables del modelo:", len(prep.VARIABLES_MODELO))

    columnas_salida = ["UNIQUEID", "MES"] + prep.VARIABLES_MODELO
    train_v1 = train[columnas_salida + ["LOAN_DEFAULT"]].copy()
    test_v1 = test[columnas_salida].copy()
    registrar("7 Selección", train_v1, f"{len(prep.VARIABLES_MODELO)} variables del modelo")
    registrar("7 Selección", test_v1, f"{len(prep.VARIABLES_MODELO)} variables del modelo", "test")

    # ------------------------------------------------------------------ Paso 8
    titulo("Paso 8 · Partición por tiempo")
    es_tr = train_v1["MES"].isin(prep.MESES_ENTRENAMIENTO)
    es_va = train_v1["MES"] == prep.MES_VALIDACION
    X_tr, y_tr = train_v1.loc[es_tr, prep.VARIABLES_MODELO], train_v1.loc[es_tr, "LOAN_DEFAULT"]
    X_va, y_va = train_v1.loc[es_va, prep.VARIABLES_MODELO], train_v1.loc[es_va, "LOAN_DEFAULT"]
    X_te = test_v1[prep.VARIABLES_MODELO]
    print(f"Entrenamiento (ago-sep): {len(X_tr):,} · impago {y_tr.mean():.2%}")
    print(f"Validación (oct):        {len(X_va):,} · impago {y_va.mean():.2%}")
    print(f"Prueba (nov):            {len(X_te):,} · sin etiqueta")

    # ------------------------------------------------------------------ Paso 9
    titulo("Paso 9 · Preprocesadores ajustados SOLO con entrenamiento")
    prep_logistica = prep.crear_preprocesador("logistica").fit(X_tr)
    prep_boosting = prep.crear_preprocesador("boosting").fit(X_tr)
    Xtr_lr, Xva_lr, Xte_lr = (prep_logistica.transform(X) for X in (X_tr, X_va, X_te))
    Xtr_gb, Xva_gb, Xte_gb = (prep_boosting.transform(X) for X in (X_tr, X_va, X_te))
    raros = prep_logistica.named_steps["raros"]
    print("Sucursales conservadas:", len(raros.frecuentes_["BRANCH_ID"]), "de", X_tr["BRANCH_ID"].nunique())
    print("Fabricantes conservados:", len(raros.frecuentes_["MANUFACTURER_ID"]), "de", X_tr["MANUFACTURER_ID"].nunique())
    topes = prep_logistica.named_steps["columnas"].named_transformers_["log"].named_steps["tope"].topes_
    print("Topes (percentil 99,9):", {k: round(v, 1) for k, v in topes.items()})
    print("Columnas regresión logística:", Xtr_lr.shape[1], "| gradient boosting:", Xtr_gb.shape[1])

    # ----------------------------------------------------------------- Paso 10
    titulo("Paso 10 · Verificación")
    nulos_lr = int(sum(X.isna().sum().sum() for X in (Xtr_lr, Xva_lr, Xte_lr)))
    nulos_gb = int(sum(X.isna().sum().sum() for X in (Xtr_gb, Xva_gb, Xte_gb)))
    mismas = list(Xtr_lr.columns) == list(Xva_lr.columns) == list(Xte_lr.columns)
    print("Vacíos después de transformar (logística / boosting):", nulos_lr, "/", nulos_gb)
    print("Mismas columnas en los tres conjuntos:", mismas)
    print(f"Filas conservadas: train {len(train_v1):,}/{len(train_raw):,} · test {len(test_v1):,}/{len(test_raw):,}")
    print(f"Impago total {train_v1['LOAN_DEFAULT'].mean():.2%} · octubre {y_va.mean():.2%}")
    assert nulos_lr == 0 and nulos_gb == 0 and mismas
    assert len(train_v1) == len(train_raw) and len(test_v1) == len(test_raw)

    comparacion = pd.DataFrame({
        nombre: {
            "LTV promedio": X["LTV"].mean(),
            "Edad promedio": X["EDAD"].mean(),
            "% sin historial en buró": (X["CNS_GRUPO"] == "Sin historial").mean() * 100,
            "% independientes": (X["EMPLOYMENT_TYPE"] == "Self employed").mean() * 100,
            "% sucursal 'OTRA'": (raros.transform(X)["BRANCH_ID"] == "OTRA").mean() * 100,
        } for nombre, X in [("Entrenamiento", X_tr), ("Validacion", X_va), ("Prueba", X_te)]
    }).round(2)
    print("\n", comparacion.to_string())

    # ----------------------------------------------------------------- Paso 11
    titulo(f"Paso 11 · Exportar versión 1 en {dir_salida}")
    os.makedirs(dir_salida, exist_ok=True)
    train_v1.to_csv(os.path.join(dir_salida, "train_limpio.csv"), index=False)
    test_v1.to_csv(os.path.join(dir_salida, "test_limpio.csv"), index=False)
    joblib.dump(prep_logistica, os.path.join(dir_salida, "prep_logistica.pkl"))
    joblib.dump(prep_boosting, os.path.join(dir_salida, "prep_boosting.pkl"))
    reporte.to_csv(os.path.join(dir_salida, "reporte_calidad.csv"), index=False)
    tabla_decisiones.to_csv(os.path.join(dir_salida, "decisiones_variables.csv"), index=False)
    comparacion.to_csv(os.path.join(dir_salida, "comparacion_conjuntos.csv"))
    registrar("11 Exportar", train_v1, "Versión 1 guardada")
    pd.DataFrame(bitacora).to_csv(os.path.join(dir_salida, "bitacora.csv"), index=False)
    print(pd.DataFrame(bitacora).to_string(index=False))
    print("\nListo. Archivos:", sorted(os.listdir(dir_salida)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Limpieza de datos - créditos vehiculares (Fase 2 CRISP-ML(Q))")
    parser.add_argument("--datos", default=".", help="Carpeta con train.csv y test.csv (por defecto, la actual)")
    parser.add_argument("--salida", default="datos_v1", help="Carpeta de salida (por defecto, datos_v1)")
    args = parser.parse_args()
    main(args.datos, args.salida)
