import pandas as pd
import streamlit as st
import os
import numpy as np

def cargar_datos(uploaded_file=None, archivo_historico="ventas_raw.xlsx"):
    """
    Carga datos históricos y nuevos desde Excel.
    Combina histórico con datos subidos, validando formato.
    """

    # -----------------------
    # Leer histórico
    # -----------------------
    if os.path.exists(archivo_historico):
        try:
            df_hist = pd.read_excel(archivo_historico)
        except Exception as e:
            st.error(f"Error al leer el histórico local: {e}")
            return None
    else:
        df_hist = pd.DataFrame()

    # -----------------------
    # Leer archivo subido
    # -----------------------
    if uploaded_file is not None:
        try:
            df_new = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"No se pudo leer el archivo subido. Error: {e}")
            return None

        # Validar columnas obligatorias
        columnas_req = ["Fecha Emisión", "Importe Final", "Doc. Auxiliar", "Razón Social"]
        faltantes = [col for col in columnas_req if col not in df_new.columns]
        if faltantes:
            st.error(f"El archivo subido no contiene estas columnas obligatorias: {faltantes}")
            return None

        # Normalizar tipos
        df_new["Fecha Emisión"] = pd.to_datetime(df_new["Fecha Emisión"], errors="coerce")
        df_new["Importe Final"] = pd.to_numeric(df_new["Importe Final"], errors="coerce")

        # Eliminar nulos
        df_new = df_new.dropna(subset=["Fecha Emisión", "Importe Final"])

        # -----------------------
        # Comparar con histórico
        # -----------------------
        if not df_hist.empty:
            # Detectar si TODO el archivo ya está en el histórico
            merged = pd.merge(
                df_new,
                df_hist,
                on=["Fecha Emisión", "Doc. Auxiliar", "Importe Final", "Razón Social"],
                how="left",
                indicator=True
            )

            if (merged["_merge"] == "both").all():
                st.warning("El archivo subido no contiene información nueva. No se modificó el histórico.")
                return df_hist  # devuelve histórico intacto

            # Concatenar solo las filas nuevas
            df_hist = pd.concat([df_hist, df_new], ignore_index=True)

        else:
            df_hist = df_new.copy()

        # Eliminar duplicados
        df_hist = df_hist.drop_duplicates(
            subset=["Fecha Emisión", "Doc. Auxiliar", "Importe Final"]
        )

        # Ordenar cronológicamente
        df_hist = df_hist.sort_values("Fecha Emisión").reset_index(drop=True)

        # Guardar histórico actualizado
        try:
            df_hist.to_excel(archivo_historico, index=False)
        except Exception as e:
            st.error(f"No se pudo guardar el histórico actualizado: {e}")

    if df_hist.empty:
        st.error("No se cargaron datos válidos. Por favor sube un archivo Excel correcto.")
        return None

    return df_hist


def procesar_serie(df):
    df_sum = df.groupby("Fecha Emisión", as_index=False)["Importe Final"].sum()
    full_range = pd.date_range(df_sum["Fecha Emisión"].min(), df_sum["Fecha Emisión"].max(), freq="D")
    df_sum = df_sum.set_index("Fecha Emisión").reindex(full_range).fillna(0).rename_axis("Fecha").reset_index()
    df_sum["Importe Final"] = df_sum["Importe Final"].replace(0, np.nan)
    df_sum["Importe Final"] = df_sum["Importe Final"].fillna(df_sum["Importe Final"].rolling(7, min_periods=1).mean())
    df_sum["Importe Final"] = df_sum["Importe Final"].fillna(method="bfill").fillna(method="ffill")
    return df_sum


def split_train_test(df, meses_test=1):
    ultimo_dia = df["Fecha"].max()
    train_end_date = ultimo_dia - pd.DateOffset(months=meses_test)
    train = df[df["Fecha"] <= train_end_date].set_index("Fecha")["Importe Final"]
    test  = df[df["Fecha"] > train_end_date].set_index("Fecha")["Importe Final"]
    return train, test