import pandas as pd
import numpy as np

def cargar_datos(ruta="ventas_raw.xlsx", uploaded_file=None):
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_excel(ruta)
    df = df[["Fecha Emisión", "Importe Final", "Doc. Auxiliar", "Razón Social"]].copy()
    df["Fecha Emisión"] = pd.to_datetime(df["Fecha Emisión"])
    return df

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
