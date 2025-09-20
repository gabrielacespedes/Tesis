import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd

from data import cargar_datos, procesar_serie, split_train_test
from model import entrenar_sarima, forecast, calcular_metricas

# ==============================
# LOGIN
# ==============================
def login():
    st.title("🔒 Login")
    st.markdown("Ingresa usuario y contraseña para acceder a la app")

    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    login_button = st.button("Entrar")

    if login_button:
        if username == "admin" and password == "admin":
            st.session_state["logged_in"] = True
        else:
            st.error("Usuario o contraseña incorrectos")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Predicción Ventas SARIMA", page_icon="📊", layout="wide")
st.title("Predicción de Ventas con SARIMA")

# ==============================
# CARGA Y PROCESO DE DATOS
# ==============================
uploaded_file = st.file_uploader("Sube archivo de ventas (Excel)", type=["xlsx"])
df_hist = cargar_datos(uploaded_file=uploaded_file)

if df_hist is None:
    st.stop()  # detiene si hubo error de validación

df_sum = procesar_serie(df_hist)
train, test = split_train_test(df_sum)


# ==============================
# ENTRENAR MODELO
# ==============================
model_fit = entrenar_sarima(train)
pred_test, pred_ci = forecast(model_fit, len(test))
rmse_weekly, mape_weekly, df_test_weekly, df_pred_weekly = calcular_metricas(test, pred_test)

# ---------------------
# TABS
# ---------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard", "Tabla de Predicciones", "Evaluación del Modelo",
    "Análisis por Clientes", "Estacionalidad y Tendencias"
])

# TAB1: Dashboard
with tab1:
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(train.index, train.values, label="Train", marker="o")
    ax.plot(test.index, test.values, label="Test", color="black", linewidth=1.2)
    ax.plot(pred_test.index, pred_test.values, label="Predicción Test", color="red")
    ax.set_title("SARIMA Histórico")
    ax.set_ylabel("Ventas (S/)")
    ax.legend()
    st.pyplot(fig)

# TAB2: Tabla de Predicciones (con selección de fecha)
with tab2:
    ultimo_dato_train = train.index.max()
    min_fecha = ultimo_dato_train + pd.Timedelta(days=1)
    max_fecha = ultimo_dato_train + pd.Timedelta(days=30)
    fecha_seleccionada = st.date_input(
        "Selecciona la fecha de predicción",
        min_value=min_fecha.date(),
        max_value=max_fecha.date(),
        value=min_fecha.date()
    )
    fecha_seleccionada = pd.Timestamp(fecha_seleccionada)
    dias_forecast = (fecha_seleccionada - ultimo_dato_train).days

    forecast_mean, forecast_ci = forecast(model_fit, dias_forecast)
    fechas_forecast = pd.date_range(ultimo_dato_train + pd.Timedelta(days=1), periods=dias_forecast)
    df_forecast = pd.DataFrame({
        "Fecha": fechas_forecast,
        "Predicción": forecast_mean,
        "LI": forecast_ci.iloc[:,0],
        "LS": forecast_ci.iloc[:,1]
    }).set_index("Fecha").resample('W-SUN').sum().reset_index()
    st.dataframe(df_forecast[["Fecha", "Predicción"]], use_container_width=True)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_forecast[["Fecha","Predicción"]].to_excel(writer, index=False, sheet_name="Forecast Semanal")
    st.download_button("Descargar forecast semanal", data=output.getvalue(),
                       file_name="forecast_sarima_semanal.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# ------------------------------
# TAB 3: Evaluación Semanal
# ------------------------------
with tab3:
    col1, col2 = st.columns(2)
    col1.metric("📏 RMSE", f"{rmse_weekly:.2f}")
    col2.metric("📐 MAPE", f"{mape_weekly:.2f} %")

    fig_eval, ax_eval = plt.subplots(figsize=(10,4))
    ax_eval.plot(df_test_weekly.index, df_test_weekly.values, label="Real Semanal", marker="o")
    ax_eval.plot(df_pred_weekly.index, df_pred_weekly.values, label="Predicción Semanal", marker="x")
    ax_eval.set_title("Evaluación Semanal del Modelo")
    ax_eval.set_ylabel("Ventas S/") 
    ax_eval.set_xlabel("Semana")
    ax_eval.legend()
    st.pyplot(fig_eval)

# ------------------------------
# TAB 4: Análisis por Clientes
# ------------------------------
with tab4:
    st.subheader("Análisis de Clientes")
    total_ventas = df_hist.groupby("Razón Social")["Importe Final"].sum().sum()
    num_clientes = df_hist["Razón Social"].nunique()
    ticket_promedio = df_hist["Importe Final"].sum() / df_hist.shape[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Ventas", f"S/ {total_ventas:,.0f}")
    col2.metric("👥 Número de Clientes", f"{num_clientes}")
    col3.metric("🧾 Ticket Promedio", f"S/ {ticket_promedio:,.2f}")

    top_clientes = df_hist.groupby(["Doc. Auxiliar", "Razón Social"])["Importe Final"].sum().sort_values(ascending=False).head(10).reset_index()
    st.markdown("### 🏆 Top 10 Clientes")
    st.dataframe(top_clientes, use_container_width=True, height=300)

    cliente_seleccionado = st.selectbox("Selecciona un cliente para ver su evolución", top_clientes["Razón Social"].unique())
    df_cliente = df_hist[df_hist["Razón Social"] == cliente_seleccionado].sort_values("Fecha Emisión")

    st.markdown(f"### 📈 Evolución de ventas: {cliente_seleccionado}")
    fig_cliente, ax_cliente = plt.subplots(figsize=(10,4))
    ax_cliente.plot(df_cliente["Fecha Emisión"], df_cliente["Importe Final"], marker="o", color="tab:blue")
    ax_cliente.set_xlabel("Fecha")
    ax_cliente.set_ylabel("Ventas (S/)")
    ax_cliente.set_title(f"Ventas Diarias de {cliente_seleccionado}")
    st.pyplot(fig_cliente)

# ------------------------------
# TAB 5: Estacionalidad por Día de la Semana
# ------------------------------
with tab5:
    st.subheader("Estacionalidad por Día de la Semana")

    # nombres de días a español
    map_dias = {
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
    }
    df_sum["DiaSemana"] = df_sum["Fecha"].dt.day_name().map(map_dias)

    ultimo_dato = df_sum["Fecha"].max()

    # Último mes
    inicio_ultimo_mes = ultimo_dato - pd.DateOffset(months=1)
    df_ultimo_mes = df_sum[df_sum["Fecha"] > inicio_ultimo_mes]

    ventas_dia_ultimo_mes = df_ultimo_mes.groupby("DiaSemana")["Importe Final"].sum().reindex(
        ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
    )

    fig_mes, ax_mes = plt.subplots(figsize=(10,5))
    ax_mes.bar(ventas_dia_ultimo_mes.index, ventas_dia_ultimo_mes.values, color="tab:blue")
    ax_mes.set_title("Último Mes")
    ax_mes.set_ylabel("Ventas (S/)")
    ax_mes.set_xlabel("Día de la Semana")
    st.pyplot(fig_mes)

    # Último año
    inicio_ultimo_anio = ultimo_dato - pd.DateOffset(years=1)
    df_ultimo_anio = df_sum[df_sum["Fecha"] > inicio_ultimo_anio]

    ventas_dia_ultimo_anio = df_ultimo_anio.groupby("DiaSemana")["Importe Final"].sum().reindex(
        ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
    )

    fig_anio, ax_anio = plt.subplots(figsize=(10,5))
    ax_anio.bar(ventas_dia_ultimo_anio.index, ventas_dia_ultimo_anio.values, color="tab:green")
    ax_anio.set_title("Último Año")
    ax_anio.set_ylabel("Ventas (S/)")
    ax_anio.set_xlabel("Día de la Semana")
    st.pyplot(fig_anio)


