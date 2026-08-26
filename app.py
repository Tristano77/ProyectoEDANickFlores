import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

st.set_page_config(
    page_title="Telco Customer Churn - EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# MENÚ LATERAL
# ==========================================================

st.sidebar.title("📊 Telco Customer Churn")

st.sidebar.markdown("---")

opcion = st.sidebar.selectbox(
    "Seleccione una opción",
    (
        "Home",
        "Carga del dataset",
        "EDA",
        "Conclusiones"
    )
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Proyecto aplicado de Análisis Exploratorio de Datos (EDA)"
)


# ==========================================================
# VARIABLES DE SESIÓN
# ==========================================================

if "df_telco" not in st.session_state:
    st.session_state.df_telco = None


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

def cargar_dataset(uploaded_file):
    """
    Carga el archivo CSV y realiza una validación básica.
    """
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("El archivo CSV está vacío.")
            return None

        columnas_esperadas = {
            "customerID", "gender", "SeniorCitizen", "Partner",
            "Dependents", "tenure", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
        }

        columnas_faltantes = columnas_esperadas - set(df.columns)

        if columnas_faltantes:
            st.warning(
                "El archivo fue cargado, pero faltan columnas esperadas: "
                + ", ".join(sorted(columnas_faltantes))
            )

        return df

    except Exception as e:
        st.error(f"No fue posible cargar el archivo: {e}")
        return None


def mostrar_resumen_dataset(df):
    """
    Muestra las principales características del dataset.
    """
    filas, columnas = df.shape

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Filas", f"{filas:,}")

    with c2:
        st.metric("Columnas", columnas)

    with c3:
        st.metric("Valores nulos", int(df.isna().sum().sum()))



def clasificar_variables(df):
    """
    Clasifica las variables según el tipo de dato almacenado
    en el DataFrame.

    Retorna:
        numericas: lista de variables numéricas
        categoricas: lista de variables categóricas
    """

    numericas = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categoricas = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numericas, categoricas


# ==========================================================
# MÓDULO 1 - HOME
# ==========================================================

def home():

    st.title("📊 Telco Customer Churn - Análisis Exploratorio de Datos")

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            """
            ### 🎯 Objetivo

            Desarrollar una aplicación interactiva en Streamlit para
            realizar un **Análisis Exploratorio de Datos (EDA)** sobre
            clientes de una empresa de telecomunicaciones.

            El análisis busca identificar patrones asociados a la
            **fuga de clientes (Churn)** mediante limpieza,
            transformación, estadística descriptiva y visualización.
            """
        )

    with col2:
        st.markdown(
            """
            ### 📌 Alcance del proyecto

            Este proyecto **no desarrolla modelos predictivos**.
            El objetivo es comprender los datos y obtener hallazgos
            útiles para la toma de decisiones relacionadas con la
            retención de clientes.

            El dataset contiene información sobre:

            - Características demográficas.
            - Servicios contratados.
            - Tipo de contrato.
            - Método de pago.
            - Antigüedad del cliente.
            - Cargos mensuales y totales.
            - Estado de fuga del cliente.
            """
        )

    st.markdown("---")

    st.header("👤 Información del estudiante")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("**Nombre:** Jhon Doe")

    with c2:
        st.write("**Especialización:** Python for Analytics")

    with c3:
        st.write("**Año:** 2026")

    st.markdown("---")

    st.header("🗂️ Información del dataset")

    st.write(
        "El dataset **TelcoCustomerChurn.csv** contiene información "
        "de clientes, sus servicios contratados, facturación mensual, "
        "tiempo de permanencia y estado actual en la empresa."
    )

    st.markdown(
        """
        **Variable objetivo del análisis:** `Churn`

        - `Yes`: el cliente abandonó la empresa.
        - `No`: el cliente permanece en la empresa.
        """
    )

    st.markdown("---")

    st.header("🛠️ Tecnologías utilizadas")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Lenguaje", "Python")

    with c2:
        st.metric("Framework", "Streamlit")

    with c3:
        st.metric("Datos", "Pandas")

    with c4:
        st.metric("Cálculo", "NumPy")

    with c5:
        st.metric("Visualización", "Matplotlib / Seaborn")

    st.markdown("---")

    st.success(
        "Utilice el menú lateral para cargar el dataset y comenzar el EDA."
    )


# ==========================================================
# MÓDULO 2 - CARGA DEL DATASET
# ==========================================================

def carga_dataset():

    st.title("📂 Carga del dataset")

    st.markdown(
        """
        ### Descripción

        Antes de realizar cualquier análisis, el usuario debe cargar
        el archivo CSV mediante `st.file_uploader()`.

        Una vez cargado correctamente se mostrará una vista previa,
        las dimensiones y una validación básica de los datos.
        """
    )

    st.divider()

    archivo = st.file_uploader(
        "Seleccione el archivo TelcoCustomerChurn.csv",
        type=["csv"]
    )

    if archivo is None:
        st.info(
            "⬆️ Cargue el archivo CSV para habilitar el análisis."
        )
        return

    df = cargar_dataset(archivo)

    if df is None:
        return

    st.session_state.df_telco = df

    st.success("✅ Dataset cargado correctamente.")

    st.divider()

    st.subheader("📐 Dimensiones del dataset")

    mostrar_resumen_dataset(df)

    st.divider()

    st.subheader("👀 Vista previa")

    cantidad_filas = st.slider(
        "Cantidad de registros a visualizar",
        min_value=5,
        max_value=min(20, len(df)),
        value=min(10, len(df)),
        step=1
    )

    st.dataframe(
        df.head(cantidad_filas),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("📋 Columnas disponibles")

    columnas_df = pd.DataFrame({
        "Variable": df.columns,
        "Tipo de dato": df.dtypes.astype(str).values
    })

    st.dataframe(
        columnas_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("🔎 Validación inicial")

    c1, c2 = st.columns(2)

    with c1:
        st.write("**Valores nulos por columna**")
        nulos = df.isna().sum().to_frame("Valores nulos")
        st.dataframe(
            nulos,
            use_container_width=True
        )

    with c2:
        st.write("**Valores duplicados**")
        duplicados = int(df.duplicated().sum())
        st.metric(
            "Registros duplicados",
            duplicados
        )

    st.warning(
        "La etapa de análisis EDA se habilitará sobre el dataset "
        "cargado en memoria."
    )


# ==========================================================
# MÓDULO 3 - EDA
# ==========================================================

def eda():

    st.title("🔎 Análisis Exploratorio de Datos")

    if st.session_state.df_telco is None:
        st.warning(
            "⚠️ Primero debe cargar el dataset desde "
            "'Carga del dataset'."
        )
        return

    df = st.session_state.df_telco

    st.success(
        f"Dataset disponible: {df.shape[0]:,} registros y "
        f"{df.shape[1]} variables."
    )

    tabs = st.tabs([
        "1. Información general",
        "2. Variables",
        "3. Estadísticas",
        "4. Valores faltantes",
        "5. Distribuciones",
        "6. Categóricas",
        "7. Numérica vs Churn",
        "8. Categórica vs Churn",
        "9. Análisis dinámico",
        "10. Hallazgos"
    ])

    # ==========================================================
    # ÍTEM 1
    # ==========================================================

    with tabs[0]:

        st.subheader("1. Información general del dataset")

        st.markdown(
            """
            En este apartado se revisa la estructura general del
            dataset mediante sus dimensiones, tipos de datos y
            cantidad de valores nulos.
            """
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Filas", f"{df.shape[0]:,}")

        with c2:
            st.metric("Columnas", df.shape[1])

        with c3:
            st.metric(
                "Valores nulos",
                int(df.isna().sum().sum())
            )

        st.divider()

        st.markdown("#### Tipos de datos y valores nulos")

        resumen = pd.DataFrame({
            "Variable": df.columns,
            "Tipo de dato": df.dtypes.astype(str).values,
            "Valores no nulos": df.notna().sum().values,
            "Valores nulos": df.isna().sum().values
        })

        st.dataframe(
            resumen,
            use_container_width=True,
            hide_index=True
        )

    # ==========================================================
    # ÍTEM 2 - CLASIFICACIÓN DE VARIABLES
    # ==========================================================

    with tabs[1]:

        st.subheader("2. Clasificación de variables")

        st.markdown(
            """
            Las variables se clasifican automáticamente mediante una
            **función personalizada** según el tipo de dato almacenado
            en el DataFrame.

            Esta clasificación permite diferenciar las variables que
            serán analizadas mediante estadísticas numéricas de
            aquellas que requieren análisis de frecuencias y
            proporciones.
            """
        )

        st.divider()

        numericas, categoricas = clasificar_variables(df)

        total_variables = len(df.columns)
        total_numericas = len(numericas)
        total_categoricas = len(categoricas)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Total de variables",
                total_variables
            )

        with c2:
            st.metric(
                "Variables numéricas",
                total_numericas
            )

        with c3:
            st.metric(
                "Variables categóricas",
                total_categoricas
            )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### 🔢 Variables numéricas")

            if numericas:
                df_numericas = pd.DataFrame({
                    "Variable": numericas,
                    "Tipo de dato": [
                        str(df[col].dtype) for col in numericas
                    ]
                })

                st.dataframe(
                    df_numericas,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No se encontraron variables numéricas.")

        with col2:

            st.markdown("#### 🔤 Variables categóricas")

            if categoricas:
                df_categoricas = pd.DataFrame({
                    "Variable": categoricas,
                    "Tipo de dato": [
                        str(df[col].dtype) for col in categoricas
                    ]
                })

                st.dataframe(
                    df_categoricas,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No se encontraron variables categóricas.")

        st.divider()

        st.markdown("#### 📊 Distribución de la clasificación")

        clasificacion = pd.DataFrame({
            "Tipo de variable": [
                "Numéricas",
                "Categóricas"
            ],
            "Cantidad": [
                total_numericas,
                total_categoricas
            ]
        })

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            clasificacion["Tipo de variable"],
            clasificacion["Cantidad"]
        )

        ax.set_xlabel("Tipo de variable")
        ax.set_ylabel("Cantidad")
        ax.set_title("Cantidad de variables por tipo")

        for i, valor in enumerate(clasificacion["Cantidad"]):
            ax.text(
                i,
                valor,
                str(valor),
                ha="center",
                va="bottom"
            )

        fig.tight_layout()
        st.pyplot(fig)

        st.divider()

        st.markdown("#### 🔍 Exploración de variables categóricas")

        mostrar_unicos = st.checkbox(
            "Mostrar cantidad de valores únicos por variable categórica"
        )

        if mostrar_unicos and categoricas:

            unicos = pd.DataFrame({
                "Variable": categoricas,
                "Valores únicos": [
                    df[col].nunique(dropna=True)
                    for col in categoricas
                ]
            }).sort_values(
                "Valores únicos",
                ascending=False
            )

            st.dataframe(
                unicos,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        st.markdown("#### 💡 Interpretación")

        st.write(
            f"El dataset contiene **{total_numericas} variables "
            f"numéricas** y **{total_categoricas} variables "
            f"categóricas**, de un total de **{total_variables} "
            f"variables**."
        )

        if "TotalCharges" in categoricas:
            st.warning(
                "⚠️ `TotalCharges` aparece como variable categórica "
                "porque actualmente está almacenada como `object`. "
                "Este comportamiento será revisado durante la etapa "
                "de limpieza y transformación de datos."
            )

    # ==========================================================
    # ÍTEMS PENDIENTES
    # ==========================================================

    nombres_pendientes = [
        "Estadísticas descriptivas",
        "Análisis de valores faltantes",
        "Distribución de variables numéricas",
        "Análisis de variables categóricas",
        "Análisis bivariado: numérico vs Churn",
        "Análisis bivariado: categórico vs Churn",
        "Análisis basado en parámetros seleccionados",
        "Hallazgos clave"
    ]

    for i in range(2, 10):
        with tabs[i]:
            st.subheader(
                f"{i + 1}. {nombres_pendientes[i - 2]}"
            )
            st.info(
                "Este ítem se implementará en el siguiente paso."
            )


# ==========================================================
# MÓDULO 4 - CONCLUSIONES
# ==========================================================

def conclusiones():

    st.title("📝 Conclusiones")

    if st.session_state.df_telco is None:
        st.warning(
            "Primero debe cargar el dataset y completar el EDA."
        )
        return

    st.info(
        "Las cinco conclusiones finales se generarán a partir de "
        "los hallazgos obtenidos durante el EDA."
    )


# ==========================================================
# ENRUTAMIENTO PRINCIPAL
# ==========================================================

if opcion == "Home":
    home()

elif opcion == "Carga del dataset":
    carga_dataset()

elif opcion == "EDA":
    eda()

elif opcion == "Conclusiones":
    conclusiones()
