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



def calcular_estadisticas_descriptivas(df):
    """
    Calcula estadísticas descriptivas para las variables numéricas.

    Incluye:
    - Conteo
    - Media
    - Mediana
    - Moda
    - Desviación estándar
    - Mínimo
    - Cuartiles
    - Máximo
    """

    numericas = df.select_dtypes(include=np.number).columns.tolist()

    if not numericas:
        return pd.DataFrame()

    estadisticas = pd.DataFrame(index=numericas)

    estadisticas["count"] = df[numericas].count()
    estadisticas["media"] = df[numericas].mean()
    estadisticas["mediana"] = df[numericas].median()
    estadisticas["moda"] = df[numericas].mode().iloc[0]
    estadisticas["desv. estándar"] = df[numericas].std()
    estadisticas["mínimo"] = df[numericas].min()
    estadisticas["Q1 (25%)"] = df[numericas].quantile(0.25)
    estadisticas["Q3 (75%)"] = df[numericas].quantile(0.75)
    estadisticas["máximo"] = df[numericas].max()

    estadisticas["rango"] = (
        estadisticas["máximo"] - estadisticas["mínimo"]
    )

    estadisticas["IQR"] = (
        estadisticas["Q3 (75%)"] - estadisticas["Q1 (25%)"]
    )

    return estadisticas



def analizar_valores_faltantes(df):
    """
    Analiza valores faltantes del DataFrame.

    Considera:
    1. Valores nulos detectados por Pandas.
    2. Cadenas vacías.
    3. Cadenas compuestas únicamente por espacios.

    Retorna una tabla resumen por variable.
    """

    nulos_pandas = df.isna().sum()

    vacios = pd.Series(0, index=df.columns, dtype="int64")
    espacios = pd.Series(0, index=df.columns, dtype="int64")

    for columna in df.select_dtypes(include=["object", "category"]).columns:
        serie = df[columna].astype("string")

        vacios[columna] = (
            serie.eq("").fillna(False).sum()
        )

        espacios[columna] = (
            serie.str.strip().eq("").fillna(False).sum()
            - vacios[columna]
        )

    resumen = pd.DataFrame({
        "Variable": df.columns,
        "Nulos (NaN)": nulos_pandas.values,
        "Cadenas vacías": vacios.values,
        "Solo espacios": espacios.values
    })

    resumen["Total faltantes detectados"] = (
        resumen["Nulos (NaN)"]
        + resumen["Cadenas vacías"]
        + resumen["Solo espacios"]
    )

    resumen["Porcentaje (%)"] = (
        resumen["Total faltantes detectados"]
        / len(df)
        * 100
    ).round(2)

    return resumen



def preparar_numericas_para_graficos(df):
    """
    Obtiene las variables numéricas disponibles para el análisis
    de distribución.

    También intenta convertir TotalCharges a numérico sin modificar
    el DataFrame original. Esto permite visualizarla correctamente
    sin alterar todavía el dataset almacenado en session_state.
    """

    datos = df.copy()

    if "TotalCharges" in datos.columns:
        datos["TotalCharges"] = pd.to_numeric(
            datos["TotalCharges"],
            errors="coerce"
        )

    numericas = datos.select_dtypes(
        include=np.number
    ).columns.tolist()

    return datos, numericas



def preparar_datos_churn(df):
    """
    Prepara una copia del DataFrame para el análisis bivariado
    de variables numéricas frente a Churn.

    TotalCharges se convierte temporalmente a numérico si existe.
    Churn se conserva en su representación categórica.
    """

    datos = df.copy()

    if "TotalCharges" in datos.columns:
        datos["TotalCharges"] = pd.to_numeric(
            datos["TotalCharges"],
            errors="coerce"
        )

    return datos


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
    # ÍTEMS 3 AL 10
    # ==========================================================

    with tabs[2]:

        st.subheader("3. Estadísticas descriptivas")

        st.markdown(
            """
            Las estadísticas descriptivas permiten resumir el
            comportamiento de las variables numéricas del dataset.

            Se analizarán principalmente la **media, mediana, moda,
            dispersión, valores mínimos y máximos**, además de los
            cuartiles.
            """
        )

        st.divider()

        estadisticas = calcular_estadisticas_descriptivas(df)

        if estadisticas.empty:

            st.warning(
                "No se encontraron variables numéricas para "
                "calcular estadísticas descriptivas."
            )

        else:

            # ------------------------------------------------------
            # Tabla equivalente a describe() ampliada
            # ------------------------------------------------------

            st.markdown(
                "#### 📊 Resumen estadístico"
            )

            st.dataframe(
                estadisticas.round(2),
                use_container_width=True
            )

            st.caption(
                "La tabla amplía la información de "
                "`df.describe()` incorporando también la moda, "
                "el rango y el IQR."
            )

            st.divider()

            # ------------------------------------------------------
            # Métricas generales
            # ------------------------------------------------------

            st.markdown(
                "#### 📌 Variables numéricas disponibles"
            )

            variables_numericas = estadisticas.index.tolist()

            seleccion = st.selectbox(
                "Seleccione una variable para analizar",
                variables_numericas,
                key="eda_estadistica_variable"
            )

            serie = df[seleccion].dropna()

            media = serie.mean()
            mediana = serie.median()
            moda = serie.mode()

            if len(moda) > 0:
                moda_valor = moda.iloc[0]
            else:
                moda_valor = np.nan

            desviacion = serie.std()
            minimo = serie.min()
            maximo = serie.max()

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Media",
                    f"{media:,.2f}"
                )

            with c2:
                st.metric(
                    "Mediana",
                    f"{mediana:,.2f}"
                )

            with c3:
                st.metric(
                    "Moda",
                    f"{moda_valor:,.2f}"
                )

            c4, c5, c6 = st.columns(3)

            with c4:
                st.metric(
                    "Desviación estándar",
                    f"{desviacion:,.2f}"
                )

            with c5:
                st.metric(
                    "Mínimo",
                    f"{minimo:,.2f}"
                )

            with c6:
                st.metric(
                    "Máximo",
                    f"{maximo:,.2f}"
                )

            st.divider()

            # ------------------------------------------------------
            # Interpretación
            # ------------------------------------------------------

            st.markdown(
                "#### 💡 Interpretación básica"
            )

            diferencia = abs(media - mediana)

            if media > mediana:
                tendencia = (
                    "La media es superior a la mediana, lo que puede "
                    "indicar una distribución con mayor concentración "
                    "de valores hacia la derecha."
                )
            elif media < mediana:
                tendencia = (
                    "La media es inferior a la mediana, lo que puede "
                    "indicar una distribución con mayor concentración "
                    "de valores hacia la izquierda."
                )
            else:
                tendencia = (
                    "La media y la mediana son prácticamente iguales, "
                    "lo que sugiere una distribución relativamente "
                    "centrada alrededor de su valor central."
                )

            st.write(
                f"Para **{seleccion}**, la media es "
                f"**{media:,.2f}**, mientras que la mediana es "
                f"**{mediana:,.2f}**."
            )

            st.write(tendencia)

            st.write(
                f"La desviación estándar es **{desviacion:,.2f}**, "
                f"por lo que existe una dispersión de los valores "
                f"alrededor de la media. El rango observado va desde "
                f"**{minimo:,.2f}** hasta **{maximo:,.2f}**."
            )

            if diferencia > 0:
                st.info(
                    "La diferencia entre media y mediana debe "
                    "interpretarse junto con la distribución gráfica "
                    "de la variable; por sí sola no demuestra la "
                    "existencia de valores atípicos."
                )

            st.divider()

            # ------------------------------------------------------
            # Distribución de la variable seleccionada
            # ------------------------------------------------------

            st.markdown(
                "#### 📈 Distribución de la variable seleccionada"
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.hist(
                serie,
                bins=30,
                edgecolor="black"
            )

            ax.axvline(
                media,
                linestyle="--",
                linewidth=2,
                label=f"Media: {media:,.2f}"
            )

            ax.axvline(
                mediana,
                linestyle=":",
                linewidth=2,
                label=f"Mediana: {mediana:,.2f}"
            )

            ax.set_title(
                f"Distribución de {seleccion}"
            )
            ax.set_xlabel(seleccion)
            ax.set_ylabel("Frecuencia")
            ax.legend()

            fig.tight_layout()

            st.pyplot(fig)

    # ==========================================================
    # ÍTEM 4 - VALORES FALTANTES
    # ==========================================================

    with tabs[3]:

        st.subheader("4. Análisis de valores faltantes")

        st.markdown(
            """
            En esta etapa se identifican los valores faltantes del
            dataset y se calcula su proporción respecto al total de
            registros.

            Además de los valores `NaN` detectados directamente por
            Pandas, se revisan cadenas vacías y cadenas que contienen
            únicamente espacios, ya que pueden representar valores
            faltantes encubiertos.
            """
        )

        st.divider()

        resumen_faltantes = analizar_valores_faltantes(df)

        total_nulos = int(
            resumen_faltantes["Nulos (NaN)"].sum()
        )

        total_vacios = int(
            resumen_faltantes["Cadenas vacías"].sum()
        )

        total_espacios = int(
            resumen_faltantes["Solo espacios"].sum()
        )

        total_faltantes = int(
            resumen_faltantes["Total faltantes detectados"].sum()
        )

        # ------------------------------------------------------
        # Métricas
        # ------------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Nulos (NaN)",
                f"{total_nulos:,}"
            )

        with c2:
            st.metric(
                "Cadenas vacías",
                f"{total_vacios:,}"
            )

        with c3:
            st.metric(
                "Solo espacios",
                f"{total_espacios:,}"
            )

        with c4:
            st.metric(
                "Total detectado",
                f"{total_faltantes:,}"
            )

        st.divider()

        # ------------------------------------------------------
        # Tabla completa
        # ------------------------------------------------------

        st.markdown(
            "#### 📋 Detalle de valores faltantes por variable"
        )

        st.dataframe(
            resumen_faltantes.sort_values(
                "Total faltantes detectados",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ------------------------------------------------------
        # Variables afectadas
        # ------------------------------------------------------

        afectadas = resumen_faltantes[
            resumen_faltantes["Total faltantes detectados"] > 0
        ].copy()

        if afectadas.empty:

            st.success(
                "✅ No se encontraron valores faltantes ni cadenas "
                "vacías en las variables analizadas."
            )

        else:

            st.markdown(
                "#### ⚠️ Variables con valores faltantes"
            )

            st.write(
                f"Se encontraron **{len(afectadas)} variables** "
                "con algún tipo de valor faltante."
            )

            # --------------------------------------------------
            # Gráfico
            # --------------------------------------------------

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.bar(
                afectadas["Variable"],
                afectadas["Total faltantes detectados"]
            )

            ax.set_title(
                "Valores faltantes por variable"
            )
            ax.set_xlabel("Variable")
            ax.set_ylabel("Cantidad")

            plt.xticks(
                rotation=45,
                ha="right"
            )

            fig.tight_layout()

            st.pyplot(fig)

            # --------------------------------------------------
            # Porcentaje
            # --------------------------------------------------

            st.markdown(
                "#### 📊 Porcentaje de faltantes"
            )

            fig2, ax2 = plt.subplots(figsize=(10, 5))

            ax2.bar(
                afectadas["Variable"],
                afectadas["Porcentaje (%)"]
            )

            ax2.set_title(
                "Porcentaje de registros faltantes por variable"
            )
            ax2.set_xlabel("Variable")
            ax2.set_ylabel("Porcentaje (%)")

            plt.xticks(
                rotation=45,
                ha="right"
            )

            fig2.tight_layout()

            st.pyplot(fig2)

            st.divider()

            # --------------------------------------------------
            # Interpretación
            # --------------------------------------------------

            st.markdown(
                "#### 💡 Interpretación"
            )

            variable_mayor = afectadas.loc[
                afectadas["Total faltantes detectados"].idxmax()
            ]

            st.write(
                f"La variable con mayor cantidad de valores "
                f"faltantes detectados es **"
                f"{variable_mayor['Variable']}**, con "
                f"**{int(variable_mayor['Total faltantes detectados']):,} "
                f"registros**, equivalentes al "
                f"**{variable_mayor['Porcentaje (%)']:.2f}%** "
                "de sus observaciones."
            )

            if total_nulos > 0:
                st.info(
                    "Los valores `NaN` son reconocidos directamente "
                    "por Pandas y deberán evaluarse antes de aplicar "
                    "estadísticas o visualizaciones que dependan de "
                    "la variable afectada."
                )

            if total_vacios + total_espacios > 0:
                st.warning(
                    "También se detectaron valores faltantes "
                    "representados mediante texto vacío o espacios. "
                    "Estos valores deben normalizarse durante la "
                    "etapa de limpieza."
                )

        st.divider()

        # ------------------------------------------------------
        # Control opcional para revisar registros
        # ------------------------------------------------------

        revisar = st.checkbox(
            "Mostrar registros que contienen valores faltantes",
            key="eda_valores_faltantes_checkbox"
        )

        if revisar:

            mascara = df.isna().any(axis=1)

            # Incorporar cadenas vacías / espacios en columnas de texto
            for columna in df.select_dtypes(
                include=["object", "category"]
            ).columns:

                serie = df[columna].astype("string")

                mascara = (
                    mascara
                    | serie.eq("").fillna(False)
                    | serie.str.strip().eq("").fillna(False)
                )

            registros_faltantes = df.loc[mascara]

            st.write(
                f"Registros encontrados: "
                f"**{len(registros_faltantes):,}**"
            )

            st.dataframe(
                registros_faltantes,
                use_container_width=True,
                hide_index=True
            )

    # ==========================================================
    # ÍTEM 5 - DISTRIBUCIÓN DE VARIABLES NUMÉRICAS
    # ==========================================================

    with tabs[4]:

        st.subheader("5. Distribución de variables numéricas")

        st.markdown(
            """
            En este apartado se analiza la distribución de las
            variables numéricas mediante **histogramas y boxplots**.

            Estas visualizaciones permiten observar la concentración
            de los datos, su dispersión, posibles asimetrías y la
            presencia visual de valores atípicos.
            """
        )

        st.divider()

        datos_graficos, numericas_graficos = (
            preparar_numericas_para_graficos(df)
        )

        if not numericas_graficos:

            st.warning(
                "No se encontraron variables numéricas disponibles "
                "para realizar el análisis."
            )

        else:

            # ------------------------------------------------------
            # Selector de variables
            # ------------------------------------------------------

            seleccionadas = st.multiselect(
                "Seleccione una o más variables numéricas",
                numericas_graficos,
                default=numericas_graficos[:1],
                key="eda_distribuciones_numericas"
            )

            if not seleccionadas:

                st.info(
                    "Seleccione al menos una variable para generar "
                    "las visualizaciones."
                )

            else:

                st.divider()

                # --------------------------------------------------
                # Histogramas
                # --------------------------------------------------

                st.markdown("#### 📊 Histogramas")

                for variable in seleccionadas:

                    serie = datos_graficos[variable].dropna()

                    if serie.empty:
                        st.warning(
                            f"No existen valores válidos para "
                            f"`{variable}`."
                        )
                        continue

                    fig, ax = plt.subplots(figsize=(10, 4))

                    sns.histplot(
                        data=datos_graficos,
                        x=variable,
                        bins=30,
                        kde=True,
                        ax=ax
                    )

                    ax.set_title(
                        f"Distribución de {variable}"
                    )
                    ax.set_xlabel(variable)
                    ax.set_ylabel("Frecuencia")

                    fig.tight_layout()
                    st.pyplot(fig)

                    # Medidas básicas para interpretar la forma
                    media = serie.mean()
                    mediana = serie.median()
                    desviacion = serie.std()

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Media",
                            f"{media:,.2f}"
                        )

                    with c2:
                        st.metric(
                            "Mediana",
                            f"{mediana:,.2f}"
                        )

                    with c3:
                        st.metric(
                            "Desviación estándar",
                            f"{desviacion:,.2f}"
                        )

                    if media > mediana:
                        st.write(
                            f"**Interpretación:** en `{variable}`, "
                            "la media se encuentra por encima de la "
                            "mediana, lo que puede ser consistente "
                            "con una asimetría positiva."
                        )

                    elif media < mediana:
                        st.write(
                            f"**Interpretación:** en `{variable}`, "
                            "la media se encuentra por debajo de la "
                            "mediana, lo que puede ser consistente "
                            "con una asimetría negativa."
                        )

                    else:
                        st.write(
                            f"**Interpretación:** en `{variable}`, "
                            "la media y la mediana son similares, "
                            "por lo que la distribución presenta "
                            "mayor simetría alrededor del centro."
                        )

                st.divider()

                # --------------------------------------------------
                # Boxplots
                # --------------------------------------------------

                st.markdown("#### 📦 Boxplots")

                for variable in seleccionadas:

                    serie = datos_graficos[variable].dropna()

                    if serie.empty:
                        continue

                    fig, ax = plt.subplots(figsize=(10, 3))

                    sns.boxplot(
                        x=serie,
                        ax=ax
                    )

                    ax.set_title(
                        f"Boxplot de {variable}"
                    )
                    ax.set_xlabel(variable)

                    fig.tight_layout()
                    st.pyplot(fig)

                st.divider()

                # --------------------------------------------------
                # Comparación conjunta
                # --------------------------------------------------

                st.markdown(
                    "#### 🔍 Comparación de escalas"
                )

                st.dataframe(
                    datos_graficos[seleccionadas]
                    .describe()
                    .T
                    .round(2),
                    use_container_width=True
                )

                st.info(
                    "⚠️ Las variables pueden encontrarse en escalas "
                    "muy diferentes. Por ello, los boxplots se "
                    "presentan individualmente para evitar que una "
                    "variable con valores grandes oculte la "
                    "distribución de otra."
                )

                # --------------------------------------------------
                # Nota sobre TotalCharges
                # --------------------------------------------------

                if "TotalCharges" in seleccionadas:
                    st.warning(
                        "`TotalCharges` estaba almacenada originalmente "
                        "como texto (`object`). Para este gráfico se "
                        "realizó una conversión temporal a numérico "
                        "mediante `pd.to_numeric(..., errors='coerce')`. "
                        "El DataFrame original no se modifica en este "
                        "paso."
                    )

                st.divider()

                # --------------------------------------------------
                # Resumen automático
                # --------------------------------------------------

                st.markdown(
                    "#### 💡 Resumen del análisis"
                )

                st.write(
                    f"Se analizaron **{len(seleccionadas)} variable(s) "
                    "numérica(s)**. Los histogramas permiten estudiar "
                    "la forma de las distribuciones, mientras que los "
                    "boxplots facilitan la identificación visual de "
                    "posibles valores atípicos."
                )

    # ==========================================================
    # ÍTEM 6 - ANÁLISIS DE VARIABLES CATEGÓRICAS
    # ==========================================================

    with tabs[5]:

        st.subheader("6. Análisis de variables categóricas")

        st.markdown(
            """
            Se analizan las variables categóricas mediante tablas de
            frecuencia y gráficos de barras. El objetivo es conocer
            cómo se distribuyen las categorías y detectar posibles
            categorías dominantes o poco frecuentes.
            """
        )

        st.divider()

        _, categoricas = clasificar_variables(df)

        # Excluir identificadores si existen
        categoricas_analisis = [
            col for col in categoricas
            if col not in ["customerID"]
        ]

        if not categoricas_analisis:

            st.warning(
                "No se encontraron variables categóricas "
                "disponibles para analizar."
            )

        else:

            seleccion_cat = st.selectbox(
                "Seleccione una variable categórica",
                categoricas_analisis,
                key="eda_categorica_variable"
            )

            serie_cat = df[seleccion_cat].astype("string")

            frecuencia = (
                serie_cat
                .fillna("Valor faltante")
                .value_counts(dropna=False)
                .reset_index()
            )

            frecuencia.columns = [
                "Categoría",
                "Frecuencia"
            ]

            frecuencia["Porcentaje (%)"] = (
                frecuencia["Frecuencia"]
                / frecuencia["Frecuencia"].sum()
                * 100
            ).round(2)

            # --------------------------------------------------
            # Tabla de frecuencias
            # --------------------------------------------------

            st.markdown("#### 📋 Tabla de frecuencias")

            st.dataframe(
                frecuencia,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # --------------------------------------------------
            # Gráfico de barras
            # --------------------------------------------------

            st.markdown("#### 📊 Distribución de categorías")

            fig, ax = plt.subplots(figsize=(10, 5))

            sns.countplot(
                data=df.assign(
                    _categoria=serie_cat.fillna("Valor faltante")
                ),
                x="_categoria",
                order=frecuencia["Categoría"].tolist(),
                ax=ax
            )

            ax.set_title(
                f"Distribución de {seleccion_cat}"
            )
            ax.set_xlabel(seleccion_cat)
            ax.set_ylabel("Frecuencia")

            plt.xticks(
                rotation=45,
                ha="right"
            )

            fig.tight_layout()
            st.pyplot(fig)

            st.divider()

            # --------------------------------------------------
            # Categoría predominante
            # --------------------------------------------------

            categoria_mayor = frecuencia.iloc[0]

            st.markdown("#### 💡 Interpretación")

            st.write(
                f"La categoría más frecuente de **{seleccion_cat}** "
                f"es **{categoria_mayor['Categoría']}**, con "
                f"**{int(categoria_mayor['Frecuencia']):,} "
                f"registros**, equivalente al "
                f"**{categoria_mayor['Porcentaje (%)']:.2f}%** "
                "del total."
            )

            if len(frecuencia) == 2:

                diferencia = abs(
                    frecuencia.iloc[0]["Porcentaje (%)"]
                    - frecuencia.iloc[1]["Porcentaje (%)"]
                )

                if diferencia < 10:
                    st.info(
                        "Las dos categorías presentan una distribución "
                        "relativamente equilibrada."
                    )
                else:
                    st.info(
                        "Existe una diferencia apreciable en la "
                        "frecuencia de las dos categorías."
                    )

            elif len(frecuencia) > 2:

                porcentaje_mayor = categoria_mayor[
                    "Porcentaje (%)"
                ]

                if porcentaje_mayor >= 70:
                    st.warning(
                        "Una categoría concentra una proporción "
                        "elevada de los registros. Esto debe "
                        "considerarse al interpretar posteriormente "
                        "su relación con Churn."
                    )

            st.divider()

            # --------------------------------------------------
            # Comparación rápida de varias categóricas
            # --------------------------------------------------

            st.markdown(
                "#### 🔎 Resumen de variables categóricas"
            )

            resumen_categoricas = []

            for variable in categoricas_analisis:

                valores = (
                    df[variable]
                    .value_counts(dropna=False)
                )

                resumen_categoricas.append({
                    "Variable": variable,
                    "Categorías": df[variable].nunique(
                        dropna=True
                    ),
                    "Categoría más frecuente": (
                        str(valores.index[0])
                        if len(valores) > 0
                        else "Sin datos"
                    ),
                    "Frecuencia máxima": (
                        int(valores.iloc[0])
                        if len(valores) > 0
                        else 0
                    )
                })

            st.dataframe(
                pd.DataFrame(resumen_categoricas),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Este resumen permite identificar rápidamente "
                "variables con pocas categorías y aquellas donde "
                "una categoría concentra gran parte de los registros."
            )

    # ==========================================================
    # ÍTEM 7 - ANÁLISIS BIVARIADO: NUMÉRICA VS CHURN
    # ==========================================================

    with tabs[6]:

        st.subheader("7. Análisis bivariado: variables numéricas vs Churn")

        st.markdown(
            """
            En este apartado se estudia la relación entre las variables
            numéricas y **Churn**, comparando el comportamiento de los
            clientes que permanecen con aquellos que abandonan el servicio.

            Se utilizan gráficos de distribución y medidas estadísticas
            agrupadas por Churn para identificar diferencias entre ambos
            grupos.
            """
        )

        st.divider()

        datos_churn = preparar_datos_churn(df)

        if "Churn" not in datos_churn.columns:

            st.error(
                "No se encontró la variable objetivo `Churn` en el dataset."
            )

        else:

            numericas_churn = datos_churn.select_dtypes(
                include=np.number
            ).columns.tolist()

            if not numericas_churn:

                st.warning(
                    "No se encontraron variables numéricas para comparar "
                    "con Churn."
                )

            else:

                variable_num = st.selectbox(
                    "Seleccione una variable numérica",
                    numericas_churn,
                    key="eda_bivariado_numerica_churn"
                )

                # --------------------------------------------------
                # Distribución por Churn
                # --------------------------------------------------

                st.markdown(
                    "#### 📊 Distribución según Churn"
                )

                datos_plot = datos_churn[
                    [variable_num, "Churn"]
                ].dropna()

                fig, ax = plt.subplots(figsize=(10, 5))

                sns.histplot(
                    data=datos_plot,
                    x=variable_num,
                    hue="Churn",
                    kde=True,
                    bins=30,
                    element="step",
                    stat="density",
                    common_norm=False,
                    ax=ax
                )

                ax.set_title(
                    f"{variable_num} según Churn"
                )
                ax.set_xlabel(variable_num)
                ax.set_ylabel("Densidad")

                fig.tight_layout()
                st.pyplot(fig)

                st.divider()

                # --------------------------------------------------
                # Boxplot
                # --------------------------------------------------

                st.markdown(
                    "#### 📦 Comparación mediante boxplot"
                )

                fig2, ax2 = plt.subplots(figsize=(9, 5))

                sns.boxplot(
                    data=datos_plot,
                    x="Churn",
                    y=variable_num,
                    ax=ax2
                )

                ax2.set_title(
                    f"{variable_num} según condición de Churn"
                )
                ax2.set_xlabel("Churn")
                ax2.set_ylabel(variable_num)

                fig2.tight_layout()
                st.pyplot(fig2)

                st.divider()

                # --------------------------------------------------
                # Estadísticas agrupadas
                # --------------------------------------------------

                st.markdown(
                    "#### 📋 Estadísticas por grupo de Churn"
                )

                estadisticas_churn = (
                    datos_plot
                    .groupby("Churn")[variable_num]
                    .agg(
                        ["count", "mean", "median", "std", "min", "max"]
                    )
                    .round(2)
                    .reset_index()
                )

                estadisticas_churn.columns = [
                    "Churn",
                    "Cantidad",
                    "Media",
                    "Mediana",
                    "Desv. estándar",
                    "Mínimo",
                    "Máximo"
                ]

                st.dataframe(
                    estadisticas_churn,
                    use_container_width=True,
                    hide_index=True
                )

                st.divider()

                # --------------------------------------------------
                # Diferencia de medias
                # --------------------------------------------------

                st.markdown(
                    "#### 🔎 Comparación entre grupos"
                )

                grupos = (
                    datos_plot
                    .groupby("Churn")[variable_num]
                    .agg(["count", "mean", "median"])
                )

                if "Yes" in grupos.index and "No" in grupos.index:

                    media_no = grupos.loc["No", "mean"]
                    media_yes = grupos.loc["Yes", "mean"]

                    mediana_no = grupos.loc["No", "median"]
                    mediana_yes = grupos.loc["Yes", "median"]

                    diferencia_media = media_yes - media_no
                    diferencia_porcentual = (
                        diferencia_media / media_no * 100
                        if media_no != 0
                        else np.nan
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Media Churn = No",
                            f"{media_no:,.2f}"
                        )

                    with c2:
                        st.metric(
                            "Media Churn = Yes",
                            f"{media_yes:,.2f}"
                        )

                    with c3:
                        st.metric(
                            "Diferencia de medias",
                            f"{diferencia_media:,.2f}"
                        )

                    st.write(
                        f"La mediana para clientes con **Churn = No** "
                        f"es **{mediana_no:,.2f}**, mientras que para "
                        f"**Churn = Yes** es **{mediana_yes:,.2f}**."
                    )

                    if not np.isnan(diferencia_porcentual):

                        st.write(
                            f"La diferencia entre las medias representa "
                            f"aproximadamente un "
                            f"**{abs(diferencia_porcentual):.2f}%** "
                            f"respecto a la media del grupo Churn = No."
                        )

                    st.info(
                        "Esta comparación es descriptiva: una diferencia "
                        "entre grupos no implica por sí sola causalidad."
                    )

                else:

                    st.info(
                        "No se encontraron simultáneamente las categorías "
                        "`Yes` y `No` en Churn para realizar la comparación."
                    )

                st.divider()

                # --------------------------------------------------
                # Resumen de todas las variables numéricas
                # --------------------------------------------------

                st.markdown(
                    "#### 📈 Resumen de medias por Churn"
                )

                resumen_bivariado = []

                for variable in numericas_churn:

                    tmp = datos_churn[
                        [variable, "Churn"]
                    ].dropna()

                    medias = tmp.groupby("Churn")[variable].mean()

                    resumen_bivariado.append({
                        "Variable": variable,
                        "Media Churn = No": medias.get("No", np.nan),
                        "Media Churn = Yes": medias.get("Yes", np.nan),
                        "Diferencia (Yes - No)": (
                            medias.get("Yes", np.nan)
                            - medias.get("No", np.nan)
                        )
                    })

                resumen_bivariado = pd.DataFrame(
                    resumen_bivariado
                ).round(2)

                st.dataframe(
                    resumen_bivariado,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "La tabla resume las diferencias descriptivas de las "
                    "medias entre clientes con y sin Churn. No constituye "
                    "una prueba estadística de significancia."
                )

    # ==========================================================
    # ÍTEM 8 - ANÁLISIS BIVARIADO: CATEGÓRICA VS CHURN
    # ==========================================================

    with tabs[7]:

        st.subheader(
            "8. Análisis bivariado: variables categóricas vs Churn"
        )

        st.markdown(
            """
            En este apartado se analiza la relación entre las variables
            categóricas y **Churn**. Se utilizan frecuencias, porcentajes
            y gráficos para comparar la proporción de abandono entre
            las diferentes categorías.
            """
        )

        st.divider()

        if "Churn" not in df.columns:

            st.error(
                "No se encontró la variable objetivo `Churn` en el dataset."
            )

        else:

            _, categoricas = clasificar_variables(df)

            categoricas_churn = [
                col for col in categoricas
                if col not in ["customerID", "Churn"]
            ]

            if not categoricas_churn:

                st.warning(
                    "No se encontraron variables categóricas disponibles "
                    "para analizar frente a Churn."
                )

            else:

                variable_cat = st.selectbox(
                    "Seleccione una variable categórica",
                    categoricas_churn,
                    key="eda_bivariado_categorica_churn"
                )

                datos_bivariado = df[
                    [variable_cat, "Churn"]
                ].copy()

                datos_bivariado[variable_cat] = (
                    datos_bivariado[variable_cat]
                    .astype("string")
                    .fillna("Valor faltante")
                )

                datos_bivariado["Churn"] = (
                    datos_bivariado["Churn"]
                    .astype("string")
                    .fillna("Valor faltante")
                )

                # --------------------------------------------------
                # Tabla de contingencia
                # --------------------------------------------------

                st.markdown(
                    "#### 📋 Tabla de frecuencias por Churn"
                )

                tabla_frecuencia = pd.crosstab(
                    datos_bivariado[variable_cat],
                    datos_bivariado["Churn"]
                )

                st.dataframe(
                    tabla_frecuencia,
                    use_container_width=True
                )

                st.divider()

                # --------------------------------------------------
                # Porcentaje de Churn dentro de cada categoría
                # --------------------------------------------------

                st.markdown(
                    "#### 📊 Porcentaje de Churn por categoría"
                )

                tabla_porcentaje = pd.crosstab(
                    datos_bivariado[variable_cat],
                    datos_bivariado["Churn"],
                    normalize="index"
                ) * 100

                tabla_porcentaje = tabla_porcentaje.round(2)

                st.dataframe(
                    tabla_porcentaje,
                    use_container_width=True
                )

                # --------------------------------------------------
                # Gráfico de proporciones
                # --------------------------------------------------

                if "Yes" in tabla_porcentaje.columns:

                    st.markdown(
                        "#### 📈 Tasa de abandono por categoría"
                    )

                    tasa_churn = (
                        tabla_porcentaje["Yes"]
                        .sort_values(ascending=False)
                    )

                    fig, ax = plt.subplots(
                        figsize=(10, 5)
                    )

                    sns.barplot(
                        x=tasa_churn.values,
                        y=tasa_churn.index,
                        ax=ax
                    )

                    ax.set_title(
                        f"Tasa de Churn según {variable_cat}"
                    )
                    ax.set_xlabel("Churn (%)")
                    ax.set_ylabel(variable_cat)

                    fig.tight_layout()
                    st.pyplot(fig)

                    st.divider()

                    # --------------------------------------------------
                    # Categoría con mayor y menor Churn
                    # --------------------------------------------------

                    categoria_mayor = tasa_churn.idxmax()
                    valor_mayor = tasa_churn.max()

                    categoria_menor = tasa_churn.idxmin()
                    valor_menor = tasa_churn.min()

                    c1, c2 = st.columns(2)

                    with c1:
                        st.metric(
                            "Mayor tasa de Churn",
                            f"{valor_mayor:.2f}%"
                        )
                        st.write(
                            f"Categoría: **{categoria_mayor}**"
                        )

                    with c2:
                        st.metric(
                            "Menor tasa de Churn",
                            f"{valor_menor:.2f}%"
                        )
                        st.write(
                            f"Categoría: **{categoria_menor}**"
                        )

                    diferencia = valor_mayor - valor_menor

                    st.write(
                        f"La diferencia entre la categoría con mayor "
                        f"y menor tasa de abandono es de "
                        f"**{diferencia:.2f} puntos porcentuales**."
                    )

                else:

                    st.info(
                        "No existe la categoría `Churn = Yes` en los "
                        "datos seleccionados, por lo que no es posible "
                        "calcular una tasa de abandono."
                    )

                st.divider()

                # --------------------------------------------------
                # Distribución de Churn dentro de cada categoría
                # --------------------------------------------------

                st.markdown(
                    "#### 📊 Distribución de Churn"
                )

                fig2, ax2 = plt.subplots(
                    figsize=(10, 5)
                )

                sns.countplot(
                    data=datos_bivariado,
                    x=variable_cat,
                    hue="Churn",
                    ax=ax2
                )

                ax2.set_title(
                    f"Distribución de Churn según {variable_cat}"
                )
                ax2.set_xlabel(variable_cat)
                ax2.set_ylabel("Cantidad")

                plt.xticks(
                    rotation=45,
                    ha="right"
                )

                fig2.tight_layout()
                st.pyplot(fig2)

                st.divider()

                # --------------------------------------------------
                # Resumen de todas las variables categóricas
                # --------------------------------------------------

                st.markdown(
                    "#### 🔎 Resumen de Churn por variables categóricas"
                )

                resumen_categorico = []

                for variable in categoricas_churn:

                    temp = df[
                        [variable, "Churn"]
                    ].copy()

                    temp[variable] = (
                        temp[variable]
                        .astype("string")
                        .fillna("Valor faltante")
                    )

                    temp["Churn"] = (
                        temp["Churn"]
                        .astype("string")
                        .fillna("Valor faltante")
                    )

                    tabla = pd.crosstab(
                        temp[variable],
                        temp["Churn"],
                        normalize="index"
                    ) * 100

                    if "Yes" in tabla.columns and len(tabla) > 0:

                        mayor = tabla["Yes"].idxmax()

                        resumen_categorico.append({
                            "Variable": variable,
                            "Categoría con mayor Churn": str(mayor),
                            "Mayor tasa Churn (%)": (
                                tabla["Yes"].max()
                            )
                        })

                if not resumen_categorico.empty:

                    resumen_categorico = pd.DataFrame(
                        resumen_categorico
                    ).sort_values(
                        "Mayor tasa Churn (%)",
                        ascending=False
                    )

                    resumen_categorico[
                        "Mayor tasa Churn (%)"
                    ] = resumen_categorico[
                        "Mayor tasa Churn (%)"
                    ].round(2)

                    st.dataframe(
                        resumen_categorico,
                        use_container_width=True,
                        hide_index=True
                    )

                st.info(
                    "⚠️ Las diferencias observadas son descriptivas. "
                    "Para afirmar que existe una asociación estadísticamente "
                    "significativa sería necesario aplicar una prueba "
                    "estadística, como chi-cuadrado."
                )

    # ==========================================================
    # ÍTEM 9 - ANÁLISIS BASADO EN PARÁMETROS SELECCIONADOS
    # ==========================================================

    with tabs[8]:

        st.subheader(
            "9. Análisis basado en parámetros seleccionados"
        )

        st.markdown(
            """
            Este apartado permite realizar un análisis dinámico del
            dataset. El usuario puede seleccionar las variables que
            desea estudiar y definir el tipo de análisis mediante los
            controles interactivos de Streamlit.
            """
        )

        st.divider()

        # ----------------------------------------------------------
        # Preparación de variables
        # ----------------------------------------------------------

        datos_dinamicos = df.copy()

        if "TotalCharges" in datos_dinamicos.columns:
            datos_dinamicos["TotalCharges"] = pd.to_numeric(
                datos_dinamicos["TotalCharges"],
                errors="coerce"
            )

        variables_numericas, variables_categoricas = (
            clasificar_variables(datos_dinamicos)
        )

        # TotalCharges puede convertirse temporalmente en numérica
        if (
            "TotalCharges" in datos_dinamicos.columns
            and "TotalCharges" not in variables_numericas
        ):
            if datos_dinamicos["TotalCharges"].notna().sum() > 0:
                variables_numericas.append("TotalCharges")

        # ----------------------------------------------------------
        # Controles
        # ----------------------------------------------------------

        tipo_analisis = st.selectbox(
            "Seleccione el tipo de análisis",
            [
                "Distribución numérica",
                "Distribución categórica",
                "Variable vs Churn"
            ],
            key="eda_dinamico_tipo"
        )

        if tipo_analisis == "Distribución numérica":

            seleccion_dinamica = st.multiselect(
                "Seleccione una o más variables numéricas",
                variables_numericas,
                default=variables_numericas[:1],
                key="eda_dinamico_numericas"
            )

            if not seleccion_dinamica:

                st.info(
                    "Seleccione al menos una variable numérica."
                )

            else:

                st.markdown(
                    "#### 📊 Distribuciones seleccionadas"
                )

                for variable in seleccion_dinamica:

                    serie = datos_dinamicos[variable].dropna()

                    if serie.empty:
                        continue

                    fig, ax = plt.subplots(figsize=(9, 4))

                    sns.histplot(
                        serie,
                        bins=30,
                        kde=True,
                        ax=ax
                    )

                    ax.set_title(
                        f"Distribución de {variable}"
                    )
                    ax.set_xlabel(variable)
                    ax.set_ylabel("Frecuencia")

                    fig.tight_layout()
                    st.pyplot(fig)

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Media",
                            f"{serie.mean():,.2f}"
                        )

                    with c2:
                        st.metric(
                            "Mediana",
                            f"{serie.median():,.2f}"
                        )

                    with c3:
                        st.metric(
                            "Desv. estándar",
                            f"{serie.std():,.2f}"
                        )

        elif tipo_analisis == "Distribución categórica":

            seleccion_dinamica = st.multiselect(
                "Seleccione una o más variables categóricas",
                [
                    col for col in variables_categoricas
                    if col != "customerID"
                ],
                default=[],
                key="eda_dinamico_categoricas"
            )

            numero_categorias = st.slider(
                "Número máximo de categorías a mostrar",
                min_value=2,
                max_value=20,
                value=10,
                step=1,
                key="eda_dinamico_top_categorias"
            )

            if not seleccion_dinamica:

                st.info(
                    "Seleccione al menos una variable categórica."
                )

            else:

                for variable in seleccion_dinamica:

                    frecuencia = (
                        datos_dinamicos[variable]
                        .astype("string")
                        .fillna("Valor faltante")
                        .value_counts()
                        .head(numero_categorias)
                    )

                    fig, ax = plt.subplots(figsize=(9, 4))

                    sns.barplot(
                        x=frecuencia.values,
                        y=frecuencia.index,
                        ax=ax
                    )

                    ax.set_title(
                        f"Top {numero_categorias} categorías de {variable}"
                    )
                    ax.set_xlabel("Frecuencia")
                    ax.set_ylabel(variable)

                    fig.tight_layout()
                    st.pyplot(fig)

                    tabla = pd.DataFrame({
                        "Categoría": frecuencia.index,
                        "Frecuencia": frecuencia.values,
                        "Porcentaje (%)": (
                            frecuencia.values
                            / frecuencia.sum()
                            * 100
                        ).round(2)
                    })

                    st.dataframe(
                        tabla,
                        use_container_width=True,
                        hide_index=True
                    )

        else:

            if "Churn" not in datos_dinamicos.columns:

                st.error(
                    "No se encontró la variable `Churn` en el dataset."
                )

            else:

                tipo_variable = st.selectbox(
                    "Seleccione el tipo de variable a comparar con Churn",
                    [
                        "Numérica",
                        "Categórica"
                    ],
                    key="eda_dinamico_tipo_churn"
                )

                if tipo_variable == "Numérica":

                    seleccion_dinamica = st.multiselect(
                        "Seleccione variables numéricas",
                        variables_numericas,
                        default=variables_numericas[:1],
                        key="eda_dinamico_num_churn"
                    )

                    for variable in seleccion_dinamica:

                        temp = datos_dinamicos[
                            [variable, "Churn"]
                        ].dropna()

                        fig, ax = plt.subplots(figsize=(9, 4))

                        sns.boxplot(
                            data=temp,
                            x="Churn",
                            y=variable,
                            ax=ax
                        )

                        ax.set_title(
                            f"{variable} vs Churn"
                        )
                        ax.set_xlabel("Churn")
                        ax.set_ylabel(variable)

                        fig.tight_layout()
                        st.pyplot(fig)

                        st.dataframe(
                            temp.groupby("Churn")[variable]
                            .agg(
                                ["count", "mean", "median", "std"]
                            )
                            .round(2),
                            use_container_width=True
                        )

                else:

                    variables_cat_churn = [
                        col for col in variables_categoricas
                        if col not in ["customerID", "Churn"]
                    ]

                    seleccion_dinamica = st.multiselect(
                        "Seleccione variables categóricas",
                        variables_cat_churn,
                        default=variables_cat_churn[:1],
                        key="eda_dinamico_cat_churn"
                    )

                    for variable in seleccion_dinamica:

                        temp = datos_dinamicos[
                            [variable, "Churn"]
                        ].copy()

                        temp[variable] = (
                            temp[variable]
                            .astype("string")
                            .fillna("Valor faltante")
                        )

                        tabla = pd.crosstab(
                            temp[variable],
                            temp["Churn"],
                            normalize="index"
                        ) * 100

                        if "Yes" in tabla.columns:

                            tasa = (
                                tabla["Yes"]
                                .sort_values(ascending=False)
                            )

                            fig, ax = plt.subplots(
                                figsize=(9, 4)
                            )

                            sns.barplot(
                                x=tasa.values,
                                y=tasa.index,
                                ax=ax
                            )

                            ax.set_title(
                                f"Tasa de Churn según {variable}"
                            )
                            ax.set_xlabel("Churn (%)")
                            ax.set_ylabel(variable)

                            fig.tight_layout()
                            st.pyplot(fig)

                            st.dataframe(
                                tabla.round(2),
                                use_container_width=True
                            )

                        else:

                            st.info(
                                f"No existe la categoría "
                                f"`Churn = Yes` para {variable}."
                            )

        st.divider()

        st.markdown(
            "#### 💡 Interpretación"
        )

        st.write(
            """
            El análisis dinámico permite explorar diferentes variables
             de la aplicación. La selección de
            columnas mediante `multiselect`, el tipo de análisis mediante
            `selectbox` y el número de categorías mediante `slider`
            permiten adaptar el EDA a diferentes preguntas analíticas.
            """
        )

    # ==========================================================
    # ÍTEM 10 - HALLAZGOS CLAVE
    # ==========================================================

    with tabs[9]:

        st.subheader("10. Hallazgos clave del análisis exploratorio")

        st.markdown(
            """
            Esta sección resume los principales hallazgos obtenidos
            durante el análisis exploratorio de datos, priorizando
            aquellos relacionados con **Churn**.
            """
        )

        st.divider()

        if "Churn" not in df.columns:

            st.error(
                "No se encontró la variable `Churn`. "
                "No es posible generar los hallazgos asociados al abandono."
            )

        else:

            datos_hallazgos = df.copy()

            # Conversión temporal de TotalCharges
            if "TotalCharges" in datos_hallazgos.columns:
                datos_hallazgos["TotalCharges"] = pd.to_numeric(
                    datos_hallazgos["TotalCharges"],
                    errors="coerce"
                )

            # --------------------------------------------------
            # 1. Tasa general de Churn
            # --------------------------------------------------

            churn_counts = (
                datos_hallazgos["Churn"]
                .astype("string")
                .value_counts(dropna=False)
            )

            total_clientes = len(datos_hallazgos)

            churn_yes = int(churn_counts.get("Yes", 0))
            churn_no = int(churn_counts.get("No", 0))

            tasa_churn = (
                churn_yes / total_clientes * 100
                if total_clientes > 0
                else 0
            )

            st.markdown("#### 📌 Indicadores principales")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Total de clientes",
                    f"{total_clientes:,}"
                )

            with c2:
                st.metric(
                    "Clientes con Churn",
                    f"{churn_yes:,}"
                )

            with c3:
                st.metric(
                    "Tasa global de Churn",
                    f"{tasa_churn:.2f}%"
                )

            st.divider()

            # --------------------------------------------------
            # 2. Visualización resumen de Churn
            # --------------------------------------------------

            st.markdown("#### 📊 Visualización resumen")

            fig, ax = plt.subplots(figsize=(8, 5))

            etiquetas = ["Churn = No", "Churn = Yes"]
            valores = [churn_no, churn_yes]

            sns.barplot(
                x=etiquetas,
                y=valores,
                ax=ax
            )

            ax.set_title("Distribución global de Churn")
            ax.set_xlabel("Condición")
            ax.set_ylabel("Cantidad de clientes")

            for i, valor in enumerate(valores):
                ax.text(
                    i,
                    valor,
                    f"{valor:,}",
                    ha="center",
                    va="bottom"
                )

            fig.tight_layout()
            st.pyplot(fig)

            st.divider()

            # --------------------------------------------------
            # 3. Hallazgos de variables numéricas
            # --------------------------------------------------

            st.markdown(
                "#### 🔢 Hallazgos en variables numéricas"
            )

            numericas_hallazgos = datos_hallazgos.select_dtypes(
                include=np.number
            ).columns.tolist()

            resumen_numerico = []

            if "Churn" in datos_hallazgos.columns:

                for variable in numericas_hallazgos:

                    temp = datos_hallazgos[
                        [variable, "Churn"]
                    ].dropna()

                    medias = temp.groupby("Churn")[variable].mean()

                    if "Yes" in medias.index and "No" in medias.index:

                        media_no = medias["No"]
                        media_yes = medias["Yes"]

                        diferencia = media_yes - media_no

                        resumen_numerico.append({
                            "Variable": variable,
                            "Media Churn = No": media_no,
                            "Media Churn = Yes": media_yes,
                            "Diferencia": diferencia
                        })

            if not resumen_numerico.empty:

                resumen_numerico = pd.DataFrame(
                    resumen_numerico
                )

                resumen_numerico["Diferencia absoluta"] = (
                    resumen_numerico["Diferencia"].abs()
                )

                resumen_numerico = resumen_numerico.sort_values(
                    "Diferencia absoluta",
                    ascending=False
                )

                st.dataframe(
                    resumen_numerico.drop(
                        columns=["Diferencia absoluta"]
                    ).round(2),
                    use_container_width=True,
                    hide_index=True
                )

                principal_num = resumen_numerico.iloc[0]

                st.write(
                    f"Entre las variables numéricas analizadas, "
                    f"**{principal_num['Variable']}** presenta la mayor "
                    f"diferencia absoluta de medias entre clientes con "
                    f"y sin Churn: "
                    f"**{abs(principal_num['Diferencia']):,.2f}**."
                )

            else:

                st.info(
                    "No fue posible calcular diferencias de medias "
                    "para las variables numéricas."
                )

            st.divider()

            # --------------------------------------------------
            # 4. Hallazgos de variables categóricas
            # --------------------------------------------------

            st.markdown(
                "#### 🏷️ Hallazgos en variables categóricas"
            )

            _, categoricas_hallazgos = clasificar_variables(
                datos_hallazgos
            )

            categoricas_hallazgos = [
                col for col in categoricas_hallazgos
                if col not in ["customerID", "Churn"]
            ]

            resumen_categorico = []

            for variable in categoricas_hallazgos:

                temp = datos_hallazgos[
                    [variable, "Churn"]
                ].copy()

                temp[variable] = (
                    temp[variable]
                    .astype("string")
                    .fillna("Valor faltante")
                )

                temp["Churn"] = (
                    temp["Churn"]
                    .astype("string")
                    .fillna("Valor faltante")
                )

                tabla = pd.crosstab(
                    temp[variable],
                    temp["Churn"],
                    normalize="index"
                ) * 100

                if "Yes" in tabla.columns and len(tabla) > 0:

                    categoria = tabla["Yes"].idxmax()
                    tasa = tabla["Yes"].max()

                    resumen_categorico.append({
                        "Variable": variable,
                        "Categoría": str(categoria),
                        "Tasa Churn (%)": tasa
                    })

            if not resumen_categorico.empty:

                resumen_categorico = pd.DataFrame(
                    resumen_categorico
                ).sort_values(
                    "Tasa Churn (%)",
                    ascending=False
                )

                st.dataframe(
                    resumen_categorico.round(2),
                    use_container_width=True,
                    hide_index=True
                )

                principal_cat = resumen_categorico.iloc[0]

                st.write(
                    f"La combinación categórica con mayor tasa de "
                    f"abandono encontrada en el análisis corresponde a "
                    f"**{principal_cat['Variable']} = "
                    f"{principal_cat['Categoría']}**, con una tasa de "
                    f"Churn de **{principal_cat['Tasa Churn (%)']:.2f}%**."
                )

            else:

                st.info(
                    "No fue posible calcular tasas de Churn para "
                    "las variables categóricas."
                )

            st.divider()

            # --------------------------------------------------
            # 5. Resumen ejecutivo
            # --------------------------------------------------

            st.markdown("#### 🎯 5 conclusiones para la toma de decisiones")

            st.markdown(
                f"""
                **1. El nivel de Churn requiere atención prioritaria.**  
                La tasa global de abandono es de **{tasa_churn:.2f}%**,
                equivalente a **{churn_yes:,} clientes**.

                **Decisión:** establecer la reducción de Churn como un
                indicador principal para evaluar las estrategias de
                retención.
                """
            )

            if not resumen_categorico.empty:

                principal_cat = resumen_categorico.iloc[0]

                st.markdown(
                    f"""
                    **2. Existen segmentos con mayor riesgo de abandono.**  
                    El segmento **{principal_cat['Variable']} =
                    {principal_cat['Categoría']}** presenta la mayor tasa
                    de Churn identificada, con
                    **{principal_cat['Tasa Churn (%)']:.2f}%**.

                    **Decisión:** priorizar este segmento en las campañas
                    de retención y evaluar sus características antes de
                    aplicar acciones generales.
                    """
                )

            else:

                st.markdown(
                    """
                    **2. Las variables categóricas permiten segmentar
                    el riesgo de abandono.**

                    **Decisión:** utilizar las diferencias observadas
                    entre categorías para definir segmentos de clientes
                    y priorizar posteriormente los de mayor riesgo.
                    """
                )

            if not resumen_numerico.empty:

                principal_num = resumen_numerico.iloc[0]

                st.markdown(
                    f"""
                    **3. Las variables numéricas presentan diferencias
                    entre clientes con y sin Churn.**  
                    La variable **{principal_num['Variable']}** presenta
                    la mayor diferencia absoluta de medias, de
                    **{abs(principal_num['Diferencia']):,.2f}**.

                    **Decisión:** incorporar esta variable en la
                    segmentación y profundizar su comportamiento antes
                    de definir políticas de retención.
                    """
                )

            else:

                st.markdown(
                    """
                    **3. Las variables numéricas deben incorporarse a la
                    segmentación del riesgo.**

                    **Decisión:** complementar este análisis con métricas
                    de comportamiento para priorizar acciones de
                    retención.
                    """
                )

            st.markdown(
                """
                **4. La estrategia de retención debe ser segmentada.**  
                Las diferencias observadas muestran que el comportamiento
                de los clientes no es homogéneo.

                **Decisión:** diseñar acciones diferenciadas según las
                características del cliente y su nivel de riesgo,
                evitando una estrategia única para toda la cartera.
                """
            )

            st.markdown(
                """
                **5. Los hallazgos del EDA deben servir como base para
                analítica predictiva.**  
                El análisis exploratorio permite identificar variables
                y segmentos relevantes, pero no demuestra causalidad.

                **Decisión:** complementar estos resultados con pruebas
                estadísticas y desarrollar posteriormente un modelo
                predictivo de Churn que permita priorizar clientes con
                mayor probabilidad de abandono.
                """
            )

            st.success(
                """
                **Prioridad de negocio:** identificar segmentos de alto
                riesgo, focalizar las acciones de retención y medir
                continuamente la evolución de la tasa de Churn para
                evaluar el impacto de las decisiones implementadas.
                """
            )

    # ==========================================================
    # FIN DEL EDA
    # ==========================================================



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
