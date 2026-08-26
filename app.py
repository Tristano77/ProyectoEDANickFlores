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
                variables_numericas
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
            "Mostrar registros que contienen valores faltantes"
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
                default=numericas_graficos[:1]
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
    # ÍTEMS PENDIENTES
    # ==========================================================

    nombres_pendientes = [
        "Análisis de variables categóricas",
        "Análisis bivariado: numérico vs Churn",
        "Análisis bivariado: categórico vs Churn",
        "Análisis basado en parámetros seleccionados",
        "Hallazgos clave"
    ]

    for i in range(5, 10):
        with tabs[i]:
            st.subheader(
                f"{i + 1}. {nombres_pendientes[i - 5]}"
            )
            st.info(
                "Este ítem se implementará en el siguiente paso."
            )
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
            "Mostrar registros que contienen valores faltantes"
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
    # ÍTEMS PENDIENTES
    # ==========================================================

    nombres_pendientes = [
        "Distribución de variables numéricas",
        "Análisis de variables categóricas",
        "Análisis bivariado: numérico vs Churn",
        "Análisis bivariado: categórico vs Churn",
        "Análisis basado en parámetros seleccionados",
        "Hallazgos clave"
    ]

    for i in range(4, 10):
        with tabs[i]:
            st.subheader(
                f"{i + 1}. {nombres_pendientes[i - 4]}"
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
