import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Vacantes y Predicción",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Cargar los datos
df = pd.read_csv('datos_pre_analisis.csv')


# Función para preprocesar los datos
@st.cache_data
def preprocesar_datos(df):
    if df.empty:
        return df
    
    # Transformación con LabelEncoder
    clase_LabelEncoder = LabelEncoder()
    columnas_categ = ["Puesto", "Expertise", "Ubicación", "Servicios", "Habilidades", "Herramientas", "Educación"]
    for col in columnas_categ:
        if col in df.columns:
            df[col] = clase_LabelEncoder.fit_transform(df[col])
    
    # Eliminar columnas innecesarias
    columnas_a_eliminar = ["Título", "Empresa", "Modalidad", "Sector", "Descripción", "Otro Idioma", "EntornoTEC", "Beneficios"]
    df = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns], errors='ignore')
    return df

# Página principal
def pagina_principal():
    st.image("analysis.jpg", use_container_width=True)  # Asegúrate de poner la ruta correcta
    st.title("📊 Análisis de Vacantes y Predicción de Puestos")
    st.write("""
        Bienvenido a la aplicación de análisis de vacantes. Aquí puedes explorar los datos, 
        aplicar filtros avanzados y realizar predicciones sobre el puesto de trabajo más probable.
    """)
    
    # Usar HTML para centrar el texto de los actores y sus links de LinkedIn
    st.markdown("""
        <div style="text-align: center;">
            <h3>Actores del proyecto:</h3>
            <ol>
                <li>
                    <p>Abdelkader El yagoubi El mahdi</p>
                    <p><a href="https://www.linkedin.com/in/abdelkader-elyagoubi-elmahdi/" target="_blank">LinkedIn de Abdelkader El yagoubi El mahdi</a></p>
                </li>
                <li>
                    <p>Pablo Contreras Evangelista</p>
                    <p><a href="https://www.linkedin.com/in/pablo-contreras-evangelista/" target="_blank">LinkedIn de Pablo Contreras Evangelista</a></p>
                </li>
                <li>
                    <p>Carmen Rodríguez Pando</p>
                    <p><a href="https://www.linkedin.com/in/carmen-rodriguez-pando" target="_blank">LinkedIn de Carmen Rodríguez Pando</a></p>
                </li>
            </ol>
        </div>
    """, unsafe_allow_html=True)


# Página de exploración de datos
def pagina_exploracion(df):
    st.header("📋 Exploración de Datos")
    st.write("Visualiza los datos y aplica los filtros seleccionados en la barra lateral.")
    
    # Asegúrate de que las columnas "Puesto" y "Ubicación" sean de tipo texto
    if "Puesto" in df.columns:
        df["Puesto"] = df["Puesto"].astype(str)
    if "Ubicación" in df.columns:
        df["Ubicación"] = df["Ubicación"].astype(str)
    
    # Filtro por Puesto de trabajo: Mostrar nombres completos de los puestos (no números)
    puestos_filtrados = st.sidebar.selectbox(
        "Selecciona Puesto de Trabajo",
        options=df["Puesto"].unique(),
        index=0,  # valor predeterminado
    )
    
    # Filtro por Ubicación: Mostrar nombres completos de las ubicaciones (no números)
    ubicaciones_filtradas = st.sidebar.selectbox(
        "Selecciona Ubicación",
        options=df["Ubicación"].unique(),
        index=0,  # valor predeterminado
    )
    
    # Filtro por sueldo medio (presentarlo de forma legible en palabras)
    if "sueldo_medio" in df.columns:
        salario_min, salario_max = st.sidebar.slider(
            "Selecciona el rango de salario medio",
            min_value=int(df["sueldo_medio"].min()),
            max_value=int(df["sueldo_medio"].max()),
            value=(int(df["sueldo_medio"].min()), int(df["sueldo_medio"].max())),
            step=500,
        )
        
        # Mostrar el rango de salario como texto legible
        st.sidebar.write(f"Salario medio entre {salario_min} y {salario_max} EUR")
    
    # Aplicar filtros
    df_filtrado = df.copy()
    
    # Aplicar filtro de Puesto de trabajo
    if 'Puesto' in df.columns and puestos_filtrados:
        df_filtrado = df_filtrado[df_filtrado["Puesto"] == puestos_filtrados]
    
    # Aplicar filtro de Ubicación
    if 'Ubicación' in df.columns and ubicaciones_filtradas:
        df_filtrado = df_filtrado[df_filtrado["Ubicación"] == ubicaciones_filtradas]
    
    # Aplicar filtro de rango de salario
    if "sueldo_medio" in df.columns:
        df_filtrado = df_filtrado[ 
            (df_filtrado["sueldo_medio"] >= salario_min) & 
            (df_filtrado["sueldo_medio"] <= salario_max)
        ]
    
    # Mostrar el DataFrame filtrado
    if not df_filtrado.empty:
        st.dataframe(df_filtrado)
    else:
        st.warning("No hay datos disponibles con los filtros seleccionados.")

        
# Página de visualización de datos
def pagina_visualizacion(df):
    st.header("📊 Visualización de Datos")
    st.write("Explora las tendencias en los datos mediante gráficos interactivos.")
    
    grafico = st.selectbox("Selecciona el tipo de gráfico", ["Barras", "Dispersión", "Cajas"])
    
    if grafico == "Barras" and "Ubicación" in df.columns and "sueldo_medio" in df.columns:
        fig = px.bar(
            df,
            x="Ubicación",
            y="sueldo_medio",
            color="Puesto",
            title="Sueldo Medio por Ubicación"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif grafico == "Dispersión" and "Educación" in df.columns and "sueldo_medio" in df.columns:
        fig = px.scatter(
            df,
            x="Educación",
            y="sueldo_medio",
            color="Puesto",
            title="Educación vs Sueldo Medio"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif grafico == "Cajas" and "Puesto" in df.columns and "sueldo_medio" in df.columns:
        fig = px.box(
            df,
            x="Puesto",
            y="sueldo_medio",
            color="Puesto",
            title="Distribución del Sueldo por Puesto"
        )
        st.plotly_chart(fig, use_container_width=True)

# Página de predicción
def pagina_prediccion():
    st.header("🔮 Predicción de Puestos")
    st.markdown("### Introduce los valores para predecir el puesto más probable")
    
    # Datos de entrada para la predicción
    nivel_expertise = ["Principiante", "Intermedio", "Avanzado", "Experto"]
    ciudades = ["Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza", "Málaga", "Murcia", "Palma", "Las Palmas", "Bilbao", "Alicante", "Córdoba", "Valladolid", "Vigo", "Gijón"]
    
    expertise = st.selectbox("Nivel de Expertise", options=nivel_expertise)
    ubicacion = st.selectbox("Ubicación", options=ciudades)
    servicios = st.selectbox("Beneficios", options=["Seguro médico", "Transporte", "Formación"])
    habilidades = st.selectbox("Habilidades", options=["Análisis", "Gestión", "Desarrollo"])
    herramientas = st.selectbox("Herramientas", options=["Excel", "Python", "SQL"])
    educacion = st.selectbox("Nivel de Educación", options=["Bachillerato", "Grado", "Máster"])
    sueldo_medio = st.number_input("Sueldo Medio", min_value=20000, max_value=200000, value=30000)
    
    if st.button("Predecir Puesto"):
        # Creación de un diccionario con los valores de entrada
        inputs = {
            "Nivel de Expertise": expertise,
            "Ubicación": ubicacion,
            "Servicios": servicios,
            "Habilidades": habilidades,
            "Herramientas": herramientas,
            "Nivel de Educación": educacion,
            "Sueldo Medio": sueldo_medio
        }
        
        # Mostrar los resultados de entrada en formato horizontal
        st.write("### Detalles de los valores introducidos:")
        st.table(pd.DataFrame([inputs]).T)


# Función para la página de Power BI
def pagina_powerbi():
    st.header("📊 Panel de Power BI - Análisis de Vacantes")
    powerbi_width = 600
    powerbi_height = 373.5
    
    st.markdown(f'''
        <iframe title="03-Proyecto_empleo_BI" width="{powerbi_width}" height="{powerbi_height}" 
        src="https://app.powerbi.com/view?r=eyJrIjoiYzM4ZTI1NGUtYWQwOS00MzI3LWIyNjMtY2EzY2IxNmMyZDdlIiwidCI6IjVlNzNkZTM1LWU4MjUtNGVkNS1iZTIyLTg4NTYzNTI3MDkxZSIsImMiOjl9" 
        frameborder="0" allowFullScreen="true"></iframe>
    ''', unsafe_allow_html=True)

# Configurar las páginas con el menú de navegación
page = st.sidebar.selectbox(
    "Selecciona una Página",
    options=["Página Principal", "Exploración de Datos", "Visualización de Datos", "Predicción", "Power BI"]
)

# Mostrar las páginas correspondientes
if page == "Página Principal":
    pagina_principal()
elif page == "Exploración de Datos":
    pagina_exploracion(df)
elif page == "Visualización de Datos":
    pagina_visualizacion(df)
elif page == "Predicción":
    pagina_prediccion()
elif page == "Power BI":
    pagina_powerbi()