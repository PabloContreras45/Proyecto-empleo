import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Configuración de página
st.set_page_config(
    page_title="Análisis de Vacantes y Predicción",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.image('analysis.jpg', use_container_width=True)

# Cargar datos
@st.cache_data
def cargar_datos(filepath):
    df = pd.read_csv(filepath)
    return df

datos = cargar_datos("datos_pre_analisis.csv")

# Preprocesamiento de datos para el modelo de predicción
@st.cache_data
def preprocesar_datos(df):
    # Usar LabelEncoder para transformar las columnas categóricas
    clase_LabelEncoder = LabelEncoder()
    df["Puesto"] = clase_LabelEncoder.fit_transform(df["Puesto"])
    df["Expertise"] = clase_LabelEncoder.fit_transform(df["Expertise"])
    df["Ubicación"] = clase_LabelEncoder.fit_transform(df["Ubicación"])
    df["Servicios"] = clase_LabelEncoder.fit_transform(df["Servicios"])
    df["Habilidades"] = clase_LabelEncoder.fit_transform(df["Habilidades"])
    df["Herramientas"] = clase_LabelEncoder.fit_transform(df["Herramientas"])
    df["Educación"] = clase_LabelEncoder.fit_transform(df["Educación"])

    # Eliminar columnas innecesarias
    columnas_a_eliminar = ["Título", "Empresa", "Modalidad", "Sector", "Descripción", "Otro Idioma", "EntornoTEC", "Beneficios"]
    columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
    df = df.drop(columns=columnas_existentes)
    
    return df

# Preprocesamos los datos para el modelo
df = preprocesar_datos(datos)

# Preparar datos para el modelo de predicción
X = np.array(df.drop("Puesto", axis=1))
y = np.array(df["Puesto"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

x_scaler = MinMaxScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predicción y precisión del modelo
yhat = model.predict(X_test)
accuracy = accuracy_score(y_test, yhat)

# Barra lateral: Filtros
st.sidebar.title("Filtros Avanzados")

# Filtro por Puesto
with st.sidebar.expander("Filtrar por Puesto", expanded=True):
    puestos_filtrados = st.multiselect(
        "Selecciona Puestos de Trabajo",
        options=datos["Puesto"].unique(),
        default=datos["Puesto"].unique()[:5],
    )

# Filtro por Ubicación
with st.sidebar.expander("Filtrar por Ubicación", expanded=True):
    ubicaciones_filtradas = st.multiselect(
        "Selecciona Ubicaciones",
        options=datos["Ubicación"].unique(),
        default=datos["Ubicación"].unique()[:5],
    )

# Filtro por Rango de Salario
with st.sidebar.expander("Filtrar por Rango de Salario", expanded=True):
    salario_min, salario_max = st.slider(
        "Selecciona el rango de salario medio",
        min_value=int(datos["sueldo_medio"].min()),
        max_value=int(datos["sueldo_medio"].max()),
        value=(int(datos["sueldo_medio"].min()), int(datos["sueldo_medio"].max())),
        step=500,
    )

# Filtro por Nivel de Experiencia
with st.sidebar.expander("Filtrar por Nivel de Experiencia", expanded=True):
    niveles_experiencia = datos["Expertise"].unique()
    experiencia_filtrada = st.multiselect(
        "Nivel de Experiencia",
        options=niveles_experiencia,
        default=niveles_experiencia[:3],
    )

# Aplicar los filtros
datos_filtrados = datos[
    (datos["Puesto"].isin(puestos_filtrados)) &
    (datos["Ubicación"].isin(ubicaciones_filtradas)) &
    (datos["sueldo_medio"].between(salario_min, salario_max)) &
    (datos["Expertise"].isin(experiencia_filtrada))
]

# Páginas de Streamlit
pagina = st.sidebar.radio(
    "Selecciona una página",
    ["Análisis General", "Gráficos Específicos", "Predicción de Puestos"],
)

# Página 1: Análisis General
if pagina == "Análisis General":
    st.title("📊 Análisis General de Vacantes")
    st.markdown("### Resumen del dataset filtrado")

    col1, col2 = st.columns([2, 1])

    # Gráfico 1: Distribución de sueldos
    fig_salarios = px.histogram(
        datos_filtrados, 
        x="sueldo_medio", 
        nbins=30, 
        title="Distribución de Sueldos Medios",
        color_discrete_sequence=["#1f77b4"]
    )
    col1.plotly_chart(fig_salarios, use_container_width=True)
    col1.caption("La mayoría de los sueldos medios se encuentran concentrados en un rango específico.")

    # Gráfico 2: Vacantes por modalidad
    modalidad_df = datos_filtrados["Modalidad"].value_counts().reset_index()
    modalidad_df.columns = ["Modalidad", "count"]
    fig_modalidad = px.pie(
        modalidad_df, 
        names="Modalidad", 
        values="count",
        title="Modalidad de Trabajo",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    col2.plotly_chart(fig_modalidad, use_container_width=True)
    col2.caption("Este gráfico muestra las modalidades de trabajo más comunes.")

# Página 2: Gráficos Específicos
elif pagina == "Gráficos Específicos":
    st.title("📈 Gráficos Específicos")
    st.markdown("### Análisis detallado por puesto y empresa")

    col1, col2 = st.columns(2)

    # Gráfico: Puestos más comunes
    puestos_df = datos_filtrados["Puesto"].value_counts().reset_index()
    puestos_df.columns = ["Puesto", "count"]
    fig_puestos = px.bar(
        puestos_df, 
        x="Puesto", 
        y="count", 
        title="Vacantes por Puesto",
        color_discrete_sequence=["#2ca02c"]
    )
    col1.plotly_chart(fig_puestos, use_container_width=True)
    col1.caption("Estos son los puestos con más ofertas disponibles.")

    # Gráfico: Empresas con más vacantes
    empresas_df = datos_filtrados["Empresa"].value_counts().head(10).reset_index()
    empresas_df.columns = ["Empresa", "count"]
    fig_empresas = px.bar(
        empresas_df, 
        x="Empresa", 
        y="count", 
        title="Top 10 Empresas por Vacantes",
        color_discrete_sequence=["#ff7f0e"]
    )
    col2.plotly_chart(fig_empresas, use_container_width=True)
    col2.caption("Las empresas más activas publicando ofertas.")

# Página 3: Predicción de Puestos
# Definir las características de entrada
caracteristicas_esperadas = [
    "Expertise", "Ubicación", "Servicios", "Habilidades", 
    "Herramientas", "Educación", "sueldo_medio", "otra_caracteristica_1", "otra_caracteristica_2"
]

# Asegurémonos de que las entradas del usuario coincidan con las características del modelo
# Lista de ciudades de España para el selector
ciudades = ["Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza", "Málaga", "Murcia", "Palma", "Las Palmas", "Bilbao", "Alicante", "Córdoba", "Valladolid", "Vigo", "Gijón"]

# Esto debería haberse hecho previamente al entrenar el modelo, pero por si no está definido:
clase_LabelEncoder = LabelEncoder()

# Suponiendo que 'datos["Puesto"]' es la columna con las clases que se usan en la predicción de puestos:
clase_LabelEncoder.fit(datos["Puesto"])  # Codifica las clases de los puestos

# Definir las opciones para el selector de nivel de experiencia y otros
nivel_expertise = ["Principiante", "Intermedio", "Avanzado", "Experto"]
nivel_ubicacion = ciudades  # Usamos la lista de ciudades para las ubicaciones

# Características de entrada que se deben modificar con palabras
if pagina == "Predicción de Puestos":
    st.title("🔮 Predicción de Puestos")
    st.markdown("### Introduce los valores para predecir el puesto más probable")

    # Entradas para cada característica esperada con palabras
    expertise = st.selectbox("Nivel de Expertise", options=nivel_expertise)
    ubicacion = st.selectbox("Ubicación", options=ciudades)
    servicios = st.selectbox("Beneficios", options=datos["Servicios"].unique())
    habilidades = st.selectbox("Habilidades", options=datos["Habilidades"].unique())
    herramientas = st.selectbox("Herramientas", options=datos["Herramientas"].unique())
    educacion = st.selectbox("Nivel de Educación", options=datos["Educación"].unique())
    sueldo_medio = st.number_input("Sueldo Medio", min_value=20000, max_value=200000, value=30000)

    # Asegúrate de que las entradas no sean `NaN` y estén definidas
    def get_index(value, options_list):
        # Usamos np.where() para encontrar el índice
        result = np.where(options_list == value)[0]
        if result.size > 0:
            return result[0]  # Si encontramos el valor, devolvemos el primer índice
        else:
            return -1  # En caso contrario, devolvemos -1

    # Realizar predicción y mostrar resultados
    if st.button("Predecir Puesto"):
        # Convertir las entradas en los valores que espera el modelo
        inputs = {
            "Expertise": get_index(expertise, np.array(nivel_expertise)),
            "Ubicación": get_index(ubicacion, np.array(nivel_ubicacion)),
            "Servicios": get_index(servicios, np.array(datos["Servicios"].unique())),
            "Habilidades": get_index(habilidades, np.array(datos["Habilidades"].unique())),
            "Herramientas": get_index(herramientas, np.array(datos["Herramientas"].unique())),
            "Educación": get_index(educacion, np.array(datos["Educación"].unique())),
            "sueldo_medio": sueldo_medio
        }

        # Si alguno de los valores es -1, significa que no fue seleccionado correctamente
        if -1 in inputs.values():
            st.error("Por favor, asegúrate de seleccionar opciones válidas para todas las características.")
        else:
            # Asegurarse de que estamos pasando todas las características necesarias (9 en total)
            # Agregar valores para las características faltantes
            inputs_completos = np.array([
                inputs["Expertise"], inputs["Ubicación"], inputs["Servicios"], 
                inputs["Habilidades"], inputs["Herramientas"], inputs["Educación"], 
                inputs["sueldo_medio"], 0, 0  # Las características faltantes que se necesitan
            ])

            # Escalar las entradas
            inputs_completos_scaled = x_scaler.transform([inputs_completos])

            # Obtener las probabilidades de las clases
            probabilidad = model.predict_proba(inputs_completos_scaled)

            # Mostrar las probabilidades en porcentaje para cada clase
            clases = clase_LabelEncoder.classes_  # Las clases disponibles (puestos)
            probabilidad = probabilidad[0] * 100  # Convertir a porcentaje

            # Mostrar el puesto más probable y sus probabilidades
            st.write("El puesto más probable es:", clase_LabelEncoder.inverse_transform([np.argmax(probabilidad)])[0])
            st.write("Probabilidades de cada puesto:")

            for i, clase in enumerate(clases):
                st.write(f"{clase}: {probabilidad[i]:.2f}%")