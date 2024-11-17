import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset y procesarlo
@st.cache_data
def load_data():
    # Cargar los datos desde un archivo CSV
    df = pd.read_csv('Datos_postlimpieza.csv')

    # Crear un LabelEncoder para las variables categóricas
    label_encoder = LabelEncoder()
    # Aplicar LabelEncoder a las columnas categóricas, excepto "Ubicación"
    df['Puesto'] = label_encoder.fit_transform(df['Puesto'])
    df['Expertise'] = label_encoder.fit_transform(df['Expertise'])
    df['Servicios'] = label_encoder.fit_transform(df['Servicios'])
    df['Habilidades'] = label_encoder.fit_transform(df['Habilidades'])
    df['Herramientas'] = label_encoder.fit_transform(df['Herramientas'])
    df['Educación'] = label_encoder.fit_transform(df['Educación'])

    # Eliminar columnas no necesarias para el modelo
    df = df.drop(columns=['Título', 'Empresa', 'Modalidad', 'Sector', 'Descripción', 'Otro Idioma', 'EntornoTEC', 'Beneficios'])

    return df, label_encoder

# Función para borrar el caché y actualizar los datos
def clear_data_cache():
    st.cache_data.clear_cache()
    st.success("¡Caché borrado con éxito! Los datos se actualizarán en la siguiente carga.")

# Cargar los datos
df, label_encoder = load_data()

# Seleccionar las columnas relevantes para el modelo
feature_columns = ['Expertise', 'Ubicación', 'Servicios', 'Habilidades', 'Herramientas', 'Educación']
X = df[feature_columns]  # Variables predictoras
y = df['Puesto']  # Variable objetivo

# Dividir en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Crear un LabelEncoder para "Ubicación"
ubicacion_encoder = LabelEncoder()
X_train['Ubicación'] = ubicacion_encoder.fit_transform(X_train['Ubicación'])
X_test['Ubicación'] = ubicacion_encoder.transform(X_test['Ubicación'])

# Escalar las características (normalización)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Ajustar y transformar el conjunto de entrenamiento
X_test_scaled = scaler.transform(X_test)        # Transformar el conjunto de prueba

# Crear y entrenar el modelo KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)

# Configuración de la app de Streamlit
st.title("Clasificador de Puesto de Trabajo")

# Mostrar datos crudos (opcional)
if st.checkbox("Mostrar datos crudos"):
    st.write(df)

# Interfaz de usuario para realizar predicciones
st.subheader("Hacer una predicción")

# Crear selectboxes para que el usuario seleccione valores para hacer una predicción
expertise = st.selectbox("Seleccione su nivel de Expertise", df['Expertise'].unique())
ubicacion = st.selectbox("Seleccione su Ubicación", df['Ubicación'].unique())
servicios = st.selectbox("Seleccione los Servicios", df['Servicios'].unique())
habilidades = st.selectbox("Seleccione sus Habilidades", df['Habilidades'].unique())
herramientas = st.selectbox("Seleccione las Herramientas", df['Herramientas'].unique())
educacion = st.selectbox("Seleccione su nivel de Educación", df['Educación'].unique())

# Convertir los valores seleccionados en un array numérico para hacer la predicción
input_data = np.array([[expertise, ubicacion, servicios, habilidades, herramientas, educacion]])

# Codificar la "Ubicación" seleccionada en un número utilizando el LabelEncoder
ubicacion_encoded = ubicacion_encoder.transform([ubicacion])[0]
input_data[0][1] = ubicacion_encoded  # Sustituir la ubicación con el valor codificado

# Escalar la entrada del usuario (aplicando el mismo escalado que a los datos)
input_data_scaled = scaler.transform(input_data)

# Realizar la predicción
if st.button("Predecir Puesto"):
    # Predecir el puesto de trabajo con el modelo entrenado
    prediction = model.predict(input_data_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    st.write(f"El puesto de trabajo predicho es: {predicted_label}")

# Evaluación del modelo
if st.button("Evaluar modelo"):
    y_pred = model.predict(X_test_scaled)

    # Evaluar precisión
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Precisión del modelo: {accuracy:.2f}")

    # Extraer etiquetas presentes en y_test
    labels_present = np.unique(y_test)
    target_names = label_encoder.inverse_transform(labels_present)

    # Reporte de clasificación
    st.subheader("Reporte de clasificación:")
    classification_rep = classification_report(
        y_test,
        y_pred,
        labels=labels_present,
        target_names=target_names
    )
    st.text(classification_rep)

    # Mostrar matriz de confusión como un gráfico
    st.subheader("Matriz de Confusión:")
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels_present)

    # Usar seaborn para mejorar la visualización de la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    st.pyplot(plt)

    # Mostrar la distribución de las clases predichas vs reales
    st.subheader("Distribución de las clases predichas vs reales:")
    results = pd.DataFrame({"Real": y_test, "Predicción": y_pred})
    st.write(results)

    # Graficar la distribución de las predicciones
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Real', data=results, palette='viridis')
    st.pyplot(plt)

# Análisis de outliers en "sueldo_medio"
st.subheader("Análisis de Outliers en Sueldo Medio")

# Paso 1: Limpiar y convertir los valores de "sueldo_medio" a valores numéricos
df['sueldo_medio'] = pd.to_numeric(df['sueldo_medio'].replace('[^0-9]', '', regex=True), errors='coerce')

# Paso 2: Calcular los límites de los outliers usando Tukey's Fence
q1 = df['sueldo_medio'].quantile(0.25)
q3 = df['sueldo_medio'].quantile(0.75)
iqr = q3 - q1

# Definir los límites superior e inferior
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

# Identificar los sueldos que están fuera de estos límites
df['Outlier'] = (df['sueldo_medio'] < limite_inferior) | (df['sueldo_medio'] > limite_superior)

# Mostrar estadísticas clave
st.write(f"**Primer Cuartil (Q1):** {q1}")
st.write(f"**Tercer Cuartil (Q3):** {q3}")
st.write(f"**IQR:** {iqr}")
st.write(f"**Límite inferior:** {limite_inferior}")
st.write(f"**Límite superior:** {limite_superior}")
st.write(f"**Número de Outliers:** {df['Outlier'].sum()}")

# Paso 3: Visualizar los datos con un gráfico
st.subheader("Distribución de Sueldos con Outliers Identificados")
plt.figure(figsize=(10, 6))
sns.histplot(df, x="sueldo_medio", hue="Outlier", palette={False: "blue", True: "red"}, bins=30, kde=True)
plt.title('Distribución de Sueldos Medios con Identificación de Outliers')
plt.xlabel('Sueldo Medio')
plt.ylabel('Frecuencia')

# Mostrar el gráfico en Streamlit
st.pyplot(plt)
