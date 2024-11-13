import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar el dataset y procesarlo
@st.cache_data
def load_data():
    df = pd.read_csv('Datos_postlimpieza.csv')

    # Creamos el enconder para transformar en números las variables categóricas
    clase_LabelEncoder = LabelEncoder()
    df['Puesto'] = clase_LabelEncoder.fit_transform(df["Puesto"])
    df['Expertise'] = clase_LabelEncoder.fit_transform(df["Expertise"])
    df['Ubicación'] = clase_LabelEncoder.fit_transform(df["Ubicación"])
    df['Servicios'] = clase_LabelEncoder.fit_transform(df["Servicios"])
    df['Habilidades'] = clase_LabelEncoder.fit_transform(df["Habilidades"])
    df['Herramientas'] = clase_LabelEncoder.fit_transform(df["Herramientas"])
    df['Educación'] = clase_LabelEncoder.fit_transform(df["Educación"])

    # Eliminamos las columnas con una alta presencia de NaNs o innecesarias
    df = df.drop(columns=['Título','Empresa','Modalidad','Sector','Descripción','Otro Idioma','EntornoTEC','Beneficios'])
    return df, clase_LabelEncoder

# Función para borrar el caché y actualizar los datos
def clear_data_cache():
    st.cache_data.clear_cache()  # Borra el caché de la función load_data
    st.success("¡Caché borrado con éxito! Los datos se actualizarán en la siguiente carga.")

# Función para realizar las predicciones
def make_predictions(model, scaler, X_test, y_test, clase_LabelEncoder):
    yhat = model.predict(X_test)

    # Evaluamos la precisión
    accuracy = accuracy_score(y_test, yhat)
    
    # Reporte de clasificación
    classification_rep = classification_report(y_test, yhat, target_names=clase_LabelEncoder.classes_)
    
    # Matriz de confusión
    conf_matrix = confusion_matrix(y_test, yhat)
    
    return accuracy, classification_rep, conf_matrix, yhat

# Cargar datos
df, clase_LabelEncoder = load_data()

# Creación de las variables de entrada (X) y salida (y)
X = np.array(df.drop("Puesto", axis=1))  # Variables predictoras
y = np.array(df["Puesto"])  # Variable objetivo

# División en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Escalado de las características
x_scaler = MinMaxScaler()
X_train = x_scaler.fit_transform(X_train)  # Escalamos solo el conjunto de entrenamiento
X_test = x_scaler.transform(X_test)  # Aplicamos el mismo escalado al conjunto de prueba

# Crear y entrenar el modelo KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Configuración de la app de Streamlit
st.title("Clasificador de Puesto de Trabajo")

# Opción para borrar el caché y actualizar los datos
if st.button("Borrar caché y actualizar datos"):
    clear_data_cache()

# Mostrar datos crudos
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

# Convertir los valores seleccionados en números para hacer la predicción
data_input = np.array([[expertise, ubicacion, servicios, habilidades, herramientas, educacion]])

# Escalar la entrada del usuario
data_input_scaled = x_scaler.transform(data_input)

# Realizar la predicción
if st.button("Predecir Puesto"):
    prediction = model.predict(data_input_scaled)
    predicted_label = clase_LabelEncoder.inverse_transform(prediction)[0]
    st.write(f"El puesto de trabajo predicho es: {predicted_label}")

# Evaluación del modelo
if st.button("Evaluar modelo"):
    accuracy, classification_rep, conf_matrix, yhat = make_predictions(model, x_scaler, X_test, y_test, clase_LabelEncoder)

    # Mostrar la precisión
    st.write(f"Precisión del modelo: {accuracy:.2f}")

    # Mostrar el reporte de clasificación
    st.subheader("Reporte de clasificación:")
    st.text(classification_rep)

    # Mostrar la matriz de confusión
    st.subheader("Matriz de Confusión:")
    st.write(conf_matrix)
