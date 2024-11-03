import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF

# Cargar los datos
@st.cache_data 
def load_data():
    data = pd.read_csv('Datos_definitivos.csv')  
    return data

# Crear gráficos en función de los datos
def plot_histogram(data, column):
    plt.figure(figsize=(10, 4))
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribución de {column}')
    st.pyplot(plt.gcf())
    plt.clf()  

def plot_boxplot(data, column):
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=data[column])
    plt.title(f'Distribución de {column}')
    st.pyplot(plt.gcf())
    plt.clf()  

# Función para descargar el análisis como PDF
def generate_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Análisis de Datos", ln=True, align="C")
    
    # Añadir información del DataFrame al PDF
    for col in df.columns:
        pdf.cell(200, 10, txt=f'{col}: {df[col].mean()}', ln=True)
    
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)  
    return pdf_output

# Configuración de la app de Streamlit
st.title("Análisis Exploratorio de Datos")

# Mostrar los datos cargados
data = load_data()
if st.checkbox("Mostrar datos crudos"):
    st.write(data)

# Selección de columnas
columns = data.columns.tolist()
selected_column = st.selectbox("Seleccione una columna para el análisis", columns)

# Mostrar histogramas
st.subheader("Histograma")
plot_histogram(data, selected_column)

# Mostrar boxplots
st.subheader("Boxplot")
plot_boxplot(data, selected_column)

