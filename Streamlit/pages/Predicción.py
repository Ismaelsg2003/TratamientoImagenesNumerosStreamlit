import streamlit as st
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 1. Cargar el dataset de dígitos (8x8 imágenes)
digits = load_digits()
X = digits.data           # Datos: (n_samples, 64)
y = digits.target         # Etiquetas: (n_samples,)

# 2. Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Crear y entrenar el modelo SVM
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)

# Guardar el modelo
modelo = {"scaler": scaler, "clf": clf}
with open("svm_digits_model.pkl", "wb") as f:
    pickle.dump(modelo, f)

# Calcular la precisión
accuracy = clf.score(X_test, y_test) * 100

# Estilos CSS corregidos
st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #1E88E5;
        }
        .info-card {
            background-color: #E3F2FD;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            color: #0D47A1;
            font-weight: 500;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("<h1 class='main-title'>🔍 Reconocimiento de Dígitos</h1>", unsafe_allow_html=True)

# Tarjeta informativa corregida
st.markdown("""
    <div class="info-card">
        <p>Esta aplicación utiliza un modelo SVM entrenado para reconocer dígitos manuscritos.</p>
        <p>Puedes dibujar un dígito en el lienzo o subir una imagen para obtener una predicción.</p>
    </div>
""", unsafe_allow_html=True)

# Cargar el modelo desde el archivo
with open("svm_digits_model.pkl", "rb") as f:
    modelo = pickle.load(f)

scaler = modelo["scaler"]
clf = modelo["clf"]

# Próximos pasos
st.markdown("""
    <div class="info-card">
        <p>Ahora puedes utilizar este modelo para:</p>
        <ul style="text-align: left;">
            <li>Realizar predicciones con nuevos datos</li>
            <li>Implementarlo en una aplicación web</li>
            <li>Comparar su rendimiento con otros algoritmos</li>
        </ul>
    </div>
""", unsafe_allow_html=True)
