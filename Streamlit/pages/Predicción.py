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
        .prediction-result {
            font-size: 1.8rem;
            font-weight: bold;
            text-align: center;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 8px;
            background-color: #E8F5E9;
            border: 2px solid #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("<h1 class='main-title'>🔍 Reconocimiento de Dígitos</h1>", unsafe_allow_html=True)

# Tarjeta informativa
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

# Crear dos columnas para las opciones de entrada
col1, col2 = st.columns(2)

# --------------- Opción 1: Dibujar en el lienzo -----------------
with col1:
    st.subheader("✏️ Dibuja un Dígito")
    
    canvas = st_canvas(
        fill_color="rgb(0, 0, 0)",
        stroke_width=20,
        stroke_color="rgb(255, 255, 255)",
        background_color="rgb(0, 0, 0)",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if st.button("Predecir desde Lienzo", key="predict_canvas"):
        if canvas.image_data is not None:
            def preprocesar_canvas(image_data):
                imagen = Image.fromarray(image_data.astype("uint8"))
                imagen = imagen.convert("L")
                imagen = imagen.resize((8, 8))
                imagen_array = np.array(imagen)
                imagen_array = 16 * (imagen_array / 255)  # Escalar a [0, 16]
                imagen_array = imagen_array.flatten().reshape(1, -1)
                imagen_array = scaler.transform(imagen_array)
                return imagen_array
            
            img_processed = preprocesar_canvas(canvas.image_data)
            prediction = clf.predict(img_processed)
            
            st.markdown(f"<div class='prediction-result'>El modelo predice: **{prediction[0]}**</div>", unsafe_allow_html=True)

# --------------- Opción 2: Subir una imagen -----------------
with col2:
    st.subheader("📷 Sube una Imagen")
    
    archivo_subido = st.file_uploader("Selecciona una imagen", type=["jpg", "png"])
    
    if archivo_subido is not None:
        image = Image.open(archivo_subido)
        st.image(image, caption="Imagen subida", width=200)
        
        def preprocess_image(image):
            image = image.convert("L")
            image = image.resize((8, 8))
            image_array = np.array(image)
            image_array = 16 * (image_array / 255)  # Escalar a [0, 16]
            image_array = image_array.flatten().reshape(1, -1)
            image_array = scaler.transform(image_array)
            return image_array
        
        def predict(image):
            image_array = preprocess_image(image)
            prediccion = clf.predict(image_array)
            return prediccion[0]
        
        prediction = predict(image)
        
        st.markdown(f"<div class='prediction-result'>El modelo predice: **{prediction}**</div>", unsafe_allow_html=True)

# Información del modelo
st.markdown("""
    <div class="info-card">
        <p>📊 <b>Información del Modelo</b></p>
        <ul style="text-align: left;">
            <li><b>Tipo:</b> SVM (Support Vector Machine)</li>
            <li><b>Kernel:</b> Lineal</li>
            <li><b>Preprocesamiento:</b> Redimensionado a 8x8 y normalizado</li>
            <li><b>Precisión:</b> {:.2f}%</li>
        </ul>
    </div>
""".format(accuracy), unsafe_allow_html=True)

# Pie de página
st.markdown("""
    <div style="margin-top: 3rem; text-align: center; color: #9E9E9E; font-size: 0.9rem;">
        Aplicación desarrollada con Streamlit, OpenCV y Scikit-learn.
    </div>
""", unsafe_allow_html=True)
