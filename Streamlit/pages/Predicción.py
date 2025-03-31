import streamlit as st
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Cargar el dataset
digits = load_digits()
X = digits.data
y = digits.target

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenar el modelo SVM
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)

# Guardar el modelo
modelo = {"scaler": scaler, "clf": clf}
with open("svm_digits_model.pkl", "wb") as f:
    pickle.dump(modelo, f)

# Cargar el modelo
with open("svm_digits_model.pkl", "rb") as f:
    modelo = pickle.load(f)

scaler = modelo["scaler"]
clf = modelo["clf"]

# Precisión del modelo
accuracy = clf.score(X_test, y_test) * 100

# CSS para mejorar el diseño
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
            margin: 1rem auto;
            border-radius: 8px;
            background-color: #E8F5E9;
            border: 2px solid #4CAF50;
            width: 60%;
        }
    </style>
""", unsafe_allow_html=True)

# Título
st.markdown("<h1 class='main-title'>🔍 Reconocimiento de Dígitos</h1>", unsafe_allow_html=True)

# Tarjeta informativa
st.markdown("""
    <div class="info-card">
        <p>Esta aplicación utiliza un modelo SVM entrenado para reconocer dígitos manuscritos.</p>
        <p>Puedes dibujar un dígito en el lienzo o subir una imagen para obtener una predicción.</p>
    </div>
""", unsafe_allow_html=True)

# Crear columnas para entrada
col1, col2 = st.columns(2)
prediccion = None

# --------------- Opción 1: Dibujar en el lienzo -----------------
with col1:
    st.subheader("✏️ Dibuja un Dígito")

    # Canvas centrado en su columna
    st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Predecir desde Lienzo", key="predict_canvas"):
        if canvas.image_data is not None:
            def preprocesar_canvas(image_data):
                imagen = Image.fromarray(image_data.astype("uint8")).convert("L").resize((8, 8))
                imagen_array = 16 * (np.array(imagen) / 255)  # Escalar a [0, 16]
                imagen_array = imagen_array.flatten().reshape(1, -1)
                return scaler.transform(imagen_array)

            img_processed = preprocesar_canvas(canvas.image_data)
            prediccion = clf.predict(img_processed)[0]

# --------------- Opción 2: Subir una imagen -----------------
with col2:
    st.subheader("📷 Sube una Imagen")
    archivo_subido = st.file_uploader("Selecciona una imagen", type=["jpg", "png"])

    if archivo_subido is not None:
        image = Image.open(archivo_subido)
        st.image(image, caption="Imagen subida", width=200)

        def preprocess_image(image):
            image = image.convert("L").resize((8, 8))
            image_array = 16 * (np.array(image) / 255)  # Escalar a [0, 16]
            return scaler.transform(image_array.flatten().reshape(1, -1))

        prediccion = clf.predict(preprocess_image(image))[0]

# --------------- Mostrar predicción debajo de ambas columnas -----------------
if prediccion is not None:
    st.markdown(f"<div class='prediction-result'>El modelo predice: <b>{prediccion}</b></div>", unsafe_allow_html=True)

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
        Aplicación desarrollada con Streamlit y Scikit-learn.
    </div>
""", unsafe_allow_html=True)
