import streamlit as st
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas
# Removed unused import


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


# Guardar el modelo y el scaler juntos en un diccionario
modelo = {
    "scaler": scaler,
    "clf": clf
}

# Serializar con pickle
with open("svm_digits_model.pkl", "wb") as f:
    pickle.dump(modelo, f)


# Calcular la precisión del modelo en el conjunto de prueba
accuracy = clf.score(X_test, y_test) * 100

# Estilo CSS personalizado
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
        color: #0D47A1;  /* Añadir color de texto más oscuro */
        font-weight: 500;  /* Hacer el texto un poco más grueso */
    }
    .section-title {
        color: #0D47A1;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #BBDEFB;
    }
    .instructions {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
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
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #EEEEEE;
        text-align: center;
        color: #9E9E9E;
        font-size: 0.9rem;
    }
    .stButton>button {
        background-color: #1976D2;
        color: white !important;  /* Forzar color blanco */
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1976D2;  /* Mismo color que el normal para eliminar el efecto hover */
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("<h1 class='main-title'>🔍 Reconocimiento de Dígitos</h1>", unsafe_allow_html=True)

# Descripción de la aplicación
st.markdown("<div class='info-card'>", unsafe_allow_html=True)
st.markdown("""
Esta aplicación utiliza un modelo SVM entrenado para reconocer dígitos manuscritos. 
Puedes dibujar un dígito en el lienzo o subir una imagen para obtener una predicción.
""")
st.markdown("</div>", unsafe_allow_html=True)

# Cargar el modelo desde el archivo
with open("svm_digits_model.pkl", "rb") as f:
    modelo = pickle.load(f)

scaler = modelo["scaler"]
clf = modelo["clf"]

# Crear dos columnas para las opciones de entrada
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 class='section-title'>✏️ Dibuja un Dígito</h3>", unsafe_allow_html=True)
    
    st.markdown("<div class='instructions'>", unsafe_allow_html=True)
    st.markdown("""
    Dibuja un número del 0 al 9 en el lienzo negro y haz clic en "Predecir" para ver el resultado.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Create a canvas to draw the digit
    canvas = st_canvas(
        fill_color="rgb(0, 0, 0)",  # Fixed fill color with some opacity
        stroke_width=20,
        stroke_color="rgb(255, 255, 255)",
        background_color="rgb(0, 0, 0)",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Botón de predicción para el canvas
    if st.button("Predecir", key="predict_canvas"):
        if canvas.image_data is not None:
            # Función para preprocesar la imagen del canvas
            def preprocesar_canvas_para_svm(image_data):
                if image_data is None:
                    return None
                
                imagen = Image.fromarray(image_data.astype("uint8"))
                imagen = imagen.convert("L")
                imagen = imagen.resize((8, 8))
                
                imagen_array = np.array(imagen)
                imagen_array = 16 * (imagen_array / 255)  # Escalar a [0, 16]
                imagen_array = imagen_array.flatten().reshape(1, -1)
                imagen_array = scaler.transform(imagen_array)
                
                return imagen_array
            
            img_processed = preprocesar_canvas_para_svm(canvas.image_data)
            prediction = clf.predict(img_processed)
            
            st.markdown("<div class='prediction-result'>", unsafe_allow_html=True)
            st.markdown(f"El modelo predice que el número es: **{prediction[0]}**")
            st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<h3 class='section-title'>📷 Sube una Imagen</h3>", unsafe_allow_html=True)
    
    st.markdown("<div class='instructions'>", unsafe_allow_html=True)
    st.markdown("""
    Sube una imagen de un dígito manuscrito en formato JPG o PNG para obtener una predicción.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    archivo_subido = st.file_uploader(
        "Selecciona una imagen", type=["jpg", "png"]
    )
    
    if archivo_subido is not None:
        # Mostrar imagen con PIL
        image = Image.open(archivo_subido)
        st.image(image, caption="Imagen subida", width=200)
        
        # Función para preprocesar la imagen
        def preprocess_image(image):
            # Convertir a escala de grises
            image = image.convert("L")
            
            # Cambiar tamaño a 8x8
            image = image.resize((8, 8))
            
            image_array = np.array(image)
            image_array = 16 * (image_array / 255)  # Escalar a [0, 16]
            image_array = image_array.flatten().reshape(1, -1)
            image_array = scaler.transform(image_array)
            
            return image_array
        
        # Predice mediante la imagen
        def predict(image):
            image_array = preprocess_image(image)
            prediccion = clf.predict(image_array)
            return prediccion[0]
        
        # Make a prediction
        prediction = predict(image)
        
        st.markdown("<div class='prediction-result'>", unsafe_allow_html=True)
        st.markdown(f"El modelo predice que el número es: **{prediction}**")
        st.markdown("</div>", unsafe_allow_html=True)

# Información del modelo
st.markdown("<h3 class='section-title'>📊 Información del Modelo</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.markdown("**Tipo de Modelo**")
    st.markdown("SVM (Support Vector Machine)")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.markdown("**Preprocesamiento**")
    st.markdown("Redimensionado a 8x8 y normalizado")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Pie de página
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("Esta aplicación usa OpenCV para procesar imágenes y Scikit-learn para predecir dígitos manuscritos.")
st.markdown("</div>", unsafe_allow_html=True)

