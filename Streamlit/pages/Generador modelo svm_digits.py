from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import streamlit as st
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
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2196F3;
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
        text-align: center;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #EEEEEE;
        text-align: center;
        color: #9E9E9E;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("<h1 class='main-title'>🤖 Entrenamiento de Modelo SVM</h1>", unsafe_allow_html=True)

# Mensaje de éxito - CAMBIADO a componente nativo de Streamlit
st.success("### ✅ ¡Entrenamiento Completado con Éxito!\n\nEl modelo SVM para reconocimiento de dígitos ha sido entrenado y guardado correctamente.")

# Información del modelo en tarjetas
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-value'>{:.2f}%</div>".format(accuracy), unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Precisión del Modelo</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-value'>svm_digits_model.pkl</div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Archivo Guardado</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Detalles del dataset
st.markdown("### 📊 Detalles del Dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Total de Muestras", len(X))
col2.metric("Muestras de Entrenamiento", len(X_train))
col3.metric("Muestras de Prueba", len(X_test))

# Información adicional
st.markdown("### 📝 Información del Modelo")
st.markdown("""
- **Tipo de Modelo**: SVM (Support Vector Machine)
- **Kernel**: Linear
- **Escalado**: StandardScaler
- **División de Datos**: 80% entrenamiento, 20% prueba
""")

# Próximos pasos
st.markdown("### 🚀 Próximos Pasos")
st.info("""
Ahora puedes utilizar este modelo para:
- Realizar predicciones con nuevos datos
- Implementarlo en una aplicación web
- Comparar su rendimiento con otros algoritmos
""")

# Pie de página
st.markdown("<div class='footer'>Modelo entrenado el " + st.session_state.get('date', '2025-03-31') + "</div>", unsafe_allow_html=True)

