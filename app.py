import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import tensorflow as tf
from PIL import Image
import time

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de frutas",
    page_icon="🍎",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp { background-color: #FFE4EC; }
    .main-header { text-align: center; color: #FF5E8A; font-size: 3em; font-weight: bold; }
    .prediction-card { background-color: white; border-radius: 10px; padding: 20px; margin: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .quality-good { background-color: #2E8B57; color: white; padding: 10px; border-radius: 10px; text-align: center; }
    .quality-bad { background-color: #D43F6B; color: white; padding: 10px; border-radius: 10px; text-align: center; }
    .quality-regular { background-color: #FFA500; color: white; padding: 10px; border-radius: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Clasificador de calidad y tamaño de frutas</h1>', unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent / "models"

IMG_SIZE = 64

FEATURE_NAMES = ['area_px', 'aspect_ratio', 'coverage_ratio', 
                 'hue_mean', 'saturation_mean', 'value_mean',
                 'hue_std', 'saturation_std', 'value_std']

@st.cache_resource
def load_models():
    """Carga los modelos guardados y el scaler"""
    try:
        cnn_quality = tf.keras.models.load_model(BASE_DIR / 'cnn_quality.h5')
        xgb_size = joblib.load(BASE_DIR / 'xgb_size.pkl')
        le_quality = joblib.load(BASE_DIR / 'label_encoder_quality.pkl')
        le_size = joblib.load(BASE_DIR / 'label_encoder_size.pkl')
        
        scaler_path = BASE_DIR / 'scaler.pkl'
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        else:
            st.warning("No se encontró scaler.pkl. Las predicciones de tamaño pueden ser inexactas.")
            scaler = None
        
        return {
            'cnn_quality': cnn_quality,
            'xgb_size': xgb_size,
            'le_quality': le_quality,
            'le_size': le_size,
            'scaler': scaler
        }
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None

with st.spinner("Cargando modelos..."):
    models = load_models()

if models is None:
    st.stop()

def preprocess_image(image):
    """Preprocesa la imagen para CNN"""
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def extract_features_for_xgboost(image_rgb):
    """Extrae características para XGBoost (mismo orden que entrenamiento)"""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    area_px = image_rgb.shape[0] * image_rgb.shape[1]
    aspect_ratio = image_rgb.shape[1] / image_rgb.shape[0] if image_rgb.shape[0] > 0 else 0
    coverage_ratio = 1.0
    
    hue_mean = hsv[:,:,0].mean()
    saturation_mean = hsv[:,:,1].mean()
    value_mean = hsv[:,:,2].mean()
    hue_std = hsv[:,:,0].std()
    saturation_std = hsv[:,:,1].std()
    value_std = hsv[:,:,2].std()
    
    features = np.array([[
        area_px, aspect_ratio, coverage_ratio,
        hue_mean, saturation_mean, value_mean,
        hue_std, saturation_std, value_std
    ]])
    
    if models['scaler'] is not None:
        features = models['scaler'].transform(features)
    
    return features

def predict_quality(image_rgb):
    """Predice calidad usando CNN"""
    processed = preprocess_image(image_rgb)
    prediction = models['cnn_quality'].predict(processed, verbose=0)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]
    quality = models['le_quality'].inverse_transform([class_idx])[0]
    return quality, confidence

def predict_size(image_rgb):
    """Predice tamaño usando XGBoost con características escaladas"""
    features = extract_features_for_xgboost(image_rgb)
    prediction = models['xgb_size'].predict(features)[0]
    confidence_probs = models['xgb_size'].predict_proba(features)[0]
    confidence = np.max(confidence_probs)
    size = models['le_size'].inverse_transform([prediction])[0]
    return size, confidence

# Sidebar
st.sidebar.markdown("## Configuración")
option = st.sidebar.radio(
    "Selecciona el método de entrada:",
    ["Usar cámara en vivo", "Cargar imagen"]
)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Imagen de entrada")
    image = None
    
    if option == "Usar cámara en vivo":
        camera_image = st.camera_input("Toma una foto de la fruta")
        if camera_image is not None:
            image = Image.open(camera_image)
            image = np.array(image)
            st.image(image, caption="Fruta capturada", use_container_width=True)
    else:
        uploaded_file = st.file_uploader("Selecciona una imagen de fruta", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = np.array(image)
            st.image(image, caption="Imagen cargada", use_container_width=True)

if image is not None:
    with col2:
        st.markdown("### Resultados de la predicción")
        
        with st.spinner("Analizando la fruta..."):
            time.sleep(0.5)
            
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            elif len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            
            quality, quality_conf = predict_quality(image_resized)
            size, size_conf = predict_size(image_resized)
        
        st.markdown("#### Fruta detectada")
        
        if quality == 'good':
            st.markdown(f'<div class="quality-good">BUENA ({quality_conf*100:.1f}%)</div>', unsafe_allow_html=True)
        elif quality == 'bad':
            st.markdown(f'<div class="quality-bad">MALA ({quality_conf*100:.1f}%)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="quality-regular">REGULAR ({quality_conf*100:.1f}%)</div>', unsafe_allow_html=True)
        
        if size == 'pequeño':
            st.markdown(f'<div class="prediction-card" style="text-align:center">Pequeño ({size_conf*100:.1f}%)</div>', unsafe_allow_html=True)
        elif size == 'mediano':
            st.markdown(f'<div class="prediction-card" style="text-align:center">Mediano ({size_conf*100:.1f}%)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-card" style="text-align:center">Grande ({size_conf*100:.1f}%)</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹInformación")
st.sidebar.markdown(f"""
- **Modelo Calidad**: CNN
- **Modelo Tamaño**: XGBoost
- **Tamaño imagen**: {IMG_SIZE}x{IMG_SIZE}
- **Clases calidad**: {', '.join(models['le_quality'].classes_)}
- **Clases tamaño**: {', '.join(models['le_size'].classes_)}
""")

st.sidebar.markdown("### Instrucciones")
st.sidebar.markdown("""
1. Selecciona **Cámara en vivo** o **Cargar imagen**
2. Captura o selecciona una foto de una fruta
3. Espera la predicción automática
4. Revisa los resultados
""")