import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
import os

# -------------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------------
st.set_page_config(
    page_title="SignVision AI",
    page_icon="🤟",
    layout="wide"
)

# -------------------------------------------------------
# CSS
# -------------------------------------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background:
    linear-gradient(
        135deg,
        #0B1020 0%,
        #111827 40%,
        #0F172A 100%
    );
    color: white;
}

section[data-testid="stSidebar"]{
    background-color: #111827;
}

.titulo{
    font-size: 3.5rem;
    font-weight: bold;
    text-align: center;
    color: #00BFFF;
    margin-top: 10px;
}

.subtitulo{
    text-align:center;
    font-size:1.2rem;
    color:white;
    margin-bottom:25px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.markdown("""
<div class='titulo'>
🤟 SignVision AI
</div>

<div class='subtitulo'>
Aprendizaje del español mediante lenguaje de señas e inteligencia artificial
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# BADGES
# -------------------------------------------------------
st.markdown("""
<div style='text-align:center; margin-bottom:35px;'>

<span style='
background:#00BFFF;
padding:8px 15px;
border-radius:20px;
color:black;
font-weight:bold;
margin-right:10px;
'>
EFRAIN ALVAREZ LOBO
</span>

<span style='
background:#00E676;
padding:8px 15px;
border-radius:20px;
color:black;
font-weight:bold;
margin-right:10px;
'>
JUAN DAVID AMAYA QUINTERO
</span>

<span style='
background:#FFD600;
padding:8px 15px;
border-radius:20px;
color:black;
font-weight:bold;
'>
ANDRES FELIPE ARDILA QUIÑONES
</span>

</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
with st.sidebar:

    st.title("📚 Información")

    st.markdown("""
### Tecnologías

- TensorFlow
- ResNet50
- Deep Learning
- OpenCV
- Streamlit
- Transfer Learning
- Computer Vision
""")

    st.markdown("---")

    st.markdown("""
### Consejos

✅ Usa trazos gruesos

✅ Dibuja grande

✅ Rellena figuras

✅ Usa fondo negro

❌ Evita dibujos pequeños
""")

    st.markdown("---")

    st.info(
        "Sistema inclusivo basado en inteligencia artificial y lenguaje de señas."
    )

# -------------------------------------------------------
# CLASES
# -------------------------------------------------------
CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

CLASS_NAMES_ES = {
    'airplane': 'avion',
    'automobile': 'carro',
    'bird': 'pajaro',
    'cat': 'gato',
    'deer': 'ciervo',
    'dog': 'perro',
    'frog': 'rana',
    'horse': 'caballo',
    'ship': 'barco',
    'truck': 'camion'
}

# -------------------------------------------------------
# DESCRIPCIONES
# -------------------------------------------------------
DESCRIPCIONES = {
    'airplane': 'Medio de transporte aéreo.',
    'automobile': 'Vehículo terrestre utilizado para transporte.',
    'bird': 'Animal vertebrado con plumas.',
    'cat': 'Animal doméstico felino.',
    'deer': 'Mamífero herbívoro.',
    'dog': 'Animal doméstico y compañero humano.',
    'frog': 'Anfibio que vive en ambientes húmedos.',
    'horse': 'Animal usado para transporte y trabajo.',
    'ship': 'Medio de transporte marítimo.',
    'truck': 'Vehículo de carga terrestre.'
}

# -------------------------------------------------------
# ICONOS
# -------------------------------------------------------
CLASS_IMAGES = {
    'airplane': 'https://cdn-icons-png.flaticon.com/512/10521/10521422.png',
    'automobile': 'https://cdn-icons-png.flaticon.com/512/3085/3085330.png',
    'bird': 'https://cdn-icons-png.flaticon.com/512/3069/3069186.png',
    'cat': 'https://cdn-icons-png.flaticon.com/512/6855/6855256.png',
    'deer': 'https://cdn-icons-png.flaticon.com/512/13397/13397056.png',
    'dog': 'https://cdn-icons-png.flaticon.com/512/2295/2295142.png',
    'frog': 'https://cdn-icons-png.flaticon.com/512/5999/5999613.png',
    'horse': 'https://cdn-icons-png.flaticon.com/512/3359/3359995.png',
    'ship': 'https://cdn-icons-png.flaticon.com/512/12278/12278772.png',
    'truck': 'https://cdn-icons-png.flaticon.com/512/3231/3231941.png',
}

# -------------------------------------------------------
# MOSTRAR CLASES
# -------------------------------------------------------
st.markdown("## 🧠 Categorías reconocidas")

for i in range(0, len(CLASS_NAMES), 5):

    cols = st.columns(5)

    for j, col in enumerate(cols):

        if i + j < len(CLASS_NAMES):

            name = CLASS_NAMES[i + j]

            with col:

                st.image(CLASS_IMAGES[name], width=90)

                st.markdown(
                    f"""
                    <div style='text-align:center;'>
                    <b>{CLASS_NAMES_ES[name].capitalize()}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.markdown("---")

# -------------------------------------------------------
# SELECTOR
# -------------------------------------------------------
modo = st.radio(
    "Selecciona una opción",
    [
        "✏️ Dibujar",
        "📁 Subir imagen"
    ],
    horizontal=True
)

st.markdown("---")

# -------------------------------------------------------
# CARGAR MODELO
# -------------------------------------------------------
@st.cache_resource
def cargar_modelo():

    return tf.keras.models.load_model(
        "modelo_cifar10_resnet.keras",
        custom_objects={"preprocess_input": preprocess_input}
    )

# -------------------------------------------------------
# PREPROCESAMIENTO
# -------------------------------------------------------
def procesar_imagen(img_pil):

    img = img_pil.convert("RGB")

    img = img.resize((32, 32))

    img_array = np.array(img)

    gray = cv2.cvtColor(
        img_array,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.bitwise_not(gray)

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    kernel = np.ones((2, 2), np.uint8)

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    edges = cv2.convertScaleAbs(
        edges,
        alpha=2.0,
        beta=0
    )

    processed = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2RGB
    )

    return processed

# -------------------------------------------------------
# PREDICCIÓN
# -------------------------------------------------------
def predecir_imagen(img_pil):

    model = cargar_modelo()

    processed = procesar_imagen(img_pil)

    processed = processed.astype("float32")

    processed = np.expand_dims(processed, axis=0)

    predictions = model.predict(processed)[0]

    return predictions

# -------------------------------------------------------
# RESULTADOS
# -------------------------------------------------------
def mostrar_resultados(predictions):

    mejor_idx = np.argmax(predictions)

    mejor_clase = CLASS_NAMES[mejor_idx]

    mejor_pct = predictions[mejor_idx] * 100

    nombre_es = CLASS_NAMES_ES[mejor_clase].capitalize()

    st.markdown("---")

    st.markdown("## 🎯 Resultado principal")

    st.success(
        f"{nombre_es} — {mejor_pct:.2f}%"
    )

    st.markdown("---")

    st.subheader("📊 Top 3 predicciones")

    top3_idx = predictions.argsort()[-3:][::-1]

    for idx in top3_idx:

        pct = predictions[idx] * 100

        st.markdown(
            f"### {CLASS_NAMES_ES[CLASS_NAMES[idx]].capitalize()}"
        )

        st.progress(int(pct))

        st.markdown(f"**{pct:.2f}%**")

    st.markdown("---")

    st.subheader("📘 Información")

    st.info(
        DESCRIPCIONES[mejor_clase]
    )

# -------------------------------------------------------
# DIBUJAR
# -------------------------------------------------------
if modo == "✏️ Dibujar":

    stroke_width = st.slider(
        "Grosor del trazo",
        10,
        40,
        20
    )

    col_canvas, col_video = st.columns([1.2, 1])

    with col_canvas:

        st.markdown("## ✏️ Dibuja aquí")

        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=500,
            width=600,
            drawing_mode="freedraw",
            key="canvas",
        )

        predecir_btn = st.button(
            "🔍 Predecir",
            use_container_width=True
        )

    with col_video:

        st.markdown("## 🤟 Lengua de señas")

    if predecir_btn:

        if canvas_result.image_data is not None:

            img_check = np.array(
                canvas_result.image_data[:, :, :3]
            )

            es_negro = np.all(img_check == 0)

            if es_negro:

                st.warning(
                    "Dibuja algo antes de predecir."
                )

            else:

                img_pil = Image.fromarray(
                    canvas_result.image_data.astype("uint8"),
                    "RGBA"
                )

                predictions = predecir_imagen(img_pil)

                mejor_idx = np.argmax(predictions)

                mejor_clase = CLASS_NAMES[mejor_idx]

                nombre_video = CLASS_NAMES_ES[
                    mejor_clase
                ]

                video_path = os.path.join(
                    "videos",
                    f"{nombre_video}.mp4"
                )

                with col_video:

                    if os.path.exists(video_path):

                        st.video(video_path)

                    else:

                        st.warning(
                            f"No existe el video: {video_path}"
                        )

                mostrar_resultados(predictions)

# -------------------------------------------------------
# SUBIR IMAGEN
# -------------------------------------------------------
else:

    col_upload, col_video = st.columns([1.2, 1])

    with col_upload:

        st.markdown("## 📁 Subir imagen")

        archivo = st.file_uploader(
            "Selecciona una imagen",
            type=["jpg", "jpeg", "png"]
        )

        if archivo is not None:

            img_pil = Image.open(archivo)

            st.image(
                img_pil,
                use_container_width=True
            )

        predecir_btn = st.button(
            "🔍 Predecir imagen",
            use_container_width=True
        )

    with col_video:

        st.markdown("## 🤟 Lengua de señas")

    if predecir_btn:

        if archivo is None:

            st.warning(
                "Sube una imagen."
            )

        else:

            img_pil = Image.open(archivo)

            predictions = predecir_imagen(img_pil)

            mejor_idx = np.argmax(predictions)

            mejor_clase = CLASS_NAMES[mejor_idx]

            nombre_video = CLASS_NAMES_ES[
                mejor_clase
            ]

            video_path = os.path.join(
                "videos",
                f"{nombre_video}.mp4"
            )

            with col_video:

                if os.path.exists(video_path):

                    st.video(video_path)

                else:

                    st.warning(
                        f"No existe el video: {video_path}"
                    )

            mostrar_resultados(predictions)