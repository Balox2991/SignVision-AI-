# app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# -------------------------------------------------------
# Configuracion de la pagina
# -------------------------------------------------------
st.set_page_config(page_title="Clasificador CIFAR-10", layout="wide")

# -------------------------------------------------------
# Estilos personalizados
# -------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Mono', monospace;
            background-color: #0d0d0d;
            color: #f0f0f0;
        }

        .header-block {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            border-left: 5px solid #e94560;
            padding: 2rem 2.5rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }

        .header-block h1 {
            font-family: 'Syne', sans-serif;
            font-size: 2.8rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0;
            letter-spacing: -1px;
        }

        .header-block p {
            color: #e94560;
            font-size: 0.95rem;
            margin: 0.3rem 0 0 0;
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        .desc-box {
            background: #161616;
            border: 1px solid #2a2a2a;
            border-radius: 10px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1.5rem;
            font-size: 0.88rem;
            color: #aaaaaa;
            line-height: 1.7;
        }

        .badge {
            display: inline-block;
            background: #1a1a2e;
            border: 1px solid #e94560;
            color: #e94560;
            padding: 8px 18px;
            border-radius: 20px;
            font-size: 1rem;
            margin: 5px;
            letter-spacing: 1px;
        }

        .section-label {
            font-family: 'Syne', sans-serif;
            font-size: 1rem;
            color: #e94560;
            text-transform: uppercase;
            letter-spacing: 3px;
            margin-bottom: 0.8rem;
        }

        .result-card {
            background: #161616;
            border: 1px solid #2a2a2a;
            border-radius: 10px;
            padding: 1rem 1.5rem;
            margin-bottom: 0.8rem;
        }

        .result-card .rank {
            font-size: 0.75rem;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .result-card .class-name {
            font-family: 'Syne', sans-serif;
            font-size: 1.4rem;
            font-weight: 800;
            color: #ffffff;
        }

        .result-card .pct {
            font-size: 1rem;
            color: #e94560;
        }

        .stButton > button {
            background: #e94560;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.7rem 2rem;
            font-family: 'Space Mono', monospace;
            font-size: 0.9rem;
            letter-spacing: 1px;
            width: 100%;
            margin-top: 1rem;
            cursor: pointer;
        }

        .stButton > button:hover {
            background: #c73652;
        }

        div[data-testid="stProgress"] > div > div {
            background-color: #e94560 !important;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.markdown("""
    <div class="header-block">
        <h1>Clasificador CIFAR-10</h1>
        <p>Efrain Alvarez &nbsp;|&nbsp; Vision por Computador</p>
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Clases
# -------------------------------------------------------
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# -------------------------------------------------------
# Instrucciones y categorias
# -------------------------------------------------------
st.markdown('<div class="section-label">Instrucciones</div>', unsafe_allow_html=True)
st.markdown("""
    <div class="desc-box">
        Dibuja cualquier objeto en el lienzo usando el mouse o el dedo.
        El modelo intentara reconocer a cual de las 10 categorias pertenece tu dibujo.
        Cuando termines presiona <strong style="color:#e94560">Predecir</strong> para ver los resultados.
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Categorias disponibles</div>', unsafe_allow_html=True)
badges_html = "".join([f'<span class="badge">{c}</span>' for c in CLASS_NAMES])
st.markdown(badges_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------
# Layout: lienzo | prediccion
# -------------------------------------------------------
col_canvas, col_result = st.columns([1, 1], gap="large")

with col_canvas:
    st.markdown('<div class="section-label">Lienzo de dibujo</div>', unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=14,
        stroke_color="black",
        background_color="#ffffff",
        height=480,
        width=480,
        drawing_mode="freedraw",
        key="canvas"
    )

    predecir = st.button("Predecir")

with col_result:
    st.markdown('<div class="section-label">Top 3 predicciones</div>', unsafe_allow_html=True)

    if predecir:
        if canvas_result.image_data is not None:
            img_check = np.array(canvas_result.image_data[:, :, :3])
            es_blanco = np.all(img_check == 255)

            if es_blanco:
                st.warning("Dibuja algo antes de predecir.")
            else:
                img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                img = img.convert('RGB')
                img = img.resize((32, 32))
                img_array = np.array(img).astype('float32') / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                model = tf.keras.models.load_model('modelo_cifar10.keras')
                predictions = model.predict(img_array)[0]

                # Verificar si la mejor prediccion supera el 70%
                mejor_pct = predictions.max() * 100

                if mejor_pct < 70:
                    st.warning("No se encontro coincidencia. Intenta dibujar con mas detalle.")
                else:
                    top3_idx = predictions.argsort()[-3:][::-1]
                    medals = ["1er lugar", "2do lugar", "3er lugar"]

                    for rank, idx in enumerate(top3_idx):
                        pct = predictions[idx] * 100
                        st.markdown(f"""
                            <div class="result-card">
                                <div class="rank">{medals[rank]}</div>
                                <div class="class-name">{CLASS_NAMES[idx]}</div>
                                <div class="pct">{pct:.2f}%</div>
                            </div>
                        """, unsafe_allow_html=True)
                        st.progress(int(pct))
    else:
        st.warning("Dibuja algo antes de predecir.")