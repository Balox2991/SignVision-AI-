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

st.title("Clasificador de Imagenes CIFAR-10")
st.subheader("Efrain Alvarez")

st.markdown("""
Dibuja una imagen en el lienzo y presiona **Predecir** para clasificarla.
El modelo intentara identificar a cual de las siguientes categorias pertenece:
""")

# -------------------------------------------------------
# Categorias con imagenes
# -------------------------------------------------------
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CLASS_IMAGES = {
    'airplane'  : 'https://cdn-icons-png.flaticon.com/512/10521/10521422.png',
    'automobile': 'https://cdn-icons-png.flaticon.com/512/3085/3085330.png',
    'bird'      : 'https://cdn-icons-png.flaticon.com/512/3069/3069186.png',
    'cat'       : 'https://cdn-icons-png.flaticon.com/512/6855/6855256.png',
    'deer'      : 'https://cdn-icons-png.flaticon.com/512/13397/13397056.png',
    'dog'       : 'https://cdn-icons-png.flaticon.com/512/2295/2295142.png',
    'frog'      : 'https://cdn-icons-png.flaticon.com/512/5999/5999613.png',
    'horse'     : 'https://cdn-icons-png.flaticon.com/512/3359/3359995.png',
    'ship'      : 'https://cdn-icons-png.flaticon.com/512/12278/12278772.png',
    'truck'     : 'https://cdn-icons-png.flaticon.com/512/3231/3231941.png',
}
# Mostrar categorias en 2 filas de 5
for i in range(0, len(CLASS_NAMES), 5):
    cols = st.columns(5)
    for j, col in enumerate(cols):
        if i + j < len(CLASS_NAMES):
            name = CLASS_NAMES[i + j]
            with col:
                st.image(CLASS_IMAGES[name], width=80)
                st.markdown(f"<p style='text-align:center; font-size:0.75rem;'>{name}</p>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------------
# Canvas para dibujar
# -------------------------------------------------------


# Selectores fuera de las columnas
col_color, col_stroke = st.columns(2)
with col_color:
    stroke_color = st.color_picker("Color del trazo", "#FFFFFF", key="color_picker_trazo")
with col_stroke:
    stroke_width = st.slider("Grosor del trazo", 1, 30, 12)

# Layout lienzo | prediccion
col_canvas, col_result = st.columns([1, 1], gap="large")

with col_canvas:
    st.markdown("### Dibuja aqui")
    canvas_result = st_canvas(
    fill_color="black",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="black",
    height=500,
    width=600,
    drawing_mode="freedraw",
    key="canvas"
)
    predecir = st.button("Predecir")

with col_result:
    st.markdown("### Top 3 predicciones")
    if predecir:
        if canvas_result.image_data is not None:
            img_check = np.array(canvas_result.image_data[:, :, :3])
            es_negro = np.all(img_check == 0)

            if es_negro:
                st.warning("Dibuja algo antes de predecir.")
            else:
                img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                img = img.convert('RGB')
                img = img.resize((32, 32))
                img_array = np.array(img).astype('float32') / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                model = tf.keras.models.load_model('modelo_cifar10.keras')
                predictions = model.predict(img_array)[0]

                mejor_pct = predictions.max() * 100

                if mejor_pct < 70:
                    st.warning("No se encontro coincidencia. Intenta dibujar con mas detalle.")
                else:
                    top3_idx = predictions.argsort()[-3:][::-1]
                    for idx in top3_idx:
                        pct = predictions[idx] * 100
                        st.markdown(f"**{CLASS_NAMES[idx]}**")
                        st.progress(int(pct))
                        st.markdown(f"{pct:.2f}%")
        else:
            st.warning("Dibuja algo antes de predecir.")