import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Dog Breed AI", page_icon="🐶")

st.title("🐶 Detector de Razas de Perros")
st.write("Sube la imagen de tu perro y te diré que raza es.")

@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_perros.h5")

modelo = cargar_modelo()


with open("labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

archivo = st.file_uploader("Sube una imagen", type=["jpg","png","jpeg"])

if archivo:
    img = Image.open(archivo).convert("RGB")
    st.image(img, caption="Imagen subida", use_container_width=True)

    with st.spinner("Analizando..."):
        img = img.resize((224,224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = modelo.predict(img_array)[0]

        top3 = pred.argsort()[-3:][::-1]

        st.subheader("Resultados:")

        for i in top3:
            st.write(f"**{labels[i]}**")
            st.progress(float(pred[i]))