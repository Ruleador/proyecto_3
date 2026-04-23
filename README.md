# 🐶 — Clasificador de Razas de Perros con Deep Learning


# 🧠 Descripción del proyecto

Este programa es una aplicación de Inteligencia Artificial que identifica la raza de un perro a partir de una imagen.

Utiliza técnicas de **Deep Learning y Transfer Learning** con redes neuronales convolucionales para realizar clasificación de imágenes de forma automática.

El modelo ha sido entrenado previamente con el Stanford Dogs Dataset y desplegado en una aplicación interactiva con Streamlit.

---

## 👁️ ¿Cómo funciona?

1. El usuario sube una imagen de un perro 🐶  
2. La imagen se preprocesa a 224x224 píxeles  
3. Un modelo de red neuronal convolucional analiza la imagen  
4. Se muestran las 3 razas más probables con su nivel de confianza  

---

## 🧠 Tecnologías utilizadas

- Python 🐍  
- TensorFlow (Deep Learning)  
- MobileNetV2 (Transfer Learning)  
- Streamlit (Interfaz web)  
- NumPy (Procesamiento de datos)  
- Pillow (Procesamiento de imágenes)  

---

## 📊 Dataset

Este proyecto utiliza el **Stanford Dogs Dataset**:

👉 https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset

- 🐕 120 razas de perros  
- 🖼️ +20.000 imágenes  
- 📁 Dataset estructurado por clases  

---

## ⚙️ Arquitectura del modelo

- Modelo base: MobileNetV2 preentrenado en ImageNet  
- Fine-tuning para clasificación de razas  
- Capa final: Softmax  
- Salida: Probabilidades por clase 


## FUNCIONAMIENTO EN RED LOCAL

- Usar streamlit run app.py en Visual Code