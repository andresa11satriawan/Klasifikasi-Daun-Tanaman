import streamlit as st
import gdown
import os
from tensorflow.keras.models import model_from_json
from PIL import Image
import numpy as np

# Google Drive links
json_url = "https://drive.google.com/uc?id=1DJ9HD9XvHe3sBWHAxbqBi4vspihAynRV"
h5_url = "https://drive.google.com/uc?id=1EcaI5gCtw0JMTAIfsJT1Ea4WYXXI-xwn"

# File paths
json_file = "model_architecture.json"
h5_file = "agroscan_model.h5"

# Download model files if not exists
if not os.path.exists(json_file):
    gdown.download(json_url, json_file, quiet=False)

if not os.path.exists(h5_file):
    gdown.download(h5_url, h5_file, quiet=False)


# Load model
def load_model_files(json_file, h5_file):
    with open(json_file, "r") as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(h5_file)
    return model


# Preprocess image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)


# Streamlit UI
st.title("Deteksi Penyakit pada Daun Tanaman")
model = load_model_files(json_file, h5_file)

# Daftar kelas sesuai dataset Anda
class_names = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus", "Tomato_healthy"
]

# Upload image
uploaded_image = st.file_uploader("Unggah gambar daun tanaman:", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Display uploaded image
    image = Image.open(uploaded_image)
    image = image.resize((300, 300))  # Resize image to smaller size
    st.image(image, caption="Gambar yang diunggah", use_column_width=False)

    # Preprocess and predict
    st.write("Memproses gambar...")
    processed_image = preprocess_image(image, target_size=(128, 128))
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display prediction
    st.write(f"Prediksi: **{predicted_class}**")
    st.write(f"Tingkat keyakinan: **{confidence * 100:.2f}%**")
