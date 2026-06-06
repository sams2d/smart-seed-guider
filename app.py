import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/soil_model.h5")
    with open("model/class_indices.json", "r") as f:
        class_indices = json.load(f)
    labels = {v: k for k, v in class_indices.items()}
    return model, labels

model, labels = load_model()


with open("soil_to_crops.json", "r") as f:
    soil_to_crops = json.load(f)


st.title("🌾 Smart Crop Recommendation System")
st.write("Upload a soil image to identify its type and suggest suitable crops.")

uploaded_file = st.file_uploader("Upload soil image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Soil Image", use_container_width=True)

    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = labels[np.argmax(predictions)]

    st.subheader(f"🧭 Predicted Soil Type: **{predicted_class.title()} Soil**")

    if predicted_class in soil_to_crops:
        crops = soil_to_crops[predicted_class]
        st.success("🌱 Recommended Crops:")
        for crop in crops:
            st.write(f"- {crop}")
    else:
        st.warning("No crop data found for this soil type.")