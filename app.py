import streamlit as st
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import time

# ------------------ Load Model ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
le = pickle.load(open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb"))

# ------------------ Page Config ------------------
st.set_page_config(page_title="Crop Recommendation",
                   page_icon="🌱", layout="wide")

# ------------------ Custom Styling ------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Sidebar ------------------
st.sidebar.title("🌾 About Project")
st.sidebar.info("""
This system recommends the best crop based on soil nutrients and environmental conditions using Machine Learning.
""")

st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Your Name")

# ------------------ Title ------------------
st.title("🌱 Smart Crop Recommendation System")
st.markdown(
    "Enter soil and environmental details to get the best crop suggestion.")

# ------------------ Input Layout ------------------
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0)
    P = st.number_input("Phosphorus (P)", min_value=0)
    K = st.number_input("Potassium (K)", min_value=0)

with col2:
    temp = st.number_input("Temperature (°C)")
    humidity = st.number_input("Humidity (%)")
    ph = st.number_input("pH")
    rainfall = st.number_input("Rainfall (mm)")

# ------------------ Prediction ------------------
if st.button("🔍 Predict Crop"):
    if N == 0 or P == 0 or K == 0:
        st.warning("⚠️ Please enter valid nutrient values!")
    else:
        data = np.array([[N, P, K, temp, humidity, ph, rainfall]])

        with st.spinner("Analyzing soil and weather data..."):
            time.sleep(2)
            prediction = model.predict(data)
            crop = le.inverse_transform(prediction)

        st.success(f"🌾 Recommended Crop: **{crop[0]}**")

# ------------------ Feature Importance ------------------
if st.checkbox("📊 Show Feature Importance"):
    try:
        importances = model.feature_importances_
        features = ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"]

        fig, ax = plt.subplots()
        ax.barh(features, importances)
        ax.set_title("Feature Importance")

        st.pyplot(fig)
    except:
        st.info("Feature importance not available for this model.")

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("🚀 Developed as a Data Science Project | Smart Agriculture System 🌱")
