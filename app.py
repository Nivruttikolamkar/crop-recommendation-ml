import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.title("🌱 Crop Recommendation System")

st.write("Enter soil and environmental details:")

N = st.number_input("Nitrogen")
P = st.number_input("Phosphorus")
K = st.number_input("Potassium")
temp = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Predict Crop"):
    data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    prediction = model.predict(data)
    crop = le.inverse_transform(prediction)
    
    st.success(f"Recommended Crop: {crop[0]}")

    