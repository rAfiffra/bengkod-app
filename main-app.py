import streamlit as st
import numpy as np
import pickle

# Load model, scaler, dan encoder
model = pickle.load(open("model_obesity.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.title("🧠 Prediksi Tingkat Obesitas")

# Input pengguna
age = st.slider("Umur", 10, 100, 25)
height = st.number_input("Tinggi badan (m)", 1.4, 2.2, 1.7)
weight = st.number_input("Berat badan (kg)", 30, 200, 70)
fcvc = st.slider("Frekuensi konsumsi sayur (1-3)", 1.0, 3.0, 2.0)
ncp = st.slider("Jumlah makan per hari", 1.0, 4.0, 3.0)
ch2o = st.slider("Konsumsi air (liter)", 1.0, 3.0, 2.0)
faf = st.slider("Aktivitas fisik (jam/minggu)", 0.0, 3.0, 1.0)
tue = st.slider("Waktu layar (jam/hari)", 0.0, 2.0, 0.5)

# Fitur dalam urutan yang sama seperti pelatihan
input_data = np.array([[age, height, weight, fcvc, ncp, ch2o, faf, tue]])

# Standarisasi input
scaled_input = scaler.transform(input_data)

# Prediksi
if st.button("Prediksi"):
    pred = model.predict(scaled_input)
    kelas = label_encoder.inverse_transform(pred)[0]
    st.success(f"Tingkat Obesitas Anda: **{kelas}**")
