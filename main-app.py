import streamlit as st
import numpy as np
import pickle
import joblib
import pandas as pd

# Load model, scaler, dan encoder
model = joblib.load("model_obesity.pkl")
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.title("🧠 Prediksi Tingkat Obesitas")

# Input numerik langsung
age = st.slider("Umur", 10, 100, 25)
height = st.number_input("Tinggi badan (m)", 1.4, 2.2, 1.7)
weight = st.number_input("Berat badan (kg)", 30, 200, 70)
fcvc = st.slider("Frekuensi makan sayur (1-3x sehari)", 1.0, 3.0, 2.0)
ncp = st.slider("Jumlah makan per hari(1-3x sehari)", 1.0, 4.0, 3.0)
ch2o = st.slider("Konsumsi air harian (liter/hari)", 1.0, 3.0, 2.0)
faf = st.slider("Aktivitas fisik (jam/minggu)", 0.0, 3.0, 1.0)
tue = st.slider("Waktu penggunaan teknologi (jam/hari)", 0.0, 2.0, 1.0)

# Kategori ordinal / biner
calc = st.selectbox("Konsumsi Alkohol", ['no', 'Sometimes', 'Frequently', 'Always'])
caec = st.selectbox("Konsumsi camilan", ['no', 'Sometimes', 'Frequently', 'Always'])
favc = st.radio("Sering makan tinggi kalori?", ['yes', 'no'])
scc = st.radio("Pantau asupan kalori?", ['yes', 'no'])
smoke = st.radio("Apakah merokok?", ['yes', 'no'])
family_history = st.radio("Riwayat keluarga obesitas?", ['yes', 'no'])

# Gender one-hot
gender = st.radio("Jenis kelamin", ['Male', 'Female'])
gender_female = 1 if gender == 'Female' else 0
gender_male = 1 if gender == 'Male' else 0

# Transportasi one-hot
mtrans = st.selectbox("Transportasi utama", ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'])
m_auto = int(mtrans == 'Automobile')
m_bike = int(mtrans == 'Bike')
m_motor = int(mtrans == 'Motorbike')
m_trans = int(mtrans == 'Public_Transportation')
m_walk = int(mtrans == 'Walking')

# Prediksi
if st.button("Prediksi"):
    input_data = np.array([[age, height, weight, 
                            ['no', 'Sometimes', 'Frequently', 'Always'].index(calc),
                            int(favc == 'no'), fcvc, ncp, int(scc == 'yes'), 
                            int(smoke == 'yes'), ch2o, int(family_history == 'yes'),
                            faf, tue, ['no', 'Sometimes', 'Frequently', 'Always'].index(caec),
                            gender_female, gender_male,
                            m_auto, m_bike, m_motor, m_trans, m_walk]])
    
    st.write("🔍 Data input yang dikirim ke model:")
    st.write(pd.DataFrame(input_data, columns=[
        'Age', 'Height', 'Weight', 
        'CALC', 'FAVC', 'FCVC', 'NCP', 'SCC', 'SMOKE', 'CH2O', 
        'family_history_with_overweight', 'FAF', 'TUE', 'CAEC', 
        'Gender_Female', 'Gender_Male', 
        'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike', 
        'MTRANS_Public_Transportation', 'MTRANS_Walking'
    ]))
    st.write("Jumlah fitur input:", input_data.shape[1])
    st.write("📐 Shape input:", input_data.shape)
    st.write("🧠 Jumlah fitur scaler:", scaler.n_features_in_)
    
  
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    kelas = label_encoder.inverse_transform(prediction)[0]
    
    st.success(f"Tingkat Obesitas Anda: **{kelas}**")
