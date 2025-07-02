import streamlit as st
import pandas as pd
import joblib

# Load model, label encoder, dan kolom fitur
try:
    model = joblib.load("random_forest_model.pkl")
    le = joblib.load("label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except Exception as e:
    st.error(f"Gagal memuat model atau file .pkl: {e}")
    st.stop()

st.title("Prediksi Penyakit Hewan Ternak 🐄🐃🐏🐐")
st.markdown("Masukkan informasi hewan ternak untuk memprediksi penyakit berdasarkan gejala.")

# ✅ Pilihan dropdown disesuaikan dengan isi dataset
animal_options = ['cow', 'buffalo', 'sheep', 'goat']
symptom1_options = ['depression', 'painless lumps', 'loss of appetite', 'difficulty walking', 'lameness']
symptom2_options = [
    'loss of appetite', 'blisters on gums', 'painless lumps',
    'swelling in limb', 'blisters on tongue', 'blisters on mouth'
]
symptom3_options = [
    'loss of appetite', 'depression', 'crackling sound',
    'difficulty walking', 'painless lumps'
]

# Input pengguna
animal = st.selectbox("Jenis Hewan", animal_options)
age = st.number_input("Usia Hewan (tahun)", min_value=0, max_value=30, value=2)
temperature = st.number_input("Suhu Tubuh (°F)", min_value=90.0, max_value=110.0, value=101.0)
symptom1 = st.selectbox("Gejala 1", symptom1_options)
symptom2 = st.selectbox("Gejala 2", symptom2_options)
symptom3 = st.selectbox("Gejala 3", symptom3_options)

# Tombol prediksi
if st.button("Prediksi Penyakit"):
    # Buat DataFrame input
    input_df = pd.DataFrame([{
        "Animal": animal,
        "Age": age,
        "Temperature": temperature,
        "Symptom 1": symptom1,
        "Symptom 2": symptom2,
        "Symptom 3": symptom3
    }])

    try:
        # Lakukan encoding sesuai training
        input_encoded = pd.get_dummies(input_df)

        # Tambahkan kolom yang hilang agar sama dengan training
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Urutkan kolom agar cocok dengan training
        input_encoded = input_encoded[feature_columns]

        # Prediksi
        prediction = model.predict(input_encoded)
        hasil = le.inverse_transform(prediction)[0]

        st.success(f"🩺 Prediksi Penyakit: **{hasil}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
