import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import json

# Konfigurasi halaman
st.set_page_config(
    page_title="Obesity Classification App",
    page_icon="⚕️",
    layout="wide"
)

# ================================
# Load model
# ================================
@st.cache_resource
def load_model():
    try:
        components = joblib.load("obesity_prediction_components.joblib")
        return components
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

model_components = load_model()

# Fungsi prediksi
def predict_obesity(data, components):
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    encoding_maps = components['encoding_maps']
    feature_names = components['feature_names']
    model = components['model']

    # Encoding kategorikal
    for col in df.columns:
        if col in encoding_maps:
            df[col] = df[col].map(encoding_maps[col])
    
    df = df[feature_names]
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    
    inv_map = {v: k for k, v in encoding_maps['NObeyesdad'].items()}

    return {
        'prediction': int(pred),
        'prediction_label': inv_map[pred],
        'probability': float(prob[pred]),
        'probabilities': prob.tolist()
    }

# ================================
# UI
# ================================
st.title("⚕️ Obesity Classification App")
st.markdown("Prediksi tingkat obesitas berdasarkan data pribadi dan kebiasaan.")

# Sidebar Input
st.sidebar.header("📝 Masukkan Data Pengguna")
input_data = {
    'Gender': st.sidebar.selectbox("Gender", ['Male', 'Female']),
    'Age': st.sidebar.slider("Age", 10, 100, 25),
    'Height': st.sidebar.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.7),
    'Weight': st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0),
    'family_history_with_overweight': st.sidebar.selectbox("Family History Overweight", ['yes', 'no']),
    'FAVC': st.sidebar.selectbox("High Caloric Food Consumption (FAVC)", ['yes', 'no']),
    'FCVC': st.sidebar.slider("Frequency of Vegetable Consumption (FCVC)", 1.0, 3.0, 2.0),
    'NCP': st.sidebar.slider("Number of Main Meals (NCP)", 1.0, 4.0, 3.0),
    'CAEC': st.sidebar.selectbox("Eating Between Meals (CAEC)", ['Sometimes', 'Frequently', 'Always', 'no']),
    'SMOKE': st.sidebar.selectbox("Do You Smoke?", ['yes', 'no']),
    'CH2O': st.sidebar.slider("Daily Water Intake (liters)", 1.0, 3.0, 2.0),
    'SCC': st.sidebar.selectbox("Monitor Calories?", ['yes', 'no']),
    'FAF': st.sidebar.slider("Physical Activity Frequency (FAF)", 0.0, 3.0, 1.0),
    'TUE': st.sidebar.slider("Time Using Technology (TUE)", 0.0, 2.0, 1.0),
    'CALC': st.sidebar.selectbox("Alcohol Consumption (CALC)", ['Sometimes', 'Frequently', 'Always', 'no']),
    'MTRANS': st.sidebar.selectbox("Transport Mode", ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])
}

# Tombol Prediksi
if st.sidebar.button("🔮 Prediksi"):
    result = predict_obesity(input_data, model_components)

    st.subheader("🎯 Hasil Prediksi")
    st.markdown(f"**Kategori Obesitas:** `{result['prediction_label']}`")
    st.markdown(f"**Probabilitas:** `{result['probability']*100:.2f}%`")

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = result['probability'] * 100,
        title = {'text': "Tingkat Keyakinan (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "green" if result['probability'] > 0.75 else "orange"}
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Probabilitas semua kelas
    st.subheader("📊 Distribusi Probabilitas")
    labels = model_components['target_classes']
    prob_df = pd.DataFrame({
        'Kelas': labels,
        'Probabilitas': result['probabilities']
    }).sort_values('Probabilitas', ascending=False)
    
    fig_bar = px.bar(prob_df, x='Kelas', y='Probabilitas', color='Probabilitas',
                     title="Probabilitas Semua Kelas", height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Download hasil prediksi
    if st.download_button("📥 Unduh Hasil JSON", 
                          data=json.dumps(result, indent=2),
                          file_name=f"prediksi_obesitas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                          mime="application/json"):
        st.success("Berhasil diunduh!")

# Footer
st.markdown("---")
st.markdown("*Dataset: Obesity Classification | Model: Decision Tree*")
