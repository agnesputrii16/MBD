import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json
import plotly.graph_objects as go
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(
    page_title="Obesity Prediction App ",
    page_icon="⚕️",
    layout="wide"
)

# ===========================
# Load model
# ===========================
@st.cache_resource
def load_model():
    try:
        components = joblib.load("obesity_prediction_components.joblib")
        return components
    except Exception as e:
        st.error(f"Model gagal dimuat: {e}")
        st.stop()

model_components = load_model()

# ===========================
# Fungsi Prediksi
# ===========================
def predict_obesity(data, components):
    df = pd.DataFrame([data])
    model = components['model']
    encoding_maps = components['encoding_maps']
    feature_names = components['feature_names']
    for col in df.columns:
        if col in encoding_maps:
            df[col] = df[col].map(encoding_maps[col])
    df = df[feature_names]
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    inv_map = {v: k for k, v in encoding_maps['NObeyesdad'].items()}
    return {
        'prediction': int(prediction),
        'prediction_label': inv_map[prediction],
        'probability': float(probabilities[prediction]),
        'probabilities': probabilities.tolist()
    }

def reset_state(keys):
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]

def export_prediction(data, result):
    return json.dumps({
        "timestamp": datetime.now().isoformat(),
        "input_data": data,
        "prediction": {
            "class": result['prediction_label'],
            "confidence": result['probability'],
            "raw_prediction": result['prediction']
        }
    }, indent=2)

# ===========================
# UI
# ===========================
st.title("⚕️ Obesity Prediction App ")
st.markdown("Wanda Setya (06753), Devi Marchanda (06768), Najwa Syafira (06793), Agnes Putri (06797)")
st.markdown("Prediksi tingkat obesitas berdasarkan data pribadi dan kebiasaan hidup.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Input Features")
    with st.form("prediction_form"):
        gender = st.selectbox("Gender", ['Male', 'Female'], key='Gender')
        age = st.slider("Age", 10, 100, 25, key='Age')
        height = st.number_input("Height (m)", 1.0, 2.5, 1.7, key='Height')
        weight = st.number_input("Weight (kg)", 20.0, 200.0, 70.0, key='Weight')
        family_history = st.selectbox("Family History with Overweight", ['yes', 'no'], key='family_history_with_overweight')
        favc = st.selectbox("High Caloric Food Consumption (FAVC)", ['yes', 'no'], key='FAVC')
        fcvc = st.slider("Frequency of Vegetable Consumption (FCVC)", 1.0, 3.0, 2.0, key='FCVC')
        ncp = st.slider("Number of Main Meals (NCP)", 1.0, 4.0, 3.0, key='NCP')
        caec = st.selectbox("Eating Between Meals (CAEC)", ['Sometimes', 'Frequently', 'Always', 'no'], key='CAEC')
        smoke = st.selectbox("Do You Smoke?", ['yes', 'no'], key='SMOKE')
        ch2o = st.slider("Water Intake (liters)", 1.0, 3.0, 2.0, key='CH2O')
        scc = st.selectbox("Monitor Calories?", ['yes', 'no'], key='SCC')
        faf = st.slider("Physical Activity Frequency (FAF)", 0.0, 3.0, 1.0, key='FAF')
        tue = st.slider("Time Using Technology (TUE)", 0.0, 2.0, 1.0, key='TUE')
        calc = st.selectbox("Alcohol Consumption (CALC)", ['Sometimes', 'Frequently', 'Always', 'no'], key='CALC')
        mtrans = st.selectbox("Transport Mode", ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'], key='MTRANS')

        pred_btn, reset_btn, export_btn = st.columns(3)
        predict = pred_btn.form_submit_button("🔮 Predict")
        reset = reset_btn.form_submit_button("🔄 Reset")
        export = export_btn.form_submit_button("📤 Export Last Result")

# Reset button
if reset:
    reset_state([
        'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP',
        'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'
    ])
    st.rerun()

# Prediction
if predict:
    user_data = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_history,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans
    }

    result = predict_obesity(user_data, model_components)
    st.session_state['last_prediction'] = {'input_data': user_data, 'result': result}

    with col2:
        st.subheader("🎯 Hasil Prediksi")
        st.markdown(f"**Kategori Obesitas:** `{result['prediction_label']}`")
        st.markdown(f"**Probabilitas:** `{result['probability']*100:.2f}%`")

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['probability']*100,
            title={"text": "Tingkat Keyakinan (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "green" if result['probability'] > 0.75 else "orange"}}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Probability bar
        st.subheader("📊 Distribusi Probabilitas")
        label_names = model_components['target_classes']
        prob_df = pd.DataFrame({'Class': label_names, 'Probability': result['probabilities']})
        fig_bar = px.bar(prob_df.sort_values('Probability', ascending=False), x='Class', y='Probability',
                         color='Probability', color_continuous_scale='viridis')
        st.plotly_chart(fig_bar, use_container_width=True)

# Export
if export:
    if 'last_prediction' in st.session_state:
        export_data = export_prediction(
            st.session_state['last_prediction']['input_data'],
            st.session_state['last_prediction']['result']
        )
        st.download_button("📥 Download JSON", data=export_data,
                           file_name=f"obesity_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                           mime="application/json")
    else:
        st.warning("⚠️ Belum ada prediksi untuk diekspor.")

# Feature Importance
st.subheader("📌 Feature Importance")
try:
    feature_names = model_components['feature_names']
    importances = model_components['model'].feature_importances_
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)
    fig_feat = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Feature Importance in Decision Tree")
    st.plotly_chart(fig_feat, use_container_width=True)
except Exception as e:
    st.error(f"Gagal menampilkan feature importance: {e}")

# Footer
st.markdown("---")
st.markdown("*Dataset Obesity | Model Decision Tree*")
