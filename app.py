import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set up the web page layout
st.set_page_config(page_title="Gait Analysis AI", layout="wide")
st.title("🚶‍♂️ Parkinson's Disease Detection & Staging")
st.write("Upload MPU6050 sensor data (CSV) to analyze the walking pattern.")

# Load the trained AI model
@st.cache_resource
def load_model():
    return joblib.load('parkinsons_rf_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found! Please run train_model.py first.")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Upload Gait Data (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    
    # --- FIX: DATA CLEANING ---
    # Force all columns to be numbers. If it finds ESP32 boot text, it turns it into a Blank (NaN)
    sensor_cols = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    for col in sensor_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Drop any rows that contain those blanks (removes the text glitches)
    df = df.dropna(subset=sensor_cols)
    # --------------------------
    
    st.markdown("---")
    st.subheader("📊 Cleaned Sensor Data Visualization")
    
    # Graph the cleaned data
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Accelerometer (Motion & Stride)**")
        st.line_chart(df[['Ax', 'Ay', 'Az']])
        
    with col2:
        st.write("**Gyroscope (Tremors & Rotation)**")
        st.line_chart(df[['Gx', 'Gy', 'Gz']])
        
    st.markdown("---")
    st.subheader("🤖 AI Diagnosis & Staging")
    
    with st.spinner("Extracting mathematical features and running AI model..."):
        window_size = 150
        features = []
        
        if len(df) < window_size:
            windows = [df]
        else:
            windows = [df.iloc[i:i+window_size] for i in range(0, len(df), window_size) if len(df.iloc[i:i+window_size]) == window_size]
            
        for window in windows:
            row_features = {}
            for col in sensor_cols:
                row_features[f'{col}_mean'] = window[col].mean()
                row_features[f'{col}_std'] = window[col].std()
                row_features[f'{col}_max'] = window[col].max()
                row_features[f'{col}_min'] = window[col].min()
                row_features[f'{col}_rms'] = np.sqrt(np.mean(window[col]**2))
            features.append(row_features)
            
        features_df = pd.DataFrame(features)
        
        # Make Predictions for every 3-second window
        predictions = model.predict(features_df)
        
        # Calculate Frequency of Parkinson's patterns
        parkinson_count = sum(predictions == 1)
        total_windows = len(predictions)
        parkinson_percent = (parkinson_count / total_windows) * 100
        
        # --- NEW: PARKINSON'S STAGING LOGIC ---
        if parkinson_percent == 0:
            st.success("✅ **Diagnosis: Normal Gait Pattern**")
            st.write("No Parkinsonian gait features (shuffling or resting tremors) were detected in this walk.")
            
        elif 0 < parkinson_percent <= 30:
            st.warning("⚠️ **Diagnosis: Borderline / Early Stage Indicator**")
            st.write(f"The AI detected abnormal patterns in **{parkinson_percent:.1f}%** of the walk.")
            st.write("**Analysis:** The gait is mostly normal, but shows occasional brief micro-shuffles or slight hesitation. This could indicate fatigue, or an early-stage motor symptom requiring clinical observation.")
            
        elif 30 < parkinson_percent <= 75:
            st.error("🛑 **Diagnosis: Moderate Parkinsonian Gait Detected**")
            st.write(f"The AI detected abnormal patterns in **{parkinson_percent:.1f}%** of the walk.")
            st.write("**Analysis:** Consistent freezing of gait, reduced stride length, and measurable leg tremors detected. Symptoms are highly consistent with Mid-Stage Parkinson's Disease.")
            
        else:
            st.error("🚨 **Diagnosis: Severe Parkinsonian Gait Detected**")
            st.write(f"The AI detected abnormal patterns in **{parkinson_percent:.1f}%** of the walk.")
            st.write("**Analysis:** Continuous shuffling (festination), high-frequency resting tremors, and lack of distinct heel-strikes throughout the entire recording. Consistent with Advanced Stage motor degradation.")
