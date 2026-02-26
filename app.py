import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌍",
    layout="centered"
)
st.title("🌍 Air Quality Prediction App")
st.markdown("### Predict Air Quality based on environmental factors")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('updated_pollution_dataset.csv')
    
    # Encode target variable
    le = LabelEncoder()
    df['Air Quality'] = le.fit_transform(df['Air Quality'])
    
    # Split features and target
    X = df.drop('Air Quality', axis=1)
    y = df['Air Quality']
    
    return X, y, le

# Train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    return model, accuracy

# Load data and train model
try:
    X, y, label_encoder = load_data()
    model, accuracy = train_model(X, y)
    
    st.sidebar.success(f"✅ Model loaded successfully!")
    st.sidebar.info(f"Model Accuracy: {accuracy:.2%}")
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Input form
st.markdown("---")
st.header("📊 Enter Environmental Parameters")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input(
        "Temperature (°C)", 
        min_value=-10.0, 
        max_value=50.0, 
        value=25.0,
        step=0.1
    )
    
    humidity = st.number_input(
        "Humidity (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=60.0,
        step=0.1
    )
    
    pm25 = st.number_input(
        "PM2.5 (µg/m³)", 
        min_value=0.0, 
        max_value=500.0, 
        value=15.0,
        step=0.1
    )
    
    pm10 = st.number_input(
        "PM10 (µg/m³)", 
        min_value=0.0, 
        max_value=500.0, 
        value=30.0,
        step=0.1
    )

with col2:
    no2 = st.number_input(
        "NO2 (µg/m³)", 
        min_value=0.0, 
        max_value=200.0, 
        value=20.0,
        step=0.1
    )
    
    so2 = st.number_input(
        "SO2 (µg/m³)", 
        min_value=0.0, 
        max_value=200.0, 
        value=10.0,
        step=0.1
    )
    
    co = st.number_input(
        "CO (mg/m³)", 
        min_value=0.0, 
        max_value=10.0, 
        value=1.0,
        step=0.1
    )
    
    industrial_proximity = st.number_input(
        "Proximity to Industrial Areas (km)", 
        min_value=0.0, 
        max_value=50.0, 
        value=5.0,
        step=0.1
    )

population_density = st.number_input(
    "Population Density (per km²)", 
    min_value=0, 
    max_value=10000, 
    value=500,
    step=10
)

# Prediction button
st.markdown("---")
if st.button("🔮 Predict Air Quality", type="primary", use_container_width=True):
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Humidity': [humidity],
        'PM2.5': [pm25],
        'PM10': [pm10],
        'NO2': [no2],
        'SO2': [so2],
        'CO': [co],
        'Proximity_to_Industrial_Areas': [industrial_proximity],
        'Population_Density': [population_density]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    # Decode prediction
    quality_map = {
        0: "Good",
        1: "Moderate", 
        2: "Poor",
        3: "Hazardous"
    }
    
    quality = quality_map[prediction]
    st.markdown("## 📈 Prediction Result")
    if quality == "Good":
        color = "green"
        icon = "✅"
    elif quality == "Moderate":
        color = "orange"
        icon = "⚠️"
    elif quality == "Poor":
        color = "red"
        icon = "❗"
    else:  # Hazardous
        color = "darkred"
        icon = "☠️"
    
    st.markdown(
        f"<h2 style='text-align: center; color: {color};'>{icon} Air Quality: {quality}</h2>",
        unsafe_allow_html=True
    )
    st.markdown("### Prediction Confidence:")
    
    classes = ['Good', 'Moderate', 'Poor', 'Hazardous']
    for i, prob in enumerate(prediction_proba):
        st.progress(float(prob), text=f"{classes[i]}: {prob:.1%}")
st.markdown("---")
st.markdown("💡 *Note: This is a predictive model based on environmental data.*")