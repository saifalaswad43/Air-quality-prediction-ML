import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌍",
    layout="centered"
)
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .good { background-color: #00ff0044; border: 2px solid #00ff00; }
    .moderate { background-color: #ffff0044; border: 2px solid #ffff00; }
    .poor { background-color: #ff990044; border: 2px solid #ff9900; }
    .hazardous { background-color: #ff000044; border: 2px solid #ff0000; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🌍 Air Quality Prediction System</h1><p>Powered by Extra Trees Classifier</p></div>', unsafe_allow_html=True)
@st.cache_resource
def load_models():
    """تحميل النماذج المحفوظة"""
    with open('extra_trees_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = None
    
    return model, label_encoder, scaler

@st.cache_data
def load_data():
    """تحميل البيانات"""
    df = pd.read_csv('updated_pollution_dataset.csv')
    return df
try:
    df = load_data()
    model, label_encoder, scaler = load_models()
    feature_columns = [col for col in df.columns if col != 'Air Quality']
    X = df[feature_columns]
    y = df['Air Quality']
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/air-quality.png", width=100)
        st.title("🌍 Air Quality Predictor")
        st.markdown("---")
        
        st.success(f"✅ Model: Extra Trees Classifier")
        st.info(f"📊 Features: {len(feature_columns)}")
        st.info(f"📦 Samples: {len(df)}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.markdown("---")
            st.markdown("### 📋 Top Features")
            for i, row in feature_importance.head(4).iterrows():
                st.markdown(f"**{row['Feature']}**: {row['Importance']:.2%}")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Categories:**
        - 🟢 Good
        - 🟡 Moderate
        - 🟠 Poor
        - 🔴 Hazardous
        """)
        
        st.markdown("---")
        st.markdown(f"📁 **Files Loaded:** ✅")
        
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()
st.markdown("## 📊 Enter Environmental Parameters")
tab1, tab2, tab3 = st.tabs(["📝 Input Data", "📈 Visualizations", "ℹ️ Dataset Info"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.number_input("🌡️ Temperature (°C)", -10.0, 50.0, 25.0, 0.1)
        humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 60.0, 0.1)
        pm25 = st.number_input("🌫️ PM2.5 (µg/m³)", 0.0, 500.0, 15.0, 0.1)
        pm10 = st.number_input("🌪️ PM10 (µg/m³)", 0.0, 500.0, 30.0, 0.1)
        no2 = st.number_input("🏭 NO2 (ppb)", 0.0, 200.0, 20.0, 0.1)

    with col2:
        so2 = st.number_input("🏗️ SO2 (ppb)", 0.0, 200.0, 10.0, 0.1)
        co = st.number_input("🚗 CO (ppm)", 0.0, 10.0, 1.0, 0.1)
        industrial_proximity = st.number_input("🏢 Industrial Proximity (km)", 0.0, 50.0, 5.0, 0.1)
        population_density = st.number_input("👥 Population Density (per km²)", 0, 10000, 500, 10)
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Air Quality", type="primary", use_container_width=True)

    if predict_button:
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
        if scaler:
            input_scaled = scaler.transform(input_data)
            input_final = pd.DataFrame(input_scaled, columns=input_data.columns)
        else:
            input_final = input_data
        prediction = model.predict(input_final)[0]
        prediction_proba = model.predict_proba(input_final)[0]
        class_names = label_encoder.classes_
        predicted_class = class_names[prediction]
        color_map = {
            'Good': 'green',
            'Moderate': 'orange',
            'Poor': 'red',
            'Hazardous': 'darkred'
        }
        
        icon_map = {
            'Good': '✅',
            'Moderate': '⚠️',
            'Poor': '❗',
            'Hazardous': '☠️'
        }
        st.markdown("## 📈 Prediction Result")
        
        color = color_map.get(predicted_class, 'gray')
        icon = icon_map.get(predicted_class, '🔍')
        box_class = predicted_class.lower()
        
        st.markdown(
            f"""
            <div class="prediction-box {box_class}">
                <h1 style="font-size: 4rem;">{icon}</h1>
                <h2 style="color: {color};">Air Quality: {predicted_class}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("### 📊 Confidence Scores")
        for i, (cls, prob) in enumerate(zip(class_names, prediction_proba)):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"**{cls}:**")
            with col2:
                st.progress(float(prob), text=f"{prob:.1%}")

with tab2:
    st.markdown("## 📊 Visualizations")
    
    if st.checkbox("Show Feature Importance"):
        if hasattr(model, 'feature_importances_'):
            fig, ax = plt.subplots(figsize=(10, 5))
            feature_imp = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            ax.barh(feature_imp['Feature'], feature_imp['Importance'], color='skyblue')
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance - Extra Trees')
            st.pyplot(fig)
            plt.close()
    
    if st.checkbox("Show Class Distribution"):
        fig, ax = plt.subplots(figsize=(8, 5))
        class_counts = df['Air Quality'].value_counts()
        colors = ['green', 'orange', 'red', 'darkred']
        class_counts.plot(kind='bar', ax=ax, color=colors[:len(class_counts)])
        ax.set_title('Distribution of Air Quality Classes')
        ax.set_xlabel('Air Quality')
        ax.set_ylabel('Count')
        ax.set_xticklabels(class_counts.index, rotation=45)
        st.pyplot(fig)
        plt.close()

with tab3:
    st.markdown("## ℹ️ Dataset Information")
    
    st.markdown("### 📊 Data Overview")
    st.dataframe(df.head(10))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(feature_columns))
    with col3:
        st.metric("Target Classes", df['Air Quality'].nunique())
    
    st.markdown("### 📊 Statistical Summary")
    st.dataframe(df.describe())
    
    st.markdown("### 📊 Class Distribution")
    class_dist = df['Air Quality'].value_counts()
    st.dataframe(class_dist)
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>🌍 Air Quality Prediction System | Built with Streamlit & Extra Trees</p>
    </div>
    """,
    unsafe_allow_html=True
)