import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# ==========================================
# 1. PAGE CONFIG & UI STYLING
# ==========================================
st.set_page_config(page_title="AgriForecast AI", layout="wide", page_icon="🌾")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #1f77b4 !important; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #333333 !important; font-size: 1.1rem; }
    div[data-testid="metric-container"] {
        background-color: #ffffff; border: 1px solid #e6e9ef;
        padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA & MODEL LOADING (DELTA LOGIC)
# ==========================================
@st.cache_resource
def load_artifacts():
    # Loading the new Delta Regressor and the Classifier
    regressor = joblib.load('models/agri_delta_regressor.pkl')
    classifier = joblib.load('models/agri_failure_classifier.pkl')
    # Load baselines for delta reconstruction
    baselines = pd.read_csv('data/processed/baselines.csv')
    # Load master data for dropdowns
    master = pd.read_csv('data/processed/agriforecast_master.csv')
    return regressor, classifier, baselines, master

reg_model, clf_model, baseline_df, master_data = load_artifacts()

# ==========================================
# 3. MAIN INPUT PANEL (DYNAMIC & SMART DEFAULTS)
# ==========================================
st.title("🌾 AgriForecast: Intelligent Yield Risk Assessment")
st.write("Configure parameters and click **'Run Forecast'**. Sliders now impact results in real-time.")

with st.container():
    col_a, col_b = st.columns(2)
    with col_a:
        # 1. Setup Area with a safe default (India)
        areas = sorted(master_data['Area'].unique())
        default_area = areas.index('India') if 'India' in areas else 0
        selected_area = st.selectbox("Select Country/Region", areas, index=default_area)
        
        # 2. DYNAMIC FILTER: Only show crops that exist in the selected Area
        valid_crops = sorted(master_data[master_data['Area'] == selected_area]['Item'].unique())
        default_crop = valid_crops.index('Potatoes') if 'Potatoes' in valid_crops else 0
        selected_item = st.selectbox("Select Crop Type", valid_crops, index=default_crop)
        
        year = st.number_input("Forecast Year", 2026, 2035, 2026)
        
    with col_b:
        temp = st.number_input("Expected Temperature (°C)", -10.0, 50.0, 25.0)
        rain = st.number_input("Expected Rainfall (mm/year)", 0.0, 5000.0, 1200.0)
        pest = st.number_input("Pesticide Volume (Tonnes)", 0.0, 100000.0, 5000.0)

run_prediction = st.button("🚀 Run Forecast & Risk Analysis", type="primary", use_container_width=True)

st.divider()

# ==========================================
# 4. PREDICTION & VISUALIZATION
# ==========================================
tab1, tab2 = st.tabs(["📊 Prediction Dashboard", "🔬 Deep Analytics & Model Diagnostics"])

with tab1:
    if run_prediction:
        # A. Look up the Historical Median Baseline (WITH SAFETY NET)
        match = baseline_df[(baseline_df['Area'] == selected_area) & (baseline_df['Item'] == selected_item)]
        
        if match.empty:
            st.error(f"❌ No historical baseline data found for **{selected_item}** in **{selected_area}**. Please select a different combination.")
            st.stop() # This safely halts the script right here so it doesn't crash!
            
        hist_median = match['median_baseline'].values[0]

        # B. Prepare Input Data (Including the 'Climate_Stress' feature)
        input_df = pd.DataFrame({
            'Year': [year], 'Pesticides_Tonnes': [pest], 
            'Temperature_C': [temp], 'Rainfall_mm': [rain],
            'Climate_Stress': [temp / (rain + 1)], # Engineered interaction
            'Area': [selected_area], 'Item': [selected_item]
        })
        input_df['Area'] = input_df['Area'].astype('category')
        input_df['Item'] = input_df['Item'].astype('category')

        # C. Predict the % Shift (Delta) and Calculate Base Yield
        predicted_delta = reg_model.predict(input_df)[0]
        final_yield = hist_median * (1 + predicted_delta)

        # D. Predict Base Failure Probability
        prob_failure = clf_model.predict_proba(input_df.drop(columns=['Climate_Stress']))[0][1]

        # ==========================================
        # 🌟 THE EXPERT SYSTEM OVERRIDE 🌟
        # Handling XGBoost's inability to extrapolate extreme weather
        # ==========================================
        agronomic_penalty = 0.0
        
        # Heat Stress Rule: 5% yield drop for every degree over 33°C
        if temp > 33.0:
            agronomic_penalty += (temp - 33.0) * 0.05
            
        # Drought Stress Rule: 10% yield drop for every 100mm missing under 500mm
        if rain < 500.0:
            agronomic_penalty += ((500.0 - rain) / 100.0) * 0.10
            
        # Flood Rule: Massive drop if rainfall is biblical
        if rain > 3500.0:
            agronomic_penalty += 0.40

        # Apply the penalties (capped at a 95% total crop loss)
        agronomic_penalty = min(agronomic_penalty, 0.95)
        
        if agronomic_penalty > 0:
            final_yield = final_yield * (1 - agronomic_penalty)
            # Force the failure probability up based on extreme weather
            prob_failure = max(prob_failure, agronomic_penalty)

        # Final Risk Labeling
        is_failure = 1 if prob_failure >= 0.40 else 0
        
        # Calculate the final displayed Delta vs Historical Median
        final_delta_pct = ((final_yield - hist_median) / hist_median) * 100

        # --- Display Results ---
        out_col1, out_col2 = st.columns(2)
        with out_col1:
            st.metric(label="Predicted Yield", value=f"{final_yield:,.2f} hg/ha", 
                      delta=f"{final_delta_pct:.2f}% vs Historical Median")
        with out_col2:
            st.metric(label="Failure Probability", value=f"{prob_failure*100:.1f}%", 
                      delta="CRITICAL" if is_failure == 1 else "SAFE", 
                      delta_color="inverse" if is_failure == 1 else "normal")

        # --- Dynamic Comparison Chart ---
        st.subheader("📊 Live Comparison: Baseline vs. Prediction")
        comp_df = pd.DataFrame({
            'Category': ['Historical Median', 'Current Prediction'],
            'Yield (hg/ha)': [hist_median, final_yield]
        })
        fig = px.bar(comp_df, x='Category', y='Yield (hg/ha)', 
                     color='Category', 
                     color_discrete_map={'Historical Median': '#BDC3C7', 
                                         'Current Prediction': '#1f77b4' if is_failure == 0 else '#E74C3C'},
                     text_auto='.2s')
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: ANALYTICS (Matches all 22 generated SVGs) ---
with tab2:
    st.header("🔬 Full Enterprise Analytics Suite")
    fig_path = "outputs/"
    
    def safe_img(fn, cap):
        if os.path.exists(os.path.join(fig_path, fn)):
            st.image(os.path.join(fig_path, fn), caption=cap, use_container_width=True)

    # --- Section 1: Regressor Diagnostics ---
    st.subheader("1. Yield Delta Regressor Diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        safe_img("regressor_importance.svg", "Regression Factors")
        safe_img("prediction_scatter.svg", "Regression Quality (Actual vs Predicted)")
        safe_img("error_by_crop.svg", "Hardest Crops to Predict")
    with c2:
        safe_img("residual_distribution.svg", "Residual Distribution (Prediction Errors)")
        safe_img("prediction_tracking_line.svg", "Live Prediction Tracking")

    st.divider()

    # --- Section 2: Classifier Diagnostics ---
    st.subheader("2. Failure Classifier Diagnostics")
    c3, c4 = st.columns(2)
    with c3:
        safe_img("failure_confusion_matrix.svg", "Classifier Performance Matrix")
        safe_img("classifier_importance.svg", "Classification Factors")
        safe_img("roc_curve.svg", "Receiver Operating Characteristic (ROC)")
    with c4:
        safe_img("precision_recall_curve.svg", "Anomaly Detection Curve")
        safe_img("confidence_distribution.svg", "Model Confidence Distribution")

    st.divider()

    # --- Section 3: Global Trends ---
    st.subheader("3. Historical Patterns & Distributions")
    c5, c6 = st.columns(2)
    with c5:
        safe_img("correlation_matrix.svg", "Environmental Correlation Heatmap")
        safe_img("yield_trend_line.svg", "Global Historical Yield Trend")
        safe_img("variable_pairplot.svg", "Multi-Variable Pairplot (Anomalies Highlighted)")
    with c6:
        safe_img("yield_histogram.svg", "Distribution Density of Global Yields")
        safe_img("climate_jointplot.svg", "Global Agricultural Climate Profile")

    st.divider()

    # --- Section 4: Deep-Dive Analysis ---
    st.subheader("4. Regional & Crop Input Analysis")
    c7, c8 = st.columns(2)
    with c7:
        safe_img("top_countries_bar.svg", "Top 10 High-Yield Regions")
        safe_img("regional_yield_heatmap.svg", "Regional Yield Density Over Time")
    with c8:
        safe_img("crop_variance_boxplot.svg", "Yield Variance by Crop Species")

    st.divider()

    # --- Section 5: Advanced Temporal Analytics ---
    st.subheader("5. Advanced Stress & Temporal Analytics")
    c9, c10 = st.columns(2)
    with c9:
        safe_img("climate_stress_impact.svg", "Climate Stress vs. Yield Shift")
        safe_img("temporal_volatility_trend.svg", "Global Yield Volatility Trend")
    with c10:
        safe_img("stress_violin_plot.svg", "Stress Distribution during Normal vs Failure Years")
        safe_img("regional_anomaly_heatmap.svg", "Regional Yield Anomalies Over Time")