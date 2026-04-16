# 🌾 AgriForecast: Intelligent Yield Risk Assessment

An end-to-end Machine Learning pipeline and interactive Streamlit dashboard that predicts crop yield shifts and calculates catastrophic failure probabilities based on climate stress and pesticide usage.

## 📊 Data Source
The models in this project were trained using the **Crop Yield Prediction Dataset**, available publicly on Kaggle:
👉 [Crop Yield Prediction Dataset by patelris](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)

## 🚀 Features
* **Dual XGBoost Architecture:** Features a Regressor for yield delta prediction and a Classifier for failure probability.
* **Engineered Climate Stress:** Custom feature interactions mapping extreme weather (heat/flood) to crop survivability.
* **Enterprise Analytics Dashboard:** 22 dynamically generated visualizations mapping temporal volatility, regional anomalies, and model diagnostics.
* **Bulletproof Testing:** Fully automated Pytest/Selenium suite built to bypass React virtualized DOM limitations.

## 🛠️ Tech Stack
* **Modeling:** Python, XGBoost, Scikit-Learn, Pandas
* **Frontend:** Streamlit, Plotly
* **Analytics:** Seaborn, Matplotlib
* **Testing:** Pytest, Selenium WebDriver

## 📦 How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the pipeline to fuse data and train models: Execute `master_pipeline.ipynb`
4. Launch the app: `streamlit run app.py`
5. Run the test suite: `pytest test.py -v`