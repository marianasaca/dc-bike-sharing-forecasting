# 🚲 DC Bike Sharing Forecasting  
Machine learning analysis & forecasting of hourly bike demand in Washington, D.C.  
Includes data exploration, feature engineering, model training, evaluation, and a Streamlit dashboard.

---

## 📌 Project Overview

This project analyzes the Washington D.C. bike sharing dataset to understand the factors that drive hourly ridership and to build predictive models that forecast demand.

The workflow includes:

- Exploratory Data Analysis (EDA)
- Time series visualization
- Feature engineering (lags, weather variables, seasonality)
- Model training (Random Forest, Gradient Boosting, Ridge, Linear Regression)
- Model evaluation using MAE, RMSE, and R²
- Residual and error analysis
- Interactive **Streamlit dashboard** for exploration and forecasting

---

## 🗂 Repository Structure

dc-bike-sharing-forecasting/
│
├── app.py # Streamlit dashboard
├── bike_sharing_analysis_and_modeling.ipynb # Main analysis notebook
├── requirements.txt # Dependencies
├── README.md
│
├── data/ # Datasets
│ ├── bike-sharing-hourly.csv # Raw dataset
│ └── bike_hourly_with_features.csv # Engineered dataset
│
└── figures/ # Saved visualizations (Plotly HTML)
├── box_by_hour.html
├── box_by_season.html
├── heatmap_weekday_hour.html
├── pred_vs_actual.html
├── residual_hist.html
├── residuals_by_hour.html
├── scatter_atemp.html
├── scatter_hum.html
├── scatter_temp.html
├── scatter_windspeed.html
└── time_series_total_rides.html

## 📊 Key Visualizations

Interactive figures are available in the `figures/` folder:

- Hourly ridership distribution (boxplot)
- Seasonal ridership patterns
- Weekday vs hour heatmap
- Temperature, humidity, windspeed relationships
- Predictions vs actual values
- Residual analysis (histogram + hourly boxplot)
- Total ridership time series

---

## ⚙️ How to Run the Streamlit Dashboard

### 1️⃣ Install dependencies  
pip install -r requirements.txt

### 2️⃣ Run the app  
streamlit run app.py

This opens the interactive dashboard, where you can:

- Explore all visualizations  
- View model results  
- Compare predictions vs actual values  
- Inspect residuals and feature importance  

---

## 📘 Jupyter Notebook Contents

The notebook provides:

- Data cleaning  
- Exploratory analysis  
- Visualizations  
- Time series analysis  
- Feature engineering  
- Model training  
- Hyperparameter tuning  
- Residual diagnostics  
- Final model evaluation  

It serves as the full analytical workflow behind the dashboard.

---

## 🧰 Tech Stack

- **Python**
- pandas, NumPy  
- scikit-learn  
- statsmodels  
- Plotly  
- Streamlit  
- Jupyter Notebook  

---

## 🌟 Highlights

- Interactive Streamlit dashboard  
- Extensive EDA and weather-based analysis  
- Feature engineering for improved predictions  
- Multiple ML models with comparison  
- Full set of interactive HTML visualizations  
- Clean, professional repo structure  

---

## 👩‍💻 Author

**Mariana Saca**  
Data Analyst — Python | SQL | Machine Learning  
- LinkedIn: https://www.linkedin.com/in/marianasaca/  
- GitHub: https://github.com/marianasaca  

📧 **msaca16@gmail.com**

---

