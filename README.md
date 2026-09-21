# 💧 Water Quality Prediction — ML Capstone

**Course:** 23CSE301 Machine Learning (2026-27)  
**Dataset:** USGS Continuous In-Situ Water Quality Telemetry (`water_dataset.mat`)  
**Target:** Predicting median daily water pH across 37 Georgia stream monitoring stations  

## 🚀 Live Dashboard

This project includes an **interactive Streamlit dashboard** that:
- Trains & compares all **10 regression algorithms** in real-time
- Displays full EDA with interactive Plotly charts
- Lets you **predict pH with custom sensor inputs**

### Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit dashboard (main entry point) |
| `regression_final.ipynb` | Full regression notebook (executed) |
| `make_notebook.py` | Notebook generator script |
| `data/water_dataset.mat` | USGS Water Quality dataset |
| `requirements.txt` | Python dependencies |

## 🏆 Model Results (Regression)

| Rank | Model | R² | RMSE |
|------|-------|----|------|
| 1 | Gradient Boosting | 0.9156 | 0.0084 |
| 2 | SVR (RBF) | 0.9079 | 0.0088 |
| 3 | Random Forest | 0.9060 | 0.0089 |
| 4 | Polynomial (Deg 2) | 0.8983 | 0.0092 |
| 5 | KNN | 0.8958 | 0.0094 |

## 🧪 Algorithms Implemented

1. Linear Regression (OLS)
2. Ridge Regression (L2)
3. Lasso Regression (L1)
4. ElasticNet (L1+L2)
5. Polynomial Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. Support Vector Regressor (SVR)
10. K-Nearest Neighbors (KNN)

## 👥 Team

23CSE301 Machine Learning Capstone — Academic Year 2026-27
