# bigmart_sales_predictor
A Machine Learning web app that predicts retail product sales revenue using XGBoost. Built on BigMart Sales Dataset with feature engineering, hyperparameter tuning, and deployed on Streamlit.
🔗 Live App
https://bigmartsalespredictor-9mgjw2zf4tezuxktftxdhq.streamlit.app/

 Project Overview
- Dataset: BigMart Sales Dataset (Kaggle) — 8,523 training rows
- Problem Type:Supervised Regression (not time series)
- Target Variable: Item_Outlet_Sales (₹ revenue)
- Best Model: XGBoost (Tuned)

ML Pipeline
1. Data Cleaning (missing values, label fixing)
2. Log Transformation (skewness fix)
3. Label Encoding (train-only fit)
4. Feature Engineering (Outlet_Age, MRP_per_Visibility, Is_Food_Item)
5. Model Training (Linear Regression, Random Forest, XGBoost)
6. Hyperparameter Tuning (RandomizedSearchCV — 30 combinations)
7. Cross Validation (5-fold)
8. Deployment (Streamlit Cloud)


