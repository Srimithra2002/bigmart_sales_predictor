# bigmart_sales_predictor
BigMart Retail Sales Predictor is an end-to-end machine learning project that predicts how much revenue a product will generate in a specific retail store, and uses that prediction to alert whether current stock is too low, too high, or just right.

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

I trained three models — Linear Regression as a baseline (~55% R²), Random Forest using bagging (~58% R²), and XGBoost using sequential boosting (~63–65% R² after RandomizedSearchCV tuning). XGBoost was selected as the final model, saved using joblib, and deployed as a live Streamlit web application."
