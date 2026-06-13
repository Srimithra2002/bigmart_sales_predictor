import streamlit as st
import pandas as pd
import joblib


model = joblib.load("retail_sales_model.pkl")


st.title("Retail Sales Prediction")


mrp = st.number_input(
    "Item MRP",
    value=200.0
)

weight = st.number_input(
    "Item Weight",
    value=12.0
)

visibility = st.number_input(
    "Item Visibility",
    value=0.05
)

age = st.number_input(
    "Outlet Age",
    value=10
)


if st.button("Predict Sales"):

    data = pd.DataFrame({
        "Item_MRP": [mrp],
        "Item_Weight": [weight],
        "Item_Visibility": [visibility],
        "Outlet_Age": [age]
    })


    prediction = model.predict(data)

    st.success(
        "Predicted Sales = ₹ " + str(prediction[0])
    )