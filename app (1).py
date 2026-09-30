import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Section 8: Model Deployment with Streamlit
# Best model selected in the notebook: Decision Tree Regression
# Selected features from the notebook:
# vehicle_age, mileage, engine, max_power, seats
# ---------------------------------------------------------

st.set_page_config(
    page_title="Car Selling Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# Load the trained Decision Tree model
model = joblib.load("decision_tree_model.pkl")

st.title("🚗 Car Selling Price Prediction")
st.write(
    "Enter the vehicle details below to predict its selling price."
)

st.subheader("Vehicle Information")

# Inputs correspond exactly to the five selected features
vehicle_age = st.number_input(
    "Vehicle Age (years)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=1.0
)

mileage = st.number_input(
    "Mileage (km/l)",
    min_value=0.0,
    max_value=100.0,
    value=18.0,
    step=0.1
)

engine = st.number_input(
    "Engine (CC)",
    min_value=0.0,
    max_value=10000.0,
    value=1200.0,
    step=50.0
)

max_power = st.number_input(
    "Max Power (bhp)",
    min_value=0.0,
    max_value=2000.0,
    value=80.0,
    step=1.0
)

seats = st.number_input(
    "Number of Seats",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

# Prediction button
if st.button("Predict Selling Price", type="primary"):

    # Create input data using the exact feature names
    input_data = pd.DataFrame({
        "vehicle_age": [vehicle_age],
        "mileage": [mileage],
        "engine": [engine],
        "max_power": [max_power],
        "seats": [seats]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Selling Price: {prediction:,.2f}"
    )

    st.info(
        "The prediction was generated using the Decision Tree Regression model."
    )
