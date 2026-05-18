
import streamlit as st
import pickle
import numpy as np

# =========================
# Load Model
# =========================

with open("/content/car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# Load Encoders
# =========================

with open("/content/fuel_encoder.pkl", "rb") as f:
    fuel_encoder = pickle.load(f)

with open("/content/seller_encoder.pkl", "rb") as f:
    seller_encoder = pickle.load(f)

with open("/content/transmission_encoder.pkl", "rb") as f:
    transmission_encoder = pickle.load(f)

with open("/content/car_encoder.pkl", "rb") as f:
    car_encoder = pickle.load(f)

# =========================
# Streamlit App Title
# =========================

st.title("🚗 Car Price Prediction App")

st.write("Enter car details below:")

# =========================
# User Inputs
# =========================

car_name = st.selectbox(
    "Car Name",
    car_encoder.classes_
)

year = st.number_input(
    "Manufacturing Year",
    min_value=2000,
    max_value=2025,
    value=2015
)

present_price = st.number_input(
    "Present Price (in Lakhs)",
    min_value=0.0,
    max_value=100.0,
    value=5.0
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=500000,
    value=30000
)

fuel_type = st.selectbox(
    "Fuel Type",
    fuel_encoder.classes_
)

seller_type = st.selectbox(
    "Seller Type",
    seller_encoder.classes_
)

transmission = st.selectbox(
    "Transmission",
    transmission_encoder.classes_
)

owner = st.number_input(
    "Number of Previous Owners",
    min_value=0,
    max_value=5,
    value=0
)

# =========================
# Encode Inputs
# =========================

car_name_encoded = car_encoder.transform([car_name])[0]
fuel_encoded = fuel_encoder.transform([fuel_type])[0]
seller_encoded = seller_encoder.transform([seller_type])[0]
transmission_encoded = transmission_encoder.transform([transmission])[0]

# =========================
# Prepare Features
# =========================

features = np.array([[
    car_name_encoded,
    year,
    present_price,
    kms_driven,
    fuel_encoded,
    seller_encoded,
    transmission_encoded,
    owner
]])

# =========================
# Prediction
# =========================

if st.button("Predict Price"):

    prediction = model.predict(features)

    st.success(
        f"💰 Estimated Car Selling Price: {prediction[0]:.2f} Lakhs"
    )
