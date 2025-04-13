import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# App title
st.title("🏡 US Home Price Predictor")
st.write("Predict S&P Case-Shiller Home Price Index based on key economic indicators.")

# User inputs
mortgage_rate = st.slider("30-Year Mortgage Rate (%)", 0.0, 10.0, 6.0)
unemployment_rate = st.slider("Unemployment Rate (%)", 0.0, 15.0, 4.0)
cpi = st.slider("CPI (Consumer Price Index)", 100.0, 350.0, 300.0)
fed_funds_rate = st.slider("Federal Funds Rate (%)", 0.0, 10.0, 5.0)
building_permits = st.number_input("Building Permits (in thousands)", min_value=0.0, value=1300.0)
consumer_sentiment = st.slider("Consumer Sentiment Index", 50.0, 120.0, 70.0)

# Predict
if st.button("Predict Home Price Index"):
    input_data = np.array([[mortgage_rate, unemployment_rate, cpi,
                            fed_funds_rate, building_permits, consumer_sentiment]])
    prediction = model.predict(input_data)[0]
    st.success(f"📈 Predicted Home Price Index: {prediction:.2f}")
