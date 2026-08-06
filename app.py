import joblib
import numpy as np
import streamlit as st

# Load the trained model and scaler
model = joblib.load("ridge_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏡 House Price Prediction App")
st.write(
    "Enter the property details below to predict the estimated house price."
)

# Example input fields (Modify these variables to match your dataset's features)
# For instance, if your model takes square footage, bedrooms, and bathrooms:
sqft = st.number_input("Square Footage", min_value=500, max_value=10000, value=2000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input(
    "Number of Bathrooms", min_value=1, max_value=10, value=2
)

if st.button("Predict Price"):
  # Combine inputs into an array
  input_data = np.array([[sqft, bedrooms, bathrooms]])

  # Scale the input using your saved scaler
  scaled_data = scaler.transform(input_data)

  # Make prediction
  prediction = model.predict(scaled_data)

  # Display result
  st.success(
      f"Estimated House Price: ${prediction[0]:,.2f}"
  )