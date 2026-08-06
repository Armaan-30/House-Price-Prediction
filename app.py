import joblib
import numpy as np
import streamlit as st

# Load the trained model and scaler
model = joblib.load("ridge_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏡 House Price Prediction App")
st.write(
    "Enter all the required property features below to predict the price."
)

# Create input widgets for EVERY feature your model was trained on
# Make sure the number and order match what Colab expects!
feature1 = st.number_input("Feature 1 Name", value=0.0)
feature2 = st.number_input("Feature 2 Name", value=0.0)
# ... add input fields for all features ...

if st.button("Predict Price"):
  # Collect all inputs into a single 2D array in the correct order
  input_data = np.array([[feature1, feature2]])  # Add all your features here

  # Scale the input data using the loaded scaler
  scaled_data = scaler.transform(input_data)

  # Make prediction
  prediction = model.predict(scaled_data)

  # Display the result
  st.success(f"Estimated House Price: ${prediction[0]:,.2f}")
