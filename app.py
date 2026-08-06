import streamlit as st
import pickle
import pandas as pd
import numpy as np # Import numpy

# Load the trained model and scaler
try:
    with open('ridge_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Error: Model or scaler file not found. Make sure 'ridge_regression_model.pkl' and 'scaler.pkl' are in the same directory as this app.py file.")
    st.stop() # Stop the app if files are not found

st.title('🏡 House Price Prediction App')
st.write('Enter the property details below to predict the estimated house price.')

# Define the exact 79 feature names your model was trained on, in order
# This list was derived from x.columns in the notebook before scaling.
model_features = [
    'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
    'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
    'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
    'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
    'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
    'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
    'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
    'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
    'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',
    'MoSold', 'YrSold', 'SaleType', 'SaleCondition'
]

st.header("Property Characteristics")

# User input fields for key features (corresponding to original training features)
# You can add more input fields here, but for any feature not explicitly asked,
# a default value will be used.

# Note: Some features like Alley, PoolQC, Fence, MiscFeature had many NaNs and were dropped by df.dropna()
# In a real app, you'd handle these gracefully, potentially providing 'None' or the most common value.
# For now, we'll assign 0 as a placeholder default for all non-user-input features.

# Minimum and maximum values are taken from the original dataset for guidance.
sqft_grlivarea = st.number_input("Above Grade (Ground) Living Area (sqft)", min_value=334, max_value=5642, value=1500)
bedrooms_abvgr = st.number_input("Bedrooms Above Grade", min_value=0, max_value=8, value=3)
full_bath = st.number_input("Full Bathrooms", min_value=0, max_value=3, value=2)
overall_qual = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=7)
year_built = st.number_input("Year Built", min_value=1872, max_value=2010, value=2000)

# Example of how to add a categorical feature, assuming it was label encoded
# You would need to know the mapping from original categories to their encoded numbers
# For simplicity, if MSZoning was encoded as {'RL': 3, ...}, you'd map user choice to 3.
# mszoning_options = {'RL': 3, 'RM': 4, 'FV': 0, 'RH': 2, 'C (all)': 1}
# selected_mszoning_display = st.selectbox('MS Zoning', options=list(mszoning_options.keys()), index=0)
# mszoning_encoded = mszoning_options[selected_mszoning_display]

if st.button('Predict Sale Price'):
    # Create a dictionary to hold all 79 feature values, initialized to 0
    feature_values = {feature: 0 for feature in model_features}
    
    # Populate with user inputs
    feature_values['GrLivArea'] = sqft_grlivarea
    feature_values['BedroomAbvGr'] = bedrooms_abvgr
    feature_values['FullBath'] = full_bath
    feature_values['OverallQual'] = overall_qual
    feature_values['YearBuilt'] = year_built
    
    # If you had more inputs, you would map them here:
    # feature_values['MSZoning'] = mszoning_encoded # Example for categorical
    # feature_values['LotArea'] = st.number_input('Lot Area', ...)

    # Create a DataFrame from the feature values, ensuring correct column order
    input_df = pd.DataFrame([feature_values], columns=model_features)

    # Scale the input features
    scaled_data = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_data)

    # Display result
    st.success(
        f'The estimated Sale Price is: ${prediction[0]:,.2f}'
    )
