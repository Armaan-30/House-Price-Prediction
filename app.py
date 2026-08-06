import joblib  # Using joblib as it's standard for sklearn models
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="House Price Predictor", page_icon="🏡", layout="wide"
)

# Load the trained model and scaler with cached resource loading for speed
@st.cache_resource
def load_assets():
  try:
    model = joblib.load("ridge_regression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler
  except FileNotFoundError:
    return None, None


model, scaler = load_assets()

if model is None or scaler is None:
  st.error(
      "🚨 Error: Model or scaler file not found. Ensure"
      " 'ridge_regression_model.pkl' and 'scaler.pkl' are in the root directory."
  )
  st.stop()

# Define the exact 79 feature names your model was trained on
model_features = [
    "MSSubClass",
    "MSZoning",
    "LotFrontage",
    "LotArea",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "MasVnrArea",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinSF1",
    "BsmtFinType2",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "KitchenQual",
    "TotRmsAbvGrd",
    "Functional",
    "Fireplaces",
    "FireplaceQu",
    "GarageType",
    "GarageYrBlt",
    "GarageFinish",
    "GarageCars",
    "GarageArea",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MiscVal",
    "MoSold",
    "YrSold",
    "SaleType",
    "SaleCondition",
]

# --- Sidebar ---
with st.sidebar:
  st.image(
      "https://images.unsplash.com/photo-1564013799919-ab600027ffc6?auto=format&fit=crop&w=600&q=80",
      use_container_width=True,
  )
  st.title("About App")
  st.info(
      "This application leverages a **Ridge Regression** machine learning"
      " pipeline trained on housing data to estimate property market values"
      " based on key structural features."
  )
  st.markdown("---")
  st.markdown("### 🛠️ Tech Stack")
  st.markdown(
      "- Python • Streamlit\n- Scikit-Learn • Pandas\n- Ridge Regression Model"
  )

# --- Main Page Layout ---
st.title("🏡 Ames Housing Price Prediction")
st.markdown(
    "Provide the core structural attributes of the property below to generate an"
    " instant valuation estimate."
)
st.markdown("---")

st.subheader("📊 Key Property Characteristics")

# Creating a clean 2-column layout for inputs
col1, col2 = st.columns(2)

with col1:
  sqft_grlivarea = st.number_input(
      "Above Grade Living Area (sqft)",
      min_value=334,
      max_value=5642,
      value=1500,
      help="Total square footage of living area above ground.",
  )
  bedrooms_abvgr = st.number_input(
      "Bedrooms Above Grade",
      min_value=0,
      max_value=8,
      value=3,
      help="Number of bedrooms above basement level.",
  )
  full_bath = st.number_input(
      "Full Bathrooms",
      min_value=0,
      max_value=3,
      value=2,
      help="Full bathrooms above ground.",
  )

with col2:
  overall_qual = st.slider(
      "Overall Material & Finish Quality",
      min_value=1,
      max_value=10,
      value=7,
      help="Rate the overall material and finish of the house (1 = Very Poor, 10"
      " = Very Excellent).",
  )
  year_built = st.number_input(
      "Year Built",
      min_value=1872,
      max_value=2010,
      value=2000,
      help="Original construction date.",
  )

st.markdown("---")

# Center alignment for the prediction action button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
  predict_button = st.button(
      "🔮 Predict Sale Price", use_container_width=True, type="primary"
  )

if predict_button:
  # Create a spinner while processing prediction
  with st.spinner("Analyzing property metrics and calculating valuation..."):
    # Create feature dictionary initialized to 0 defaults
    feature_values = {feature: 0 for feature in model_features}

    # Map user parameters
    feature_values["GrLivArea"] = sqft_grlivarea
    feature_values["BedroomAbvGr"] = bedrooms_abvgr
    feature_values["FullBath"] = full_bath
    feature_values["OverallQual"] = overall_qual
    feature_values["YearBuilt"] = year_built

    # Convert to DataFrame matching exact model columns order
    input_df = pd.DataFrame([feature_values], columns=model_features)

    # Transform data and predict
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)

  # Results display box
  st.markdown("### 🏷️ Valuation Results")
  st.success(
      f"### Estimated Market Sale Price: **${prediction[0]:,.2f}**", icon="💰"
  )
  st.balloons()
