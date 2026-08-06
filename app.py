import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="House Price Predictor", page_icon="🏡", layout="wide"
)

# --- Custom Dark Minimalist UI & Sidebar Fix ---
st.markdown(
    """
    <style>
    /* Global Dark Theme */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Force Sidebar to Match Dark Theme */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    
    [data-testid="stSidebar"] * {
        color: #f3f4f6 !important;
    }
    
    /* Sleek Card Containers with Smooth Fade-in */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .card-container {
        background: #111827;
        border: 1px solid #1f2937;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        animation: fadeIn 0.5s ease-out forwards;
        margin-top: 20px;
    }
    
    /* Fix Input Labels & Text Visibility */
    label, .stSlider p, .stNumberInput label {
        color: #e5e7eb !important;
        font-weight: 500 !important;
    }
    
    /* Glowing Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f9fafb !important;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the trained model and scaler with cached resource loading
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
  st.title("🎛️ Control Panel")
  st.markdown("---")
  st.markdown("### About Model")
  st.info(
      "This application uses a regularized **Ridge Regression** architecture"
      " optimized for residential property valuation with historical and"
      " future projection algorithms."
  )
  st.markdown("---")
  target_year = st.slider(
      "📅 Target Prediction Year",
      min_value=2006,
      max_value=2035,
      value=2026,
      help=(
          "Select a future year to forecast valuation accounting for market"
          " appreciation."
      ),
  )

# --- Main Page Layout ---
st.title("🏡 Advanced Real Estate Valuation Engine")
st.markdown(
    "Provide structural parameters below. The system validates boundaries and"
    " projects valuations dynamically."
)
st.markdown("---")

st.subheader("📊 Primary Property Characteristics")

# Inputs with strict boundary warnings matching dataset limitations
col1, col2 = st.columns(2)

with col1:
  sqft_grlivarea = st.number_input(
      "Above Grade Living Area (sqft)",
      min_value=334,
      max_value=5642,
      value=1500,
      help="Dataset limits: 334 sqft to 5,642 sqft.",
  )
  bedrooms_abvgr = st.number_input(
      "Bedrooms Above Grade",
      min_value=0,
      max_value=8,
      value=3,
      help="Dataset limits: 0 to 8 bedrooms.",
  )
  full_bath = st.number_input(
      "Full Bathrooms",
      min_value=0,
      max_value=3,
      value=2,
      help="Dataset limits: 0 to 3 full baths.",
  )

with col2:
  overall_qual = st.slider(
      "Overall Material & Finish Quality",
      min_value=1,
      max_value=10,
      value=6,
      help="Scale from 1 (Very Poor) to 10 (Very Excellent).",
  )
  year_built = st.number_input(
      "Year Built",
      min_value=1872,
      max_value=2010,
      value=2000,
      help="Dataset limits: 1872 to 2010.",
  )
  total_basement = st.number_input(
      "Total Basement Area (sqft)",
      min_value=0,
      max_value=6110,
      value=850,
      help="Dataset limits: 0 to 6,110 sqft.",
  )

st.markdown("---")

# Action Button Layout
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
  predict_button = st.button("🔮 Compute Future Valuation")

if predict_button:
  with st.spinner("Processing feature vector and analyzing trends..."):
    default_medians = {
        "OverallQual": 6,
        "OverallCond": 5,
        "YearBuilt": 1973,
        "YearRemodAdd": 1994,
        "TotalBsmtSF": 991,
        "1stFlrSF": 1087,
        "GrLivArea": 1500,
        "FullBath": 2,
        "BedroomAbvGr": 3,
        "TotRmsAbvGrd": 6,
        "GarageCars": 2,
        "GarageArea": 480,
    }

    feature_values = {
        feature: default_medians.get(feature, 1) for feature in model_features
    }

    feature_values["GrLivArea"] = sqft_grlivarea
    feature_values["BedroomAbvGr"] = bedrooms_abvgr
    feature_values["FullBath"] = full_bath
    feature_values["OverallQual"] = overall_qual
    feature_values["YearBuilt"] = year_built
    feature_values["TotalBsmtSF"] = total_basement
    feature_values["YrSold"] = 2010

    input_df = pd.DataFrame([feature_values], columns=model_features)

    scaled_data = scaler.transform(input_df)
    base_prediction = model.predict(scaled_data)[0]

    years_diff = target_year - 2010
    appreciation_rate = 0.035
    projected_prediction = base_prediction * (
        (1 + appreciation_rate) ** max(0, years_diff)
    )

  st.markdown(
      f"""
    <div class="card-container">
        <h3 style="margin-top:0; color:#60a5fa;">🏷️ Valuation Results for Year {target_year}</h3>
        <p style="font-size: 1.1rem; color: #d1d5db; margin-bottom: 5px;">Base Model Output (2010 baseline): <b>₹{base_prediction:,.2f}</b></p>
        <hr style="border-color: #374151;">
        <h2 style="color: #34d399; margin-bottom: 0;">Projected Market Price: ₹{projected_prediction:,.2f}</h2>
    </div>
    """,
      unsafe_allow_html=True,
  )
  st.balloons()
