# 🏡 House Price Prediction App

An advanced, interactive machine learning web application built with **Streamlit** and **Scikit-Learn** that predicts residential property values using a trained **Ridge Regression** model. It features future market appreciation forecasting extending up to **2050** and a custom-styled, minimalist dark mode interface.

🔗 **Live App Demo:** [https://house-price-prediction-acjsb.streamlit.app/](https://house-price-prediction-acjsb.streamlit.app/)

---

## 🌟 Comprehensive Overview
This project bridges data science and full-stack web deployment, taking a trained machine learning pipeline from a Google Colab notebook and transforming it into a production-ready cloud application. It handles automated feature alignment, standard scaling normalization, boundary validation, and real-time inference calculations.

---

## 🚀 Core Features & Functionalities

- **Dynamic Real Estate Valuation:** 
  Instantly estimates residential market values based on critical structural parameters:
  - Above Grade Living Area ($\text{sqft}$)
  - Bedrooms Above Grade (Strictly bounded $\ge 1$)
  - Full Bathrooms (Strictly bounded $\ge 1$)
  - Overall Material & Finish Quality (Scale 1–10)
  - Year Built (Validated historical dataset limits)
  - Total Basement Area ($\text{sqft}$)
- **Future Year Price Forecasting:** 
  Includes an interactive sidebar control panel allowing users to target any future year up to **2050**. The application dynamically calculates and overlays compound annual real estate appreciation rates (~3.5% per year past the baseline) onto the base model predictions.
- **Strict Boundary Validation & Imputation:** 
  Prevents out-of-bounds errors and counter-intuitive low pricing bugs by enforcing dataset-accurate minimum/maximum limits and utilizing intelligent median baseline imputations for unselected features.
- **Minimalist Dark Theme UI & Animations:** 
  Crafted using custom CSS injections for a sleek dark mode layout (`#0b0f19`), responsive card containers, glowing action buttons, smooth fade-in animations, and custom visual feedback triggers (`st.snow`).

---

## 🛠️ Complete Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Core Language** | Python | Primary scripting and logic layer |
| **Frontend Framework** | Streamlit | Interactive web dashboard and user interface |
| **Machine Learning** | Scikit-Learn | Ridge Regression regression modeling & preprocessing |
| **Data Processing** | Pandas & NumPy | High-performance feature vector handling and data arrays |
| **Model Serialization** | Joblib / Pickle | Saving and loading trained model weights and scalers |

---

## 📂 Project Structure & File Architecture

```text
House-Price-Prediction/
│
├── app.py                     # Main Streamlit application script (UI, Dark Theme & Projections)
├── ridge_regression_model.pkl # Trained Ridge Regression model coefficients
├── scaler.pkl                 # Fitted StandardScaler for input normalization
└── requirements.txt           # Project dependencies list
