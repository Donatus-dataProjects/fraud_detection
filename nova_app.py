import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Load the model
try:
    loaded_model = pickle.load(open('nova_model.sav', 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")

# Function for Fraud Prediction 
def predict_fraud(input_data):
    convert_data_to_numpy = np.asarray(input_data)
    reshape_input_data = convert_data_to_numpy.reshape(1, -1)

    # Get prediction probability
    prediction_proba = loaded_model.predict_proba(reshape_input_data)

    # Extract fraud probability (class 1)
    fraud_probability = float(prediction_proba[0][1])
    
    # Determine the prediction result
    result = "Fraud Transaction." if fraud_probability > 0.5 else "Not Fraud Transaction."
    
    return result, fraud_probability

# Streamlit app title
st.markdown(
    "<h1 style='text-align: center; color: red;'>🚨 Fraud Detection App</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# Define the list of categorical columns for label encoding
categorical_cols = [
    'home_country',
    'source_currency',
    'dest_currency',
    'channel',
    'location_mismatch',
    'kyc_tier',
    'day_of_week'
]

# Initialize the label encoders for categorical features
label_encoders = {col: LabelEncoder() for col in categorical_cols}

# User Inputs in two columns
col1, col2 = st.columns(2)

with col1:
    home_country = st.selectbox('Home Country', ['US', 'CA', 'unknown'])
    source_currency = st.selectbox('Source Currency', ['USD','CAD', 'GBP'])
    dest_currency = st.selectbox('Destination Currency', ['USD', 'CNY', 'CAD', 'GBP', 'NGN', 'INR', 'PHP', 'EUR', 'MXN'])
    channel = st.selectbox('Channel', ['mobile', 'web', 'atm', 'unknown'])
    amount_src = st.number_input('Amount Source', min_value=0.0)
    amount_usd = st.number_input('Amount in USD', min_value=0.0)
    fee = st.number_input('Fee', min_value=0.0)
    exchange_rate_src_to_dest = st.number_input('Exchange Rate Source to Destination', min_value=0.0)
    location_mismatch = st.selectbox('Location Mismatch', ['True', 'False'])

with col2:
    kyc_tier = st.selectbox('KYC Tier', ['standard', 'enhanced', 'low3', 'unknown'])
    account_age_days = st.number_input('Account Age (Days)', min_value=0)
    device_trust_score = st.number_input('Device Trust Score', min_value=0.0)
    risk_score_internal = st.number_input('Risk Score Internal', min_value=0.0)
    txn_velocity_24h = st.number_input('Transaction Velocity (24 Hours)', min_value=0.0)
    corridor_risk = st.number_input('Corridor Risk', min_value=0)
    day_of_week = st.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'unknown'])
    year = st.selectbox('Year', [2025, 2024, 2023, 2022])

if st.button('Fraud Prediction App'):
    try:
        # Prepare the input data for prediction
        input_data = pd.DataFrame({
            'home_country': [home_country],
            'source_currency': [source_currency],
            'dest_currency': [dest_currency],
            'channel': [channel],
            'location_mismatch': [1 if location_mismatch == 'True' else 0],  # Convert to numeric
            'amount_src': [amount_src],
            'amount_usd': [amount_usd],
            'fee': [fee],
            'exchange_rate_src_to_dest': [exchange_rate_src_to_dest],
            'account_age_days': [account_age_days],
            'device_trust_score': [device_trust_score],
            'risk_score_internal': [risk_score_internal],
            'txn_velocity_24h': [txn_velocity_24h],
            'corridor_risk': [corridor_risk],
            'kyc_tier': [kyc_tier],
            'day_of_week': [day_of_week],
            'year': [year]
        })

        # Encode categorical variables
        for col in categorical_cols:
            input_data[col] = label_encoders[col].fit_transform(input_data[col].astype(str))

        result, fraud_probability = predict_fraud(input_data)
        st.success(result)
        st.markdown(f"**Fraud Probability:** {fraud_probability:.2f}")

    except ValueError:
        st.error("Please enter valid numeric values for all fields.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")