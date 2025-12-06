import gradio as gr
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the model
try:
    loaded_model = pickle.load(open('nova_model.sav', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")

# Initialize the label encoders for categorical features
categorical_cols = [
    'home_country',
    'source_currency',
    'dest_currency',
    'channel',
    'location_mismatch',
    'kyc_tier',
    'day_of_week'
]
label_encoders = {col: LabelEncoder() for col in categorical_cols}

# Function for Fraud Prediction 
def predict_fraud(home_country, source_currency, dest_currency, channel, location_mismatch, 
                  amount_src, amount_usd, fee, exchange_rate_src_to_dest, 
                  account_age_days, device_trust_score, risk_score_internal, 
                  txn_velocity_24h, corridor_risk, kyc_tier, day_of_week, year):
    
    input_data = pd.DataFrame({
        'home_country': [home_country],
        'source_currency': [source_currency],
        'dest_currency': [dest_currency],
        'channel': [channel],
        'location_mismatch': [1 if location_mismatch == 'True' else 0],
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

    # Get prediction probability
    prediction_proba = loaded_model.predict_proba(input_data)
    fraud_probability = float(prediction_proba[0][1])
    result = "Fraud Transaction." if fraud_probability > 0.5 else "Not Fraud Transaction."

    return result, fraud_probability

# Gradio Interface
iface = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Dropdown(['US', 'CA', 'unknown'], label="Home Country"),
        gr.Dropdown(['USD', 'CAD', 'GBP'], label="Source Currency"),
        gr.Dropdown(['USD', 'CAD', 'EUR'], label="Destination Currency"),
        gr.Dropdown(['mobile', 'web', 'atm', 'unknown'], label="Channel"),
        gr.Radio(['True', 'False'], label="Location Mismatch"),
        gr.Number(label="Amount Source", default=0.0),
        gr.Number(label="Amount in USD", default=0.0),
        gr.Number(label="Fee", default=0.0),
        gr.Number(label="Exchange Rate Source to Destination", default=0.0),
        gr.Number(label="Account Age (Days)", default=0),
        gr.Number(label="Device Trust Score", default=0.0),
        gr.Number(label="Risk Score Internal", default=0.0),
        gr.Number(label="Transaction Velocity (24 Hours)", default=0.0),
        gr.Number(label="Corridor Risk", default=0),
        gr.Dropdown(['Tier 1', 'Tier 2', 'Tier 3'], label="KYC Tier"),
        gr.Dropdown(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'unknown'], label="Day of Week"),
        gr.Number(label="Year", default=2023)
    ],
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Number(label="Fraud Probability")
    ],
    live=True
)

iface.launch()