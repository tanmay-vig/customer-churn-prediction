import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Set page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Apply dark theme and set CSS for custom color styles
st.markdown("""
    <style>
    body {
        color: white;
        background-color: #0e1117;
    }
    .title {
        color: #ff4b4b; /* Red for title */
        font-size: 2em;
    }
    .subtitle {
        color: #ffb3b3;
        font-size: 1.2em;
    }
    .result-churn {
        color: #4CAF50; /* Green for churn likely */
        font-weight: bold;
        font-size: 1.5em;
    }
    .result-no-churn {
        color: #ff4b4b; /* Red for no churn likely */
        font-weight: bold;
        font-size: 1.5em;
    }
    .error {
        color: #ff4b4b;
        font-size: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and encoders with error handling
try:
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except (FileNotFoundError, IOError):
    st.error("Error: Missing model or scaler files. Please check if all files are available.")
    st.stop()

# Title and description
st.markdown("<h1 class='title'>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter customer details below to predict the likelihood of churn.</p>", unsafe_allow_html=True)

try:
    # Input fields
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance', min_value=0.0, format="%.2f")
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, format="%.2f")
    tenure = st.slider('Tenure', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

    # Process input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography' and scale input
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    # Predict churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # Display result with green for likely churn and red for no churn
    st.markdown("<h2 class='subtitle'>Prediction Result</h2>", unsafe_allow_html=True)
    if prediction_proba > 0.5:
        st.markdown("<p class='result-churn'>The customer is likely to churn.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='result-no-churn'>The customer is not likely to churn.</p>", unsafe_allow_html=True)
    st.write(f"**Churn Probability:** {prediction_proba:.2f}")

except Exception as e:
    st.markdown("<p class='error'>An error occurred. Please check your input and try again.</p>", unsafe_allow_html=True)
