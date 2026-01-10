import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle

#load the model
model = tf.keras.models.load_model('model.h5')
#load the encoder and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:  
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:  
    label_encoder_gender = pickle.load(file)  
        
with open('scaler.pkl', 'rb') as file:  
    scaler = pickle.load(file)
    
## steamlit app 
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92) # slider for age between 18 and 92
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})
# One hot encoding for Geography
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

#combine onehot encoded data with input data
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

#scale the input data
input_scaled = scaler.transform(input_data)

#predciton churn
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

st.write(f'Customer churn probability: {prediction_proba:.2f}')
if prediction_proba > 0.5:
    st.write(f"The customer is likely to churn with a probability of {prediction_proba:.2f}")

else:
    st.write(f"The customer is unlikely to churn with a probability of {prediction_proba:.2f}")
