import streamlit as st
import numpy as np
import pickle

# Load the saved model
with open('mpg_model.pkl', 'rb') as file:
    best_rf_model = pickle.load(file)

# Function to predict MPG
def predict_mpg(horsepower, weight, acceleration, displacement, cylinders, model_year, origin):
    input_data = np.array([[horsepower, weight, acceleration, displacement, cylinders, model_year, origin]])
    prediction = best_rf_model.predict(input_data)[0]
    return prediction

# Streamlit UI
st.title("Car MPG Prediction")

# Input fields for the user
horsepower = st.number_input('Horsepower', min_value=1, max_value=1000, value=100)
weight = st.number_input('Weight (lbs)', min_value=500, max_value=5000, value=2500)
acceleration = st.number_input('Acceleration (0-60 in sec)', min_value=1, max_value=30, value=15)
displacement = st.number_input('Displacement (cubic inches)', min_value=50, max_value=500, value=150)
cylinders = st.number_input('Number of Cylinders', min_value=3, max_value=12, value=4)
model_year = st.number_input('Model Year', min_value=70, max_value=80, value=80)
origin = st.selectbox('Origin', options=[1, 2, 3], format_func=lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}[x])

# Button to trigger prediction
if st.button('Predict MPG'):
    mpg = predict_mpg(horsepower, weight, acceleration, displacement, cylinders, model_year, origin)
    st.write(f'Predicted MPG: {mpg:.2f}')
