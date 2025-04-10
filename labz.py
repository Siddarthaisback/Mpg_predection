import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import streamlit as st


column_names = ["mpg", "cylinders", "displacement", "horsepower", 
                "weight", "acceleration", "model_year", "origin", "car_name"]
df = pd.read_csv("/Users/gurushrestha/Desktop/test/silly_goose/auto-mpg.data.csv",delim_whitespace=True,names=column_names)

df = df.replace({"?": np.nan, "NA": np.nan})

numeric_data = ["mpg", "cylinders", "displacement", "horsepower", 
                "weight", "acceleration", "model_year", "origin"]
for col in numeric_data:
    df[col] = pd.to_numeric(df[col],errors="coerce") #converting the values into numeric values and errors="coerce" le errors lai nan ma lagxa
df[numeric_data] = df[numeric_data].apply(lambda x : x.fillna(x.mean()))

df = df.dropna(subset=["car_name"])

print(df.isna().sum())

X = df[["horsepower", "weight", "acceleration", "displacement", "cylinders", "model_year", "origin"]]
y = df['mpg']

print(y.shape)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate R² score and store in a variable
r2_value = r2_score(y_test, y_pred)
print(r2_value)

# Streamlit app
st.title("🚗 Car MPG Prediction App")

st.write("Enter car details below to predict the fuel efficiency (MPG).")

# User input
cylinders = st.number_input("Cylinders", min_value=3, max_value=12, value=4)
displacement = st.number_input("Displacement", min_value=50.0, max_value=500.0, value=150.0)
horsepower = st.number_input("Horsepower", min_value=30.0, max_value=300.0, value=100.0)
weight = st.number_input("Weight (lbs)", min_value=1000.0, max_value=6000.0, value=2500.0)
acceleration = st.number_input("Acceleration", min_value=5.0, max_value=25.0, value=10.0)
model_year = st.number_input("Model Year", min_value=70, max_value=82, value=76)
origin = st.selectbox("Origin", options=[1, 2, 3], format_func=lambda x: {1: "USA", 2: "Europe", 3: "Japan"}[x])

# Prediction
if st.button("Predict MPG"):
    # Make sure input data matches the order of features used in training
    input_data = np.array([[horsepower, weight, acceleration, displacement, cylinders, model_year, origin]])
    mpg_prediction = model.predict(input_data)[0]
    st.success(f"Predicted MPG: {mpg_prediction:.2f}")

# Show model R2 score
st.markdown("---")
st.write(f"Model R² score on test data: **{r2_value:.4f}**")