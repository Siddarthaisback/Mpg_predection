# MPG Prediction Model Development

## Importing Libraries and Dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

## 1. Dataset Loading and Exploration

# Define column names as the dataset doesn't have headers
column_names = ["mpg", "cylinders", "displacement", "horsepower", 
                "weight", "acceleration", "model_year", "origin", "car_name"]

# Load the dataset
df = pd.read_csv("auto-mpg.data.csv", delim_whitespace=True, names=column_names)

# Display the first few rows
print("First 5 rows of the dataset:")
df.head()

# Dataset information
print("\nDataset information:")
df.info()

# Statistical summary
print("\nStatistical summary of the dataset:")
df.describe()

## 2. Data Cleaning and Preprocessing

# Check for missing values
print("Missing values in each column:")
df.isnull().sum()

# Replace placeholder values ('?' and 'NA') with NaN
df = df.replace({"?": np.nan, "NA": np.nan})

# Convert columns to numeric type
numeric_columns = ["mpg", "cylinders", "displacement", "horsepower", 
                  "weight", "acceleration", "model_year", "origin"]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Check for missing values after conversion
print("\nMissing values after type conversion:")
df.isnull().sum()

# Fill missing values with mean for numeric columns
df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.mean()))

# Drop rows with missing car names (if any)
df = df.dropna(subset=["car_name"])

# Verify that all missing values are handled
print("\nMissing values after handling:")
df.isnull().sum()

## 3. Exploratory Data Analysis (EDA)

# Distribution of the target variable (MPG)
plt.figure(figsize=(10, 6))
sns.histplot(df['mpg'], kde=True)
plt.title('Distribution of MPG')
plt.xlabel('Miles Per Gallon')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Relationship between features and target
plt.figure(figsize=(15, 10))
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.scatterplot(x=feature, y='mpg', data=df)
    plt.title(f'MPG vs {feature}')
plt.tight_layout()
plt.show()

# Box plot for categorical features
plt.figure(figsize=(12, 6))
sns.boxplot(x='cylinders', y='mpg', data=df)
plt.title('MPG by Number of Cylinders')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='origin', y='mpg', data=df)
plt.title('MPG by Origin')
plt.xticks([0, 1, 2], ['USA', 'Europe', 'Japan'])
plt.show()

# MPG trend over model years
plt.figure(figsize=(12, 6))
year_mpg = df.groupby('model_year')['mpg'].mean().reset_index()
sns.lineplot(x='model_year', y='mpg', data=year_mpg, marker='o')
plt.title('Average MPG by Model Year')
plt.xlabel('Model Year')
plt.ylabel('Average MPG')
plt.grid(True)
plt.show()

## 4. Feature Engineering

# Create a feature for car origin category
df['origin_name'] = df['origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

# Create a feature for decade (70s vs 80s)
df['decade'] = df['model_year'].apply(lambda x: '70s' if x < 80 else '80s')

# Display new features
print("Sample with new features:")
df[['car_name', 'origin', 'origin_name', 'model_year', 'decade']].head()

## 5. Model Preparation

# Define features and target
X = df[["horsepower", "weight", "acceleration", "displacement", "cylinders", "model_year", "origin"]]
y = df['mpg']

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

## 6. Baseline Model: Linear Regression

# Create and train a linear regression model as baseline
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

# Make predictions
y_pred_baseline = baseline_model.predict(X_test)

# Evaluate baseline model
baseline_r2 = r2_score(y_test, y_pred_baseline)
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
baseline_mae = mean_absolute_error(y_test, y_pred_baseline)

print(f"Baseline Linear Regression Results:")
print(f"R² Score: {baseline_r2:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")
print(f"MAE: {baseline_mae:.4f}")

# Feature importance for linear model
baseline_coeffs = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': baseline_model.coef_
})
baseline_coeffs = baseline_coeffs.sort_values('Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=baseline_coeffs)
plt.title('Linear Regression Feature Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.grid(True, axis='x')
plt.show()

## 7. Advanced Model: Random Forest Regressor

# Create and train a random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate RF model
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae = mean_absolute_error(y_test, y_pred_rf)

print(f"Random Forest Results:")
print(f"R² Score: {rf_r2:.4f}")
print(f"RMSE: {rf_rmse:.4f}")
print(f"MAE: {rf_mae:.4f}")

# Feature importance for Random Forest
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.grid(True, axis='x')
plt.show()

## 8. Hyperparameter Tuning

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='r2'
)

grid_search.fit(X_train, y_train)

# Best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Create model with best parameters
best_rf_model = grid_search.best_estimator_

# Make predictions with tuned model
y_pred_best = best_rf_model.predict(X_test)

# Evaluate tuned model
best_r2 = r2_score(y_test, y_pred_best)
best_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
best_mae = mean_absolute_error(y_test, y_pred_best)

print(f"Tuned Random Forest Results:")
print(f"R² Score: {best_r2:.4f}")
print(f"RMSE: {best_rmse:.4f}")
print(f"MAE: {best_mae:.4f}")

## 9. Model Comparison

# Create a DataFrame to compare model performance
models = ['Linear Regression', 'Random Forest', 'Tuned Random Forest']
r2_scores = [baseline_r2, rf_r2, best_r2]
rmse_scores = [baseline_rmse, rf_rmse, best_rmse]
mae_scores = [baseline_mae, rf_mae, best_mae]

comparison_df = pd.DataFrame({
    'Model': models,
    'R² Score': r2_scores,
    'RMSE': rmse_scores,
    'MAE': mae_scores
})

print(comparison_df)

# Visualize model comparison
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='R² Score', data=comparison_df)
plt.title('Model Comparison - R² Score')
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='RMSE', data=comparison_df)
plt.title('Model Comparison - RMSE')
plt.grid(True, axis='y')
plt.show()

## 10. Analysis of Predictions

# Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual vs Predicted MPG Values')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.grid(True)
plt.show()

# Residual plot
residuals = y_test - y_pred_best
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_best, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred_best.min(), xmax=y_pred_best.max(), colors='r', linestyles='--')
plt.title('Residual Plot')
plt.xlabel('Predicted MPG')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Distribution of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

## 11. Save the Final Model

# Save the best model to a file
with open('mpg_model.pkl', 'wb') as file:
    pickle.dump(best_rf_model, file)

print("Model saved as 'mpg_model.pkl'")

# Save model metrics for the Streamlit app
with open('model_metrics.txt', 'w') as file:
    file.write(str(best_r2))

print("Model metrics saved")

## 12. Sample Prediction Function

def predict_mpg(horsepower, weight, acceleration, displacement, cylinders, model_year, origin):
    """Function to predict MPG given car specifications"""
    input_data = np.array([[horsepower, weight, acceleration, displacement, cylinders, model_year, origin]])
    prediction = best_rf_model.predict(input_data)[0]
    return prediction

# Example prediction
example_car = {
    'horsepower': 100,
    'weight': 2500,
    'acceleration': 15,
    'displacement': 150,
    'cylinders': 4,
    'model_year': 80,
    'origin': 3  # Japan
}

predicted_mpg = predict_mpg(**example_car)
print(f"Predicted MPG for example car: {predicted_mpg:.2f}")

## 13. Summary and Conclusions

"""
Summary of the Car MPG Prediction Project:

1. Data Analysis:
   - Dataset contained information about various car features and their fuel efficiency
   - Strong correlation found between MPG and features like weight, horsepower, and displacement
   - Cars from different origins showed distinct MPG patterns
   - Clear trend of increasing MPG over the model years (1970s to 1980s)

2. Model Development:
   - Baseline Linear Regression achieved R² score of approximately {baseline_r2:.4f}
   - Random Forest model improved performance to R² score of {rf_r2:.4f}
   - Hyperparameter tuning further improved the model to R² score of {best_r2:.4f}
   
3. Key Findings:
   - Weight and horsepower are the most important features for predicting MPG
   - The model can predict car fuel efficiency with high accuracy (approximately {best_r2*100:.1f}% of variance explained)
   - The prediction error (RMSE) is approximately {best_rmse:.2f} MPG

4. Next Steps:
   - Deploy the model with a user-friendly interface using Streamlit
   - Potentially collect more recent car data to make the model relevant for modern vehicles
   - Explore additional features that might impact fuel efficiency
"""

print("Analysis complete!")