import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset manually (Ensure you have downloaded Boston Housing Dataset)
df = pd.read_csv("D:\\GitHub\\cloudcredits\\Project 1\\boston.csv")  

# Define features and target variable
X = df.drop(columns=['MEDV'])  # MEDV is the target (house price)
y = df['MEDV']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Print evaluation metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plot actual vs predicted prices in BAR GRAPH form
x_labels = range(len(y_test))  # Creating indexes for bar positions

plt.figure(figsize=(10,5))
plt.bar(x_labels, y_test, color="blue", label="Actual Prices", alpha=1.0)
plt.bar(x_labels, y_pred, color="red", label="Predicted Prices", alpha=1.0)

plt.xlabel("House Index")
plt.ylabel("Price")
plt.title("Actual vs Predicted House Prices (Bar Graph)")
plt.legend()
plt.show()
