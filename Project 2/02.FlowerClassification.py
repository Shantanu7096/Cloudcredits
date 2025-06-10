import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Evaluate models
dt_accuracy = accuracy_score(y_test, dt_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)

print(f"Decision Tree Accuracy: {dt_accuracy:.2f}")
print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}")

# Confusion Matrix for Decision Tree
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, dt_pred), annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion Matrix for Logistic Regression
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, fmt="d", cmap="Greens", xticklabels=target_names, yticklabels=target_names)
plt.title("Logistic Regression - Confusion ṇMatrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred, target_names=target_names))
print("Logistic Regression Classification Report:\n", classification_report(y_test, lr_pred, target_names=target_names))