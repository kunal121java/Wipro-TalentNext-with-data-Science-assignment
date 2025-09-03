# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------
# 1. Cancer Dataset
# -----------------------
print("=== Cancer Dataset ===")

# Load dataset
cancer = pd.read_csv("cancer.csv")

# Split features & target
X = cancer.drop("target", axis=1)   # change column name if different
y = cancer["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# -----------------------
# 2. Social Network Ads Dataset
# -----------------------
print("\n=== Social Network Ads Dataset ===")

# Load dataset
ads = pd.read_csv("Social_Network_Ads.csv")

# Split features & target
X = ads.drop("Purchased", axis=1)   # adjust column names
y = ads["Purchased"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train, y_train)

# Predictions
y_pred2 = model2.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred2))
print("Classification Report:\n", classification_report(y_test, y_pred2))
