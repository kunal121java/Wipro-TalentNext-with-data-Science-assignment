# Sales Prediction using Multiple Linear Regression

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset
# Make sure Advertising.csv is in your working directory
data = pd.read_csv("Advertising.csv")

# Display first 5 rows
print("Dataset Preview:")
print(data.head())

# Step 3: Data Preprocessing
print("\nChecking for null values:")
print(data.isnull().sum())

# No categorical data in Advertising.csv (only numerical)
# If categorical, we would use pd.get_dummies()

# Step 4: Exploratory Data Analysis (EDA)
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', height=5, aspect=0.8)
plt.show()

sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()

# Step 5: Define Features (X) and Target (y)
X = data[['TV','Radio','Newspaper']]
y = data['Sales']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Build Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluation
print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 10: Coefficients
print("\nModel Coefficients:")
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Step 11: Visualization (Actual vs Predicted)
plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
