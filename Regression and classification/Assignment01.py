

# Import required libraries (common for all assignments)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------
# Assignment 1: Car Price Prediction (cars.csv)
# ---------------------------------------------------------------
"""
Question 1:
Predict the price of the car based on its features.
Use appropriate evaluation metrics.
Dataset: cars.csv
Topics Covered: Multiple Linear Regression
"""

print("\n===== Assignment 1: Car Price Prediction =====")

# Load dataset
cars = pd.read_csv("cars.csv")
print("Sample data:\n", cars.head())

# Features and Target (assuming 'Price' is target column)
X = cars.drop("Price", axis=1)
y = cars["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
car_model = LinearRegression()
car_model.fit(X_train, y_train)

# Predictions
y_pred_car = car_model.predict(X_test)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred_car))
print("R2 Score:", r2_score(y_test, y_pred_car))
print("Car price prediction completed successfully.")

# ---------------------------------------------------------------
# Assignment 2: Startup Profit Prediction (50_Startups.csv)
# ---------------------------------------------------------------
"""
Question 2:
Create a model that can predict the profit based on its features.
Use appropriate evaluation metrics.
Dataset: 50_Startups.csv
Topics Covered: Multiple Linear Regression
"""

print("\n===== Assignment 2: Startup Profit Prediction =====")

# Load dataset
startups = pd.read_csv("50_Startups.csv")
print("Sample data:\n", startups.head())

# Handle categorical data (State column)
startups = pd.get_dummies(startups, drop_first=True)

# Features and Target
X = startups.drop("Profit", axis=1)
y = startups["Profit"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
startup_model = LinearRegression()
startup_model.fit(X_train, y_train)

# Predictions
y_pred_startup = startup_model.predict(X_test)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred_startup))
print("R2 Score:", r2_score(y_test, y_pred_startup))
print("Startup profit prediction completed successfully.")

# ---------------------------------------------------------------
# Assignment 3: Salary Prediction (Salary_Data.csv)
# ---------------------------------------------------------------
"""
Question 3:
Create a model that can predict the salary based on years of experience.
Use appropriate evaluation metrics.
Dataset: Salary_Data.csv
Topics Covered: Simple Linear Regression
"""

print("\n===== Assignment 3: Salary Prediction =====")

# Load dataset
salary = pd.read_csv("Salary_Data.csv")
print("Sample data:\n", salary.head())

# Features and Target
X = salary[["YearsExperience"]]
y = salary["Salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
salary_model = LinearRegression()
salary_model.fit(X_train, y_train)

# Predictions
y_pred_salary = salary_model.predict(X_test)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred_salary))
print("R2 Score:", r2_score(y_test, y_pred_salary))

# Visualization
plt.scatter(X, y, color="blue")
plt.plot(X, salary_model.predict(X), color="red")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction using Linear Regression")
plt.show()

print("Salary prediction completed successfully.")


