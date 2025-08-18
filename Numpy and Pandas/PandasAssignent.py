import pandas as pd

# Exercise 3: Pandas-DataFrame
# Download the dataset from the link provided and rename it to cars.csv
# Link: https://www.kaggle.com/uciml/autompg-dataset/data?select=auto-mpg.csv

# a. Import Pandas (already done above)

# b. Import the Cars Dataset and store the Pandas DataFrame in the variable cars
try:
    cars = pd.read_csv('cars.csv')
except FileNotFoundError:
    print("Error: 'cars.csv' not found. Please download the file from the provided link and place it in the same directory.")
    exit()

# c. Inspect the first 10 Rows of the DataFrame
print("First 10 Rows of the DataFrame:")
print(cars.head(10))
print("-" * 40)

# d. Inspect the DataFrame cars by "printing" cars
# This prints the first and last few rows, providing a quick overview
print("Printing the entire DataFrame (first/last few rows):")
print(cars)
print("-" * 40)

# e. Inspect the last 5 Rows
print("Last 5 Rows of the DataFrame:")
print(cars.tail())
print("-" * 40)

# f. Get some meta information on our DataFrame
print("Meta Information on the DataFrame:")
cars.info()

import pandas as pd

# Exercise 4: Download 50_startups dataset
# Link: https://www.kaggle.com/datasets/farhanmd29/50-startups

# a. Create DataFrame using Pandas
# b. Read the data from 50_startups.csv file and load the data into dataframe.
try:
    startups = pd.read_csv('50_startups.csv')
except FileNotFoundError:
    print("Error: '50_startups.csv' not found. Please download the file and place it in the same directory.")
    exit()

print("50 Startups DataFrame loaded successfully.")
print("-" * 40)

# c. Check the statistical summary
# This provides descriptive statistics for numerical columns
print("Statistical Summary of the DataFrame:")
print(startups.describe())
print("-" * 40)

# d. Check for correlation coefficient between dependent and independent variables
# The dependent variable is typically 'Profit' in this dataset.
# The independent variables are 'R&D Spend', 'Administration', and 'Marketing Spend'.
print("Correlation Matrix:")
correlation_matrix = startups.corr(numeric_only=True)
print(correlation_matrix)
print("-" * 40)

# Optional: Print the correlation of each variable with 'Profit'
print("Correlation with 'Profit' column:")
print(correlation_matrix['Profit'].sort_values(ascending=False))