# Import the pandas library for data manipulation
import pandas as pd

# 1. Load the dataset
# It's good practice to wrap file operations in a try-except block
# to handle potential FileNotFoundError.
try:
    df = pd.read_csv('melb_data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'melb_data.csv' not found.")
    print("Please download the file and place it in the same directory.")
    exit()

# 2. Get a statistical perspective
# The describe() method provides a statistical summary of the numerical columns.
# It includes count, mean, standard deviation, min/max values, and quartiles.
print("\n--- Statistical Summary of the Dataset ---")
print(df.describe())
print("-" * 40)

# The info() method provides a concise summary of the DataFrame,
# including the data types of each column and the number of non-null values.
print("\n--- DataFrame Information (Data Types and Missing Values) ---")
df.info()
print("-" * 40)

# The isnull().sum() method shows the total count of missing values for each column.
# This helps in identifying which columns need cleaning.
print("\n--- Count of Missing Values per Column ---")
print(df.isnull().sum())
print("-" * 40)

# 3. Handle missing data (Imputation and Dropping)
# For numerical columns like 'Car', 'BuildingArea', and 'YearBuilt',
# a common strategy is to impute missing values. Here, we'll use the median.
# For demonstration, we'll drop rows where 'Price' is missing as it's our target variable.

# Create a copy of the DataFrame to work on
df_cleaned = df.copy()

# Drop rows with missing values in the 'Price' column
df_cleaned.dropna(subset=['Price'], inplace=True)
print("\nDropped rows with missing 'Price'. New row count:", len(df_cleaned))

# Impute missing values in 'BuildingArea' with the median
median_building_area = df_cleaned['BuildingArea'].median()
df_cleaned['BuildingArea'].fillna(median_building_area, inplace=True)

# Impute missing values in 'YearBuilt' with the mode (most frequent value)
mode_year_built = df_cleaned['YearBuilt'].mode()[0]
df_cleaned['YearBuilt'].fillna(mode_year_built, inplace=True)

# Impute missing values in 'Car' with the median
median_car = df_cleaned['Car'].median()
df_cleaned['Car'].fillna(median_car, inplace=True)

# Re-check for missing values after cleaning
print("\n--- Missing Values after Cleaning ---")
print(df_cleaned.isnull().sum())
print("-" * 40)

# 4. Handle categorical data (One-Hot Encoding)
# Categorical columns like 'Type', 'Method', and 'Regionname' need to be converted
# into a numerical format for most machine learning algorithms.
# We'll use one-hot encoding for this.

categorical_cols = ['Type', 'Method', 'Regionname']

# Use pd.get_dummies() to perform one-hot encoding
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)

print("\n--- DataFrame after One-Hot Encoding ---")
print("Original column count:", len(df_cleaned.columns))
print("New column count:", len(df_encoded.columns))
print("\nNew columns created from one-hot encoding:")
print(list(df_encoded.columns))
print("-" * 40)

print("Data preprocessing complete. The DataFrame is now ready for analysis or modeling.")


# using sk learn
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

# Load the dataset
try:
    df = pd.read_csv('melb_data.csv')
    print("Dataset loaded successfully.")
    print("-" * 40)
except FileNotFoundError:
    print("Error: 'melb_data.csv' not found.")
    print("Please download the file and place it in the same directory.")
    exit()

# 1. Handling Missing Data using SimpleImputer
# Let's focus on numerical columns with missing values
numerical_cols = ['Car', 'BuildingArea', 'YearBuilt']

# Create a copy to avoid modifying the original DataFrame
df_imputed = df.copy()

# Drop rows where 'Price' is missing since it's the target variable
df_imputed.dropna(subset=['Price'], inplace=True)

# Initialize the SimpleImputer with the median strategy
imputer = SimpleImputer(missing_values=np.nan, strategy='median')

# Apply the imputer to the selected numerical columns
for col in numerical_cols:
    # Reshape the data for the imputer
    imputed_data = imputer.fit_transform(df_imputed[[col]])
    # Update the column in the DataFrame
    df_imputed[col] = imputed_data

# 2. Check the statistical perspective after imputation
print("--- Missing Values After Imputation ---")
print(df_imputed.isnull().sum())
print("-" * 40)

# The statistical summary will now show complete counts for the imputed columns
print("--- Statistical Summary After Imputation ---")
print(df_imputed[['Car', 'BuildingArea', 'YearBuilt']].describe())