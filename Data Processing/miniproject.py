# DATA PREPROCESSING FOR HOUSE PRICE PREDICTION
import pandas as pd

try:
    # 1. LOAD THE DATA
    melb_df = pd.read_csv('melb_data.csv')
    print("Initial Data Shape:", melb_df.shape)
    
    # 2. HANDLE INAPPROPRIATE DATA
    # Drop irrelevant columns
    columns_to_drop = ['Address', 'Postcode', 'Lattitude', 'Longtitude', 'CouncilArea', 'Suburb', 'BuildingArea', 'YearBuilt', 'Regionname']
    melb_df.drop(columns_to_drop, axis=1, inplace=True)
    print("Shape after dropping columns:", melb_df.shape)
    
    # 3. HANDLE MISSING DATA
    # Drop rows with missing values in key columns
    melb_df.dropna(subset=['Price', 'Landsize'], inplace=True)
    # Impute missing values for 'Car' with the median
    median_car = melb_df['Car'].median()
    melb_df['Car'].fillna(median_car, inplace=True)
    print("\nMissing values remaining after handling:\n", melb_df.isnull().sum())
    
    # 4. HANDLE CATEGORICAL DATA
    # One-hot encode categorical columns
    categorical_cols = ['Type', 'Method', 'SellerG']
    melb_df = pd.get_dummies(melb_df, columns=categorical_cols, drop_first=True)
    
    print("\nFinal DataFrame shape after one-hot encoding:", melb_df.shape)
    print("Final Columns:\n", melb_df.columns)
    
except FileNotFoundError:
    print("Error: 'melb_data.csv' not found.")