import numpy as np

# Sample data with potential outliers
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 101]

# Step 1: Calculate Q1 and Q3
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)

# Step 2: Calculate the IQR
IQR = Q3 - Q1

# Step 3: Define the outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 4: Find and list the outliers
outliers = [x for x in data if x < lower_bound or x > upper_bound]

# Print the results
print(f"Data: {data}")
print(f"Q1 (25th percentile): {Q1}")
print(f"Q3 (75th percentile): {Q3}")
print(f"IQR: {IQR}")
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")
print(f"Identified Outliers: {outliers}")