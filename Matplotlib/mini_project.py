import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load Dataset

data = pd.read_csv("diabetes.csv")   # ensure file is in same folder
print(data.head())
print(data.info())
print(data.describe())


# 2. Data Pre-processing

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Fill missing numerical values with median
for col in data.select_dtypes(include=["float64", "int64"]).columns:
    data[col] = data[col].fillna(data[col].median())

# 3. Handle Categorical Data

# Check categorical columns
cat_cols = data.select_dtypes(include=["object"]).columns
print("Categorical Columns:", cat_cols)

# Convert categorical to numeric (if any)
if len(cat_cols) > 0:
    data = pd.get_dummies(data, drop_first=True)


# 4. Univariate Analysis


# Histogram for numerical features
data.hist(bins=20, figsize=(12,10), color="skyblue")
plt.suptitle("Histograms of Numerical Features")
plt.show()

# Boxplots for numerical features
plt.figure(figsize=(12,8))
sns.boxplot(data=data, palette="Set3")
plt.title("Boxplot for Features")
plt.xticks(rotation=45)
plt.show()

# Countplot for Outcome (if target column exists)
if "Outcome" in data.columns:
    sns.countplot(x="Outcome", data=data, palette="pastel")
    plt.title("Outcome Distribution")
    plt.show()


# 5. Bivariate Analysis


# Pairplot
sns.pairplot(data, diag_kind="kde")
plt.suptitle("Pairplot of Diabetes Data", y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Outcome vs Age (if available)
if "Outcome" in data.columns and "Age" in data.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="Outcome", y="Age", data=data, palette="muted")
    plt.title("Outcome vs Age")
    plt.show()
