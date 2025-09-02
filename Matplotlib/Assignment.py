# First Question 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
mall = pd.read_csv("Mall_Customers.csv")

# Gender distribution
sns.countplot(x="Gender", data=mall, palette="pastel")
plt.title("Gender Distribution")
plt.show()

# Age distribution
sns.histplot(mall["Age"], bins=20, kde=True, color="skyblue")
plt.title("Age Distribution")
plt.show()

# Annual Income vs Spending Score
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", hue="Gender", data=mall, palette="coolwarm")
plt.title("Income vs Spending Score")
plt.show()

# Correlation heatmap
sns.heatmap(mall.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
# --------------------------------------------------------------------------------------------------

# Load dataset
salary = pd.read_csv("Salary_Data.csv")

# Gender distribution
sns.countplot(x="Gender", data=salary, palette="Set2")
plt.title("Gender Distribution")
plt.show()

# Salary distribution
sns.histplot(salary["Salary"], bins=30, kde=True, color="purple")
plt.title("Salary Distribution")
plt.show()

# Salary vs Years of Experience
sns.scatterplot(x="Years of Experience", y="Salary", hue="Gender", data=salary)
plt.title("Salary vs Experience")
plt.show()

# Education level vs Salary (Boxplot)
plt.figure(figsize=(8,5))
sns.boxplot(x="Education Level", y="Salary", data=salary, palette="muted")
plt.xticks(rotation=45)
plt.title("Education Level vs Salary")
plt.show()
# --------------------------------------------------------------------
# Load dataset
social = pd.read_csv("Social_Network_Ads.csv")

# Age distribution
sns.histplot(social["Age"], bins=20, kde=True, color="green")
plt.title("Age Distribution")
plt.show()

# Salary distribution
sns.histplot(social["EstimatedSalary"], bins=20, kde=True, color="orange")
plt.title("Estimated Salary Distribution")
plt.show()

# Purchased vs Age
sns.boxplot(x="Purchased", y="Age", data=social, palette="coolwarm")
plt.title("Purchase Decision vs Age")
plt.show()

# Age vs Salary with Purchased hue
sns.scatterplot(x="Age", y="EstimatedSalary", hue="Purchased", data=social, palette="Set1")
plt.title("Age vs Estimated Salary (Purchased Highlight)")
plt.show()
