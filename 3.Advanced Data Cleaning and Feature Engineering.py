# ============================================================
# EXPERIMENT 3
# Advanced Data Cleaning and Feature Engineering
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# ------------------------------------------------------------
# 1. Create Sample Dataset
# ------------------------------------------------------------

data = {
    "Customer_ID": [101,102,103,104,104,106,107,108,109,110],
    "Name": ["Alice","Bob","Charli","David","David",
             "Eva","Frank","Grace","Hary","Ivy"],
    "Age": [25,200,35,-5,28,45,np.nan,38,29,150],
    "Salary": [50000,np.nan,65000,70000,70000,
               80000,np.nan,120000,40000,500000],
    "Gender": ["Female","Male","Male","Male","Male",
               "Female","Male","Female","Female","Female"],
    "City": ["Delhi","Mumbai","Delhi","Chennai","Chennai",
             "Banglore","Bangalore","Delhi","Mumabi","Mumbai"],
    "Joining_Date": [
        "01-01-2023",
        "2023/02/15",
        "15-Mar-2023",
        "2023-04-10",
        "2023-04-10",
        "05/05/2023",
        "2023.06.01",
        "07-Jul-2023",
        "2023/08/10",
        "10-09-2023"
    ],
    "Currency": [
        "USD","INR","USD","EUR","EUR",
        "USD","INR","USD","USD","EUR"
    ],
    "Monthly_Spending": [
        1500,2000,1800,1700,1700,
        3000,2500,4000,1600,100000
    ]
}

df = pd.DataFrame(data)

print("="*60)
print("ORIGINAL DATASET")
print("="*60)
print(df)

# ------------------------------------------------------------
# 2. Detect Data Quality Issues
# ------------------------------------------------------------

print("\n================ DATA QUALITY ISSUES ================\n")

print("Missing Values")
print(df.isnull().sum())

duplicates = df[df.duplicated(subset="Customer_ID")]
print("\nDuplicate Customer IDs")
print(duplicates)

invalid_age = df[(df["Age"] < 0) | (df["Age"] > 100)]
print("\nIncorrect Age Values")
print(invalid_age[["Customer_ID","Age"]])

print("\nJoining Date Formats")
print(df["Joining_Date"])

print("\nCurrencies Used")
print(df["Currency"].unique())

print("\nCities Before Correction")
print(df["City"].unique())

Q1 = df["Monthly_Spending"].quantile(0.25)
Q3 = df["Monthly_Spending"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["Monthly_Spending"] < lower) |
    (df["Monthly_Spending"] > upper)
]

print("\nOutliers")
print(outliers)

# ------------------------------------------------------------
# 3. Why Data Quality Issues Affect Analytics
# ------------------------------------------------------------

print("\n================ EFFECT ON ANALYTICS ================\n")

print("1. Missing values reduce model accuracy.")
print("2. Duplicate records bias statistical results.")
print("3. Different date formats create parsing errors.")
print("4. Currency mismatch gives incorrect financial analysis.")
print("5. Incorrect ages distort customer segmentation.")
print("6. Outliers affect averages and ML algorithms.")
print("7. Typographical errors create duplicate categories.")

# ------------------------------------------------------------
# 4. Data Cleaning
# ------------------------------------------------------------

clean = df.copy()

clean = clean.drop_duplicates(subset="Customer_ID")

clean["City"] = clean["City"].replace({
    "Banglore": "Bangalore",
    "Mumabi": "Mumbai"
})

clean["Joining_Date"] = pd.to_datetime(
    clean["Joining_Date"],
    errors="coerce",
    dayfirst=True
)

clean.loc[
    (clean["Age"] < 0) | (clean["Age"] > 100),
    "Age"
] = np.nan

rates = {
    "INR": 1,
    "USD": 83,
    "EUR": 90
}

clean["Salary_INR"] = clean.apply(
    lambda x: x["Salary"] * rates[x["Currency"]]
    if pd.notnull(x["Salary"])
    else np.nan,
    axis=1
)

print("\n================ CLEANED DATA ================\n")
print(clean)

# ------------------------------------------------------------
# 5. Missing Value Imputation Comparison
# ------------------------------------------------------------

print("\n================ IMPUTATION ==================\n")

salary_mean = clean["Salary_INR"].fillna(
    clean["Salary_INR"].mean()
)

salary_median = clean["Salary_INR"].fillna(
    clean["Salary_INR"].median()
)

gender_mode = clean["Gender"].fillna(
    clean["Gender"].mode()[0]
)

salary_ffill = clean["Salary_INR"].ffill()

salary_bfill = clean["Salary_INR"].bfill()

print("Mean Imputation")
print(salary_mean)

print("\nMedian Imputation")
print(salary_median)

print("\nMode Imputation (Gender)")
print(gender_mode)

print("\nForward Fill")
print(salary_ffill)

print("\nBackward Fill")
print(salary_bfill)

# Use Median Imputation

clean["Salary_INR"] = salary_median
clean["Age"] = clean["Age"].fillna(clean["Age"].median())

# ------------------------------------------------------------
# 6. Standardization
# ------------------------------------------------------------

scaler = StandardScaler()

clean["Salary_Standardized"] = scaler.fit_transform(
    clean[["Salary_INR"]]
)

# ------------------------------------------------------------
# 7. Normalization
# ------------------------------------------------------------

minmax = MinMaxScaler()

clean["Salary_Normalized"] = minmax.fit_transform(
    clean[["Salary_INR"]]
)

# ------------------------------------------------------------
# 8. Label Encoding
# ------------------------------------------------------------

label = LabelEncoder()

clean["Gender_Label"] = label.fit_transform(clean["Gender"])

# ------------------------------------------------------------
# 9. One-Hot Encoding
# ------------------------------------------------------------

city_encoded = pd.get_dummies(
    clean["City"],
    prefix="City"
)

clean = pd.concat(
    [clean, city_encoded],
    axis=1
)

# ------------------------------------------------------------
# 10. Feature Engineering
# ------------------------------------------------------------

# Annual Income
clean["Annual_Income"] = clean["Salary_INR"] * 12

# Age Group
clean["Age_Group"] = pd.cut(
    clean["Age"],
    bins=[0,18,30,45,60,100],
    labels=[
        "Child",
        "Young",
        "Adult",
        "Senior",
        "Old"
    ]
)

# Spending Category
clean["Spending_Category"] = pd.cut(
    clean["Monthly_Spending"],
    bins=[0,1800,3000,1000000],
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)

# Customer Value Index
clean["Customer_Value_Index"] = (
    clean["Annual_Income"] /
    clean["Monthly_Spending"]
)

# ------------------------------------------------------------
# 11. Dataset Before and After
# ------------------------------------------------------------

print("\n================ BEFORE PREPROCESSING ================\n")
print(df.head())

print("\n================ AFTER PREPROCESSING ================\n")
print(clean.head())

# ------------------------------------------------------------
# 12. Comparison
# ------------------------------------------------------------

print("\n================ COMPARISON ================\n")

comparison = pd.DataFrame({
    "Original Missing": df.isnull().sum(),
    "Processed Missing": clean.isnull().sum()
})

print(comparison)

# ------------------------------------------------------------
# 13. Analytical Observations
# ------------------------------------------------------------

print("\n================ ANALYTICAL OBSERVATIONS ================\n")

print("1. Duplicate customer IDs were removed.")
print("2. Missing salary values were handled using different imputation techniques.")
print("3. Median imputation is more robust against salary outliers than mean.")
print("4. Date formats were standardized into a common datetime format.")
print("5. Currency values were converted into INR for consistent financial analysis.")
print("6. Invalid age values were replaced and imputed.")
print("7. Typographical errors in city names were corrected.")
print("8. Standardization produced zero-mean and unit-variance salary values.")
print("9. Normalization scaled salaries between 0 and 1.")
print("10. Label Encoding converted Gender into numeric values.")
print("11. One-Hot Encoding converted City into binary features.")
print("12. New features (Annual Income, Age Group, Spending Category, Customer Value Index) improve predictive analytics.")
print("13. The cleaned dataset is more reliable for machine learning and business intelligence.")