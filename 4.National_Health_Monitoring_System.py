# ============================================================
# EXPERIMENT 4
# National Health Monitoring System
# Descriptive Statistics and Outlier Detection
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# ------------------------------------------------------------
# 1. Create Sample Patient Dataset
# ------------------------------------------------------------

data = {
    "Patient_ID": range(1, 16),
    "Hospital": [
        "Apollo", "Fortis", "AIIMS", "Apollo", "Fortis",
        "AIIMS", "Apollo", "Fortis", "AIIMS", "Apollo",
        "Fortis", "AIIMS", "Apollo", "Fortis", "AIIMS"
    ],
    "Age": [
        25, 45, 60, 35, 50,
        42, 38, 55, 65, 70,
        48, 33, 28, 75, 90
    ],
    "Blood_Pressure": [
        120, 135, 180, 128, 140,
        130, 125, 145, 190, 135,
        138, 126, 124, 200, 250
    ],
    "Sugar_Level": [
        90, 110, 180, 95, 130,
        120, 100, 140, 250, 115,
        118, 105, 98, 260, 400
    ],
    "Heart_Rate": [
        72, 80, 95, 75, 82,
        78, 74, 85, 100, 76,
        79, 73, 71, 105, 140
    ]
}

df = pd.DataFrame(data)

print("=" * 60)
print("PATIENT HEALTH DATA")
print("=" * 60)
print(df)

# ------------------------------------------------------------
# 2. Descriptive Statistics
# ------------------------------------------------------------

print("\n================ DESCRIPTIVE STATISTICS ================\n")
print(df.describe())

# ------------------------------------------------------------
# 3. Hospital-wise Patient Distribution
# ------------------------------------------------------------

print("\n================ PATIENT DISTRIBUTION ================\n")

hospital_count = df["Hospital"].value_counts()

print(hospital_count)

mean_patients = hospital_count.mean()

print("\nHospitals with Abnormal Patient Distribution")

for hospital, count in hospital_count.items():

    if count > mean_patients:
        print(f"{hospital} : Higher than average")

    elif count < mean_patients:
        print(f"{hospital} : Lower than average")

    else:
        print(f"{hospital} : Average")

# ------------------------------------------------------------
# 4. Outlier Detection using IQR
# ------------------------------------------------------------

print("\n================ IQR OUTLIER DETECTION ================\n")

medical_columns = [
    "Age",
    "Blood_Pressure",
    "Sugar_Level",
    "Heart_Rate"
]

for col in medical_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    print(f"\n{col} Outliers (IQR Method)")
    print(outliers[[col]])

# ------------------------------------------------------------
# 5. Outlier Detection using Z-Score
# ------------------------------------------------------------

print("\n================ Z-SCORE OUTLIER DETECTION ================\n")

for col in medical_columns:

    z_scores = np.abs(zscore(df[col]))

    outliers = df[z_scores > 3]

    print(f"\n{col} Outliers (Z-Score Method)")
    print(outliers[[col]])

# ------------------------------------------------------------
# 6. Mean vs Median Comparison
# ------------------------------------------------------------

print("\n================ MEAN vs MEDIAN ================\n")

for col in medical_columns:

    mean = df[col].mean()
    median = df[col].median()

    print(f"{col}")
    print("Mean :", round(mean, 2))
    print("Median :", median)

    if mean > median:
        print("Interpretation : Positively Skewed\n")

    elif mean < median:
        print("Interpretation : Negatively Skewed\n")

    else:
        print("Interpretation : Symmetrical\n")

# ------------------------------------------------------------
# 7. Statistical Summary Report
# ------------------------------------------------------------

print("\n================ STATISTICAL SUMMARY REPORT ================\n")

summary = pd.DataFrame({

    "Mean": df[medical_columns].mean(),
    "Median": df[medical_columns].median(),
    "Mode": df[medical_columns].mode().iloc[0],
    "Minimum": df[medical_columns].min(),
    "Maximum": df[medical_columns].max(),
    "Variance": df[medical_columns].var(),
    "Standard Deviation": df[medical_columns].std()

})

print(summary)

# ------------------------------------------------------------
# 8. Visualization
# ------------------------------------------------------------

sns.set(style="whitegrid")

# Histogram

for col in medical_columns:

    plt.figure(figsize=(6,4))

    sns.histplot(
        df[col],
        bins=8,
        kde=True,
        color="skyblue"
    )

    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Box Plot

for col in medical_columns:

    plt.figure(figsize=(6,4))

    sns.boxplot(
        x=df[col],
        color="orange"
    )

    plt.title(f"{col} Box Plot")
    plt.show()

# Hospital-wise Average Blood Pressure

plt.figure(figsize=(6,4))

sns.barplot(
    x="Hospital",
    y="Blood_Pressure",
    data=df,
    estimator=np.mean
)

plt.title("Average Blood Pressure by Hospital")
plt.show()

# ------------------------------------------------------------
# 9. Analytical Observations
# ------------------------------------------------------------

print("\n================ ANALYTICAL OBSERVATIONS ================\n")

print("1. Descriptive statistics summarize patient health indicators.")
print("2. Mean and median comparison helps identify skewed medical data.")
print("3. IQR method detects moderate outliers.")
print("4. Z-score method detects extreme outliers.")
print("5. Box plots clearly visualize abnormal patient records.")
print("6. Histograms show distribution of medical parameters.")
print("7. Hospital-wise analysis helps identify unusual patient distributions.")
print("8. Statistical summary supports healthcare decision making.")
print("9. Outlier detection can identify patients requiring immediate medical attention.")
print("10. This analysis helps government healthcare departments monitor disease trends.")