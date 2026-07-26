# ============================================================
# EXPERIMENT 5
# Banking Loan Risk Assessment
# Correlation, Covariance, Heatmaps & Feature Relationships
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. Create Sample Banking Dataset
# ------------------------------------------------------------

data = {
    "Customer_ID": range(1, 16),

    "Age": [
        25,30,35,40,28,
        45,50,32,38,60,
        48,36,29,55,42
    ],

    "Annual_Income": [
        300000,450000,500000,650000,350000,
        800000,900000,480000,620000,1000000,
        850000,550000,400000,950000,700000
    ],

    "Loan_Amount": [
        100000,150000,180000,250000,120000,
        300000,350000,170000,240000,400000,
        320000,210000,140000,380000,270000
    ],

    "Credit_Score": [
        650,700,720,750,680,
        780,800,710,740,820,
        790,730,690,810,760
    ],

    "EMI": [
        2500,4000,4500,6000,3000,
        7000,8000,4200,5800,9000,
        7500,5000,3500,8500,6500
    ],

    "Loan_Approved": [
        0,1,1,1,0,
        1,1,1,1,1,
        1,1,0,1,1
    ]
}

df = pd.DataFrame(data)

print("=" * 60)
print("BANKING LOAN DATASET")
print("=" * 60)
print(df)

# ------------------------------------------------------------
# 2. Covariance Matrix
# ------------------------------------------------------------

print("\n================ COVARIANCE MATRIX ================\n")

cov_matrix = df.drop(columns=["Customer_ID"]).cov()

print(cov_matrix)

# ------------------------------------------------------------
# 3. Pearson Correlation Matrix
# ------------------------------------------------------------

print("\n================ PEARSON CORRELATION MATRIX ================\n")

corr_matrix = df.drop(columns=["Customer_ID"]).corr(method="pearson")

print(corr_matrix)

# ------------------------------------------------------------
# 4. Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(8,6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f"
)

plt.title("Pearson Correlation Heatmap")
plt.show()

# ------------------------------------------------------------
# 5. Identify Highly Correlated Attributes
# ------------------------------------------------------------

print("\n================ HIGHLY CORRELATED ATTRIBUTES ================\n")

threshold = 0.80

for i in range(len(corr_matrix.columns)):

    for j in range(i):

        corr = corr_matrix.iloc[i, j]

        if abs(corr) >= threshold:

            print(f"{corr_matrix.columns[i]} <--> {corr_matrix.columns[j]}")
            print(f"Correlation = {corr:.2f}\n")

# ------------------------------------------------------------
# 6. Detect Redundant Features
# ------------------------------------------------------------

print("\n================ REDUNDANT FEATURES ================\n")

redundant = []

for i in range(len(corr_matrix.columns)):

    for j in range(i):

        if abs(corr_matrix.iloc[i, j]) > 0.90:

            redundant.append(
                (
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                )
            )

if redundant:

    for feature1, feature2, corr in redundant:

        print(f"{feature1} and {feature2}")
        print(f"Correlation = {corr:.2f}")
        print("Recommendation : One feature may be removed.\n")

else:

    print("No redundant features found.")

# ------------------------------------------------------------
# 7. Rank Variables Based on Correlation
# ------------------------------------------------------------

print("\n================ VARIABLE RANKING ================\n")

loan_corr = corr_matrix["Loan_Approved"].drop("Loan_Approved")

ranking = loan_corr.abs().sort_values(ascending=False)

print(ranking)

print("\nRanking with Correlation Values")

for feature in ranking.index:

    print(f"{feature:20} {loan_corr[feature]:.3f}")

# ------------------------------------------------------------
# 8. Analytical Report
# ------------------------------------------------------------

print("\n================ ANALYTICAL REPORT ================\n")

for feature in loan_corr.index:

    corr = loan_corr[feature]

    if abs(corr) >= 0.70:
        strength = "Strong"

    elif abs(corr) >= 0.40:
        strength = "Moderate"

    else:
        strength = "Weak"

    direction = "Positive" if corr > 0 else "Negative"

    print(f"{feature}")
    print(f"Correlation : {corr:.3f}")
    print(f"Relationship : {strength} {direction}")
    print()

print("Overall Observations")
print("---------------------")
print("1. Pearson correlation measures linear relationships.")
print("2. Covariance shows how two variables change together.")
print("3. Heatmaps provide an easy visualization of relationships.")
print("4. Highly correlated features indicate strong associations.")
print("5. Redundant features can be removed to improve model performance.")
print("6. Loan approval is influenced by financial attributes like")
print("   Annual Income, Credit Score, Loan Amount and EMI.")
print("7. Correlation does not imply causation.")
print("8. Feature ranking helps in selecting important variables")
print("   for machine learning models.")

# ------------------------------------------------------------
# 9. Optional Pairplot Visualization
# ------------------------------------------------------------

sns.pairplot(
    df.drop(columns=["Customer_ID"]),
    hue="Loan_Approved",
    palette="Set1"
)

plt.show()