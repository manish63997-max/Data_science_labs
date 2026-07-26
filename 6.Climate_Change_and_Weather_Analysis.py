# ============================================================
# EXPERIMENT 6
# Climate Change and Weather Analysis
# Exploratory Data Analysis (EDA)
# Trend Analysis and Pattern Recognition
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. Create Sample Weather Dataset
# ------------------------------------------------------------

np.random.seed(10)

years = np.repeat(range(2015, 2025), 12)
months = list(range(1, 13)) * 10

cities = np.random.choice(
    ["Delhi", "Mumbai", "Bangalore", "Chennai"],
    size=120
)

temperature = np.random.normal(30, 5, 120)
humidity = np.random.normal(65, 10, 120)
rainfall = np.random.normal(120, 40, 120)
wind_speed = np.random.normal(12, 3, 120)
air_quality = np.random.normal(90, 20, 120)

# Introduce some abnormal values (anomalies)

temperature[5] = 50
temperature[70] = 8

rainfall[15] = 350
rainfall[95] = 420

air_quality[30] = 250
air_quality[80] = 280

df = pd.DataFrame({
    "Year": years,
    "Month": months,
    "City": cities,
    "Temperature": temperature,
    "Humidity": humidity,
    "Rainfall": rainfall,
    "Wind_Speed": wind_speed,
    "Air_Quality_Index": air_quality
})

print("=" * 60)
print("CLIMATE CHANGE DATASET")
print("=" * 60)
print(df.head())

# ------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# ------------------------------------------------------------

print("\n================ DATA INFORMATION ================\n")

print(df.info())

print("\n================ MISSING VALUES ================\n")
print(df.isnull().sum())

print("\n================ DESCRIPTIVE STATISTICS ================\n")
print(df.describe())

# ------------------------------------------------------------
# 3. Monthly Statistical Summary
# ------------------------------------------------------------

print("\n================ MONTHLY SUMMARY ================\n")

monthly_summary = df.groupby("Month")[[
    "Temperature",
    "Humidity",
    "Rainfall",
    "Wind_Speed",
    "Air_Quality_Index"
]].mean()

print(monthly_summary)

# ------------------------------------------------------------
# 4. Yearly Statistical Summary
# ------------------------------------------------------------

print("\n================ YEARLY SUMMARY ================\n")

yearly_summary = df.groupby("Year")[[
    "Temperature",
    "Humidity",
    "Rainfall",
    "Wind_Speed",
    "Air_Quality_Index"
]].mean()

print(yearly_summary)

# ------------------------------------------------------------
# 5. Seasonal Trend Detection
# ------------------------------------------------------------

print("\n================ SEASONAL TRENDS ================\n")

plt.figure(figsize=(10,5))

sns.lineplot(
    data=df,
    x="Month",
    y="Temperature",
    estimator="mean",
    marker="o"
)

plt.title("Average Monthly Temperature")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))

sns.lineplot(
    data=df,
    x="Month",
    y="Rainfall",
    estimator="mean",
    marker="o",
    color="green"
)

plt.title("Average Monthly Rainfall")
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 6. Detect Abnormal Weather Events
# ------------------------------------------------------------

print("\n================ ABNORMAL WEATHER EVENTS ================\n")

high_temp = df[df["Temperature"] > 45]

print("Extreme Temperature")
print(high_temp)

heavy_rain = df[df["Rainfall"] > 300]

print("\nHeavy Rainfall")
print(heavy_rain)

poor_air = df[df["Air_Quality_Index"] > 200]

print("\nPoor Air Quality")
print(poor_air)

# ------------------------------------------------------------
# 7. Correlation Analysis
# ------------------------------------------------------------

print("\n================ CORRELATION MATRIX ================\n")

corr = df.select_dtypes(include=np.number).corr()

print(corr)

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Weather Correlation Heatmap")
plt.show()

# ------------------------------------------------------------
# 8. Detect Anomalies using Boxplots
# ------------------------------------------------------------

columns = [
    "Temperature",
    "Humidity",
    "Rainfall",
    "Wind_Speed",
    "Air_Quality_Index"
]

for col in columns:

    plt.figure(figsize=(6,4))

    sns.boxplot(x=df[col])

    plt.title(f"Boxplot of {col}")

    plt.show()

# ------------------------------------------------------------
# 9. Distribution Plots
# ------------------------------------------------------------

for col in columns:

    plt.figure(figsize=(6,4))

    sns.histplot(
        df[col],
        kde=True,
        color="skyblue"
    )

    plt.title(f"{col} Distribution")

    plt.show()

# ------------------------------------------------------------
# 10. City-wise Temperature Trend
# ------------------------------------------------------------

plt.figure(figsize=(10,5))

sns.lineplot(
    data=df,
    x="Year",
    y="Temperature",
    hue="City",
    marker="o"
)

plt.title("Temperature Trend by City")

plt.grid(True)

plt.show()

# ------------------------------------------------------------
# 11. Automated EDA Report
# ------------------------------------------------------------

print("\n================ AUTOMATED EDA REPORT ================\n")

print("Dataset Shape :", df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Records :", df.duplicated().sum())

print("\nData Types")
print(df.dtypes)

print("\nSummary Statistics")
print(df.describe())

# ------------------------------------------------------------
# 12. Analytical Observations
# ------------------------------------------------------------

print("\n================ ANALYTICAL OBSERVATIONS ================\n")

print("1. Monthly averages reveal seasonal weather patterns.")
print("2. Yearly summaries help identify long-term climate trends.")
print("3. Line plots show temperature and rainfall variations.")
print("4. Correlation heatmap identifies relationships among weather variables.")
print("5. Boxplots clearly detect abnormal weather events.")
print("6. Histograms visualize the distribution of climatic parameters.")
print("7. Air Quality Index values above 200 indicate severe pollution.")
print("8. Extremely high temperatures and rainfall are identified as anomalies.")
print("9. City-wise trend analysis compares climate changes across cities.")
print("10. Automated EDA provides a quick overview of the entire dataset.")

# ------------------------------------------------------------
# END OF PROGRAM
# ------------------------------------------------------------