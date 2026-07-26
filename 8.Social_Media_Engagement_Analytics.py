# ============================================================
# EXPERIMENT 8
# Social Media Engagement Analytics
# Dashboard & Data Visualization
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. Create Sample Social Media Dataset
# ------------------------------------------------------------

np.random.seed(100)

platforms = [
    "Instagram",
    "Facebook",
    "Twitter",
    "LinkedIn",
    "YouTube"
]

months = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"
]

records = []

for month in months:

    for platform in platforms:

        records.append({

            "Month": month,

            "Platform": platform,

            "Posts": np.random.randint(20, 120),

            "Likes": np.random.randint(1000, 20000),

            "Comments": np.random.randint(100, 3000),

            "Shares": np.random.randint(50, 2000),

            "Followers_Gained": np.random.randint(100, 5000)

        })

df = pd.DataFrame(records)

print("=" * 60)
print("SOCIAL MEDIA DATASET")
print("=" * 60)

print(df.head())

# ------------------------------------------------------------
# 2. Dataset Overview
# ------------------------------------------------------------

print("\n================ DATA INFORMATION ================\n")

print(df.info())

print("\n================ DESCRIPTIVE STATISTICS ================\n")

print(df.describe())

print("\n================ MISSING VALUES ================\n")

print(df.isnull().sum())

# ------------------------------------------------------------
# 3. Platform-wise Summary
# ------------------------------------------------------------

print("\n================ PLATFORM SUMMARY ================\n")

platform_summary = df.groupby("Platform")[[
    "Posts",
    "Likes",
    "Comments",
    "Shares",
    "Followers_Gained"
]].sum()

print(platform_summary)

# ------------------------------------------------------------
# 4. Monthly Engagement Summary
# ------------------------------------------------------------

print("\n================ MONTHLY SUMMARY ================\n")

monthly_summary = df.groupby("Month")[[
    "Likes",
    "Comments",
    "Shares",
    "Followers_Gained"
]].sum()

print(monthly_summary)

# ------------------------------------------------------------
# 5. Monthly Likes Trend
# ------------------------------------------------------------

plt.figure(figsize=(10,5))

monthly_summary["Likes"].plot(
    kind="line",
    marker="o"
)

plt.title("Monthly Likes Trend")

plt.xlabel("Month")

plt.ylabel("Likes")

plt.grid(True)

plt.show()

# ------------------------------------------------------------
# 6. Followers Gained by Platform
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

platform_summary["Followers_Gained"].plot(
    kind="bar",
    color="purple"
)

plt.title("Followers Gained by Platform")

plt.xlabel("Platform")

plt.ylabel("Followers Gained")

plt.grid(axis="y")

plt.show()

# ------------------------------------------------------------
# 7. Likes vs Comments
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="Likes",
    y="Comments",
    hue="Platform",
    s=100
)

plt.title("Likes vs Comments")

plt.grid(True)

plt.show()

# ------------------------------------------------------------
# 8. Shares Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(
    df["Shares"],
    bins=10,
    kde=True,
    color="orange"
)

plt.title("Distribution of Shares")

plt.xlabel("Shares")

plt.ylabel("Frequency")

plt.show()

# ------------------------------------------------------------
# 9. Correlation Heatmap
# ------------------------------------------------------------

numeric_data = df[[
    "Posts",
    "Likes",
    "Comments",
    "Shares",
    "Followers_Gained"
]]

corr_matrix = numeric_data.corr()

plt.figure(figsize=(7,6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

# ------------------------------------------------------------
# 10. Engagement Score
# ------------------------------------------------------------

df["Engagement_Score"] = (
    df["Likes"] +
    df["Comments"] +
    df["Shares"]
)

print("\n================ ENGAGEMENT SCORE =================\n")

print(df[[
    "Platform",
    "Month",
    "Engagement_Score"
]].head())

# ------------------------------------------------------------
# 11. Top Performing Platforms
# ------------------------------------------------------------

top_platforms = df.groupby("Platform")[
    "Engagement_Score"
].sum().sort_values(ascending=False)

print("\n================ TOP PERFORMING PLATFORMS =================\n")

print(top_platforms)

plt.figure(figsize=(8,5))

top_platforms.plot(
    kind="bar",
    color="teal"
)

plt.title("Total Engagement by Platform")

plt.xlabel("Platform")

plt.ylabel("Engagement Score")

plt.grid(axis="y")

plt.show()

# ------------------------------------------------------------
# 12. Dashboard Summary
# ------------------------------------------------------------

print("\n================ DASHBOARD SUMMARY =================\n")

print("Total Posts :", df["Posts"].sum())
print("Total Likes :", df["Likes"].sum())
print("Total Comments :", df["Comments"].sum())
print("Total Shares :", df["Shares"].sum())
print("Total Followers Gained :", df["Followers_Gained"].sum())

print("\nBest Platform :",
      top_platforms.idxmax())

print("Highest Engagement Score :",
      top_platforms.max())

# ------------------------------------------------------------
# 13. Analytical Observations
# ------------------------------------------------------------

print("\n================ ANALYTICAL OBSERVATIONS =================\n")

print("1. Platform-wise analysis identifies the most popular social media platforms.")
print("2. Monthly trends help track audience engagement over time.")
print("3. Scatter plots show the relationship between likes and comments.")
print("4. Histograms reveal the distribution of shares.")
print("5. Correlation heatmaps identify relationships among engagement metrics.")
print("6. Engagement Score combines likes, comments and shares into one metric.")
print("7. Dashboard summaries provide a quick overview of social media performance.")
print("8. Top-performing platforms can be prioritized for future campaigns.")
print("9. Visualizations simplify social media analytics and reporting.")
print("10. Data-driven insights support better marketing decisions.")

# ------------------------------------------------------------
# END OF PROGRAM
# ------------------------------------------------------------