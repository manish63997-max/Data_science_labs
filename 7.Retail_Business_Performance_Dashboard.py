# ============================================================
# EXPERIMENT 7
# Retail Business Performance Dashboard
# Sales Analysis using Pandas & Matplotlib
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. Create Sample Retail Dataset
# ------------------------------------------------------------

np.random.seed(42)

products = [
    "Laptop", "Mobile", "Tablet", "Headphones",
    "Keyboard", "Mouse", "Monitor", "Printer"
]

regions = [
    "North",
    "South",
    "East",
    "West"
]

months = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"
]

records = []

for month in months:

    for product in products:

        records.append({

            "Month": month,

            "Product": product,

            "Region": np.random.choice(regions),

            "Units_Sold": np.random.randint(50, 500),

            "Revenue": np.random.randint(5000, 50000),

            "Profit": np.random.randint(1000, 15000)

        })

df = pd.DataFrame(records)

print("=" * 60)
print("RETAIL BUSINESS DATASET")
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
# 3. Monthly Sales Summary
# ------------------------------------------------------------

print("\n================ MONTHLY SALES SUMMARY ================\n")

monthly_sales = df.groupby("Month")[[
    "Units_Sold",
    "Revenue",
    "Profit"
]].sum()

print(monthly_sales)

# ------------------------------------------------------------
# 4. Product-wise Sales Summary
# ------------------------------------------------------------

print("\n================ PRODUCT SALES SUMMARY ================\n")

product_sales = df.groupby("Product")[[
    "Units_Sold",
    "Revenue",
    "Profit"
]].sum()

print(product_sales)

# ------------------------------------------------------------
# 5. Region-wise Sales Summary
# ------------------------------------------------------------

print("\n================ REGION SALES SUMMARY ================\n")

region_sales = df.groupby("Region")[[
    "Revenue",
    "Profit"
]].sum()

print(region_sales)

# ------------------------------------------------------------
# 6. Monthly Revenue Trend
# ------------------------------------------------------------

plt.figure(figsize=(10,5))

monthly_sales["Revenue"].plot(
    kind="line",
    marker="o"
)

plt.title("Monthly Revenue Trend")

plt.xlabel("Month")

plt.ylabel("Revenue")

plt.grid(True)

plt.show()

# ------------------------------------------------------------
# 7. Product Revenue Comparison
# ------------------------------------------------------------

plt.figure(figsize=(10,5))

product_sales["Revenue"].sort_values().plot(
    kind="barh",
    color="skyblue"
)

plt.title("Revenue by Product")

plt.xlabel("Revenue")

plt.show()

# ------------------------------------------------------------
# 8. Profit Comparison by Region
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

region_sales["Profit"].plot(
    kind="bar",
    color="orange"
)

plt.title("Profit by Region")

plt.xlabel("Region")

plt.ylabel("Profit")

plt.grid(axis="y")

plt.show()

# ------------------------------------------------------------
# 9. Scatter Plot: Units Sold vs Revenue
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="Units_Sold",
    y="Revenue",
    hue="Region",
    s=100
)

plt.title("Units Sold vs Revenue")

plt.grid(True)

plt.show()

# ------------------------------------------------------------
# 10. Profit Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(
    df["Profit"],
    bins=10,
    kde=True,
    color="green"
)

plt.title("Profit Distribution")

plt.xlabel("Profit")

plt.ylabel("Frequency")

plt.show()

# ------------------------------------------------------------
# 11. Correlation Heatmap
# ------------------------------------------------------------

numeric_data = df[[
    "Units_Sold",
    "Revenue",
    "Profit"
]]

corr_matrix = numeric_data.corr()

plt.figure(figsize=(6,5))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

# ------------------------------------------------------------
# 12. Top 5 Products by Revenue
# ------------------------------------------------------------

print("\n================ TOP 5 PRODUCTS =================\n")

top_products = product_sales.sort_values(
    by="Revenue",
    ascending=False
).head(5)

print(top_products)

# ------------------------------------------------------------
# 13. Dashboard Summary
# ------------------------------------------------------------

print("\n================ DASHBOARD SUMMARY ================\n")

print("Total Revenue :", df["Revenue"].sum())
print("Total Profit  :", df["Profit"].sum())
print("Total Units Sold :", df["Units_Sold"].sum())

print("\nBest Selling Product :",
      product_sales["Revenue"].idxmax())

print("Highest Revenue Region :",
      region_sales["Revenue"].idxmax())

print("Highest Profit Region :",
      region_sales["Profit"].idxmax())

# ------------------------------------------------------------
# 14. Analytical Observations
# ------------------------------------------------------------

print("\n================ ANALYTICAL OBSERVATIONS ================\n")

print("1. Monthly sales analysis identifies seasonal business trends.")
print("2. Product-wise revenue helps identify best-selling products.")
print("3. Region-wise comparison highlights top-performing markets.")
print("4. Scatter plots reveal the relationship between units sold and revenue.")
print("5. Profit distribution helps understand business profitability.")
print("6. Correlation heatmap identifies relationships among sales variables.")
print("7. Dashboard summary provides a quick business overview.")
print("8. Top-selling products can guide inventory management.")
print("9. Regional performance assists in marketing and expansion decisions.")
print("10. Visual dashboards simplify retail business decision-making.")

# ------------------------------------------------------------
# END OF PROGRAM
# ------------------------------------------------------------