# ============================================================
# Experiment 2
# Smart City Traffic Management System
# ============================================================

import pandas as pd
import numpy as np
import glob

# ============================================================
# Function 1 : Load Multiple CSV Files
# ============================================================

def load_datasets():
    files = glob.glob("2.*Junction_*.csv")

    if len(files) == 0:
        print("No CSV files found.")
        return None

    dataframes = []

    for file in files:
        print("Loading:", file)
        df = pd.read_csv(file)
        dataframes.append(df)

    return dataframes


# ============================================================
# Function 2 : Merge All Datasets
# ============================================================

def merge_datasets(dataframes):
    merged_df = pd.concat(dataframes, ignore_index=True)

    print("\nDatasets merged successfully.")
    print("Total Records:", merged_df.shape[0])

    return merged_df


# ============================================================
# Function 3 : Dataset Inspection
# ============================================================

def inspect_dataset(df):
    print("\n==============================")
    print("DATASET INFORMATION")
    print("==============================")

    print(df.head())

    print("\nShape:", df.shape)

    print("\nData Types")
    print(df.dtypes)


# ============================================================
# Function 4 : Missing Value Report
# ============================================================

def missing_value_report(df):
    print("\n==============================")
    print("MISSING VALUE REPORT")
    print("==============================")

    missing = df.isnull().sum()
    percentage = (missing / len(df)) * 100

    report = pd.DataFrame({
        "Missing Values": missing,
        "Percentage (%)": percentage.round(2)
    })

    print(report)


# ============================================================
# Function 5 : Detect Corrupted Records
# ============================================================

def detect_corrupted_records(df):
    print("\n==============================")
    print("CORRUPTED RECORD ANALYSIS")
    print("==============================")

    print("\nNegative Vehicle Count")
    print(df[df["Vehicle_Count"] < 0])

    print("\nNegative Speed")
    print(df[df["Average_Speed"] < 0])

    print("\nInvalid Congestion Level")
    print(df[~df["Congestion_Level"].isin(["Low", "Medium", "High"])])

    print("\nNegative Signal Time")
    print(df[df["Signal_Time"] < 0])


# ============================================================
# Function 6 : Duplicate Timestamp Detection
# ============================================================

def duplicate_timestamp(df):
    print("\n==============================")
    print("DUPLICATE TIMESTAMP REPORT")
    print("==============================")

    duplicates = df[df.duplicated(
        subset=["Timestamp", "Location"],
        keep=False
    )]

    if duplicates.empty:
        print("No Duplicate Timestamp Found")
    else:
        print(duplicates)


# ============================================================
# Function 7 : Compare Missing Value Imputation Techniques
# ============================================================

def compare_imputation(df):
    print("\n==============================")
    print("IMPUTATION TECHNIQUES")
    print("==============================")

    numerical = [
        "Vehicle_Count",
        "Average_Speed",
        "Signal_Time",
        "Accidents"
    ]

    mean_df = df.copy()
    median_df = df.copy()
    mode_df = df.copy()

    for col in numerical:
        mean_df[col] = mean_df[col].fillna(mean_df[col].mean())
        median_df[col] = median_df[col].fillna(median_df[col].median())
        mode_df[col] = mode_df[col].fillna(mode_df[col].mode()[0])

    print("Mean Imputation Completed")
    print("Median Imputation Completed")
    print("Mode Imputation Completed")

    return median_df


# ============================================================
# Function 8 : Traffic Density Report
# ============================================================

def traffic_density(df):
    print("\n==============================")
    print("TRAFFIC DENSITY REPORT")
    print("==============================")

    report = df.groupby("Location").agg(
        Total_Vehicles=("Vehicle_Count", "sum"),
        Average_Speed=("Average_Speed", "mean"),
        Total_Accidents=("Accidents", "sum"),
        Average_Signal_Time=("Signal_Time", "mean")
    )

    print(report)

    print("\nMost Congested Location:")
    print(report["Total_Vehicles"].idxmax())


# ============================================================
# Function 9 : Clean Dataset
# ============================================================

def clean_dataset(df):
    df = df.drop_duplicates()

    df = df[df["Vehicle_Count"] >= 0]
    df = df[df["Average_Speed"] >= 0]
    df = df[df["Signal_Time"] >= 0]

    return df


# ============================================================
# Function 10 : Export Dataset
# ============================================================

def export_dataset(df):
    df.to_csv("Traffic_Management_Cleaned.csv", index=False)

    print("\nProcessed Dataset Saved Successfully.")


# ============================================================
# Main Program
# ============================================================

def main():

    datasets = load_datasets()

    if datasets is None:
        return

    traffic_df = merge_datasets(datasets)

    inspect_dataset(traffic_df)

    missing_value_report(traffic_df)

    detect_corrupted_records(traffic_df)

    duplicate_timestamp(traffic_df)

    traffic_df = compare_imputation(traffic_df)

    traffic_density(traffic_df)

    traffic_df = clean_dataset(traffic_df)

    export_dataset(traffic_df)


if __name__ == "__main__":
    main()