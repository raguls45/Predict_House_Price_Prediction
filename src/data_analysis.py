import pandas as pd
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Avoid the Tk backend, which requires Tcl/Tk.

import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
output_dir = PROJECT_ROOT / "reports"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = PROJECT_ROOT / "data" / "Real estate valuation data set.xlsx"

df = pd.read_excel(file_path)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nDATA TYPES:")
print(df.dtypes)


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\nMISSING VALUES:")
print(df.isnull().sum())


# ============================================================
# 4. DUPLICATES
# ============================================================

print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())


# ============================================================
# 5. STATISTICAL SUMMARY
# ============================================================

print("\nSTATISTICAL SUMMARY:")
print(df.describe())


# ============================================================
# 6. OUTLIER DETECTION USING IQR
# ============================================================

print("\n" + "=" * 60)
print("OUTLIER ANALYSIS")
print("=" * 60)

numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(f"\nColumn: {column}")
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower Limit: {lower_limit:.2f}")
    print(f"Upper Limit: {upper_limit:.2f}")
    print(f"Number of Outliers: {len(outliers)}")


# ============================================================
# 7. REMOVE ID COLUMN FOR ANALYSIS
# ============================================================

analysis_df = df.drop(columns=["No"])


# ============================================================
# 8. HOUSE PRICE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Y house price of unit area"],
    bins=30,
    kde=True
)

plt.title("Distribution of House Prices")
plt.xlabel("House Price of Unit Area")
plt.ylabel("Number of Properties")

plt.tight_layout()
plt.savefig(output_dir / "house_price_distribution.png", bbox_inches="tight")
plt.close()


# ============================================================
# 9. HOUSE PRICE BOXPLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["Y house price of unit area"]
)

plt.title("House Price Boxplot")
plt.xlabel("House Price of Unit Area")

plt.tight_layout()
plt.savefig(output_dir / "house_price_boxplot.png", bbox_inches="tight")
plt.close()


# ============================================================
# 10. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(11, 8))

correlation = analysis_df.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig(output_dir / "correlation_heatmap.png", bbox_inches="tight")
plt.close()


# ============================================================
# 11. HOUSE AGE VS PRICE
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="X2 house age",
    y="Y house price of unit area"
)

plt.title("House Age vs House Price")
plt.xlabel("House Age")
plt.ylabel("House Price")

plt.tight_layout()
plt.savefig(output_dir / "house_age_vs_price.png", bbox_inches="tight")
plt.close()


# ============================================================
# 12. MRT DISTANCE VS PRICE
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="X3 distance to the nearest MRT station",
    y="Y house price of unit area"
)

plt.title("MRT Distance vs House Price")
plt.xlabel("Distance to MRT")
plt.ylabel("House Price")

plt.tight_layout()
plt.savefig(output_dir / "mrt_distance_vs_price.png", bbox_inches="tight")
plt.close()


# ============================================================
# 13. CONVENIENCE STORES VS PRICE
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="X4 number of convenience stores",
    y="Y house price of unit area"
)

plt.title("Convenience Stores vs House Price")
plt.xlabel("Number of Convenience Stores")
plt.ylabel("House Price")

plt.tight_layout()
plt.savefig(output_dir / "convenience_stores_vs_price.png", bbox_inches="tight")
plt.close()


print("\n" + "=" * 60)
print("DATA ANALYSIS COMPLETED")
print("=" * 60)