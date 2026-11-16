"""
Fundamentals of Machine Learning - Laboratory Work 1
Introduction to Data Analysis: Preprocessing and Visualization

Variant 2: Boston Housing

Steps performed:
1. Load the dataset and inspect it (.head(), .info(), .describe()).
2. Exploratory data analysis: missing values, correlation matrix / heatmap,
   the feature most strongly correlated with MEDV, a scatter plot of that
   feature against MEDV, and a histogram of the crime rate (CRIM).
3. Data preprocessing: normalize all numerical features to the [0, 1] range
   (Min-Max scaling).
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display available; save figures to file instead
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "BostonHousing.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TARGET = "MEDV"


# ------------------- 2.1 Dataset loading and initial inspection -------------------

def load_and_inspect(path):
    """Load the dataset and print first rows, structure and basic statistics."""
    df = pd.read_csv(path)

    print("First 5 rows:")
    print(df.head())

    print("\nDataFrame info:")
    df.info()

    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nStatistical characteristics (describe):")
    print(df.describe())

    return df


# ------------------- 2.2 Exploratory data analysis and visualization -------------------

def plot_correlation_heatmap(df, filename):
    """Build the correlation matrix and visualize it as a heatmap."""
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix - Boston Housing Dataset")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    return corr


def find_feature_most_correlated_with_target(corr, target):
    """Return the name of the feature (excluding the target itself) with the
    largest absolute correlation with the target variable."""
    target_corr = corr[target].drop(labels=[target, "CAT. MEDV"], errors="ignore")
    best_feature = target_corr.abs().idxmax()
    return best_feature, target_corr[best_feature]


def plot_scatter_vs_target(df, feature, target, filename):
    """Scatter plot of the most correlated feature against the target variable."""
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=feature, y=target, alpha=0.6)
    plt.title(f"{feature} vs {target}")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_crim_histogram(df, filename):
    """Histogram of the crime rate (CRIM) distribution."""
    plt.figure(figsize=(7, 5))
    sns.histplot(df["CRIM"], bins=40, kde=True, color="steelblue")
    plt.title("Distribution of Per-Capita Crime Rate (CRIM)")
    plt.xlabel("CRIM")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# ------------------- 2.3 Data preprocessing -------------------

def normalize_features(df, target, exclude=("CAT. MEDV",)):
    """
    Normalize (Min-Max scale to [0, 1]) all numerical feature columns,
    excluding the target variable and any explicitly excluded columns.
    """
    df_scaled = df.copy()
    feature_cols = [c for c in df.columns if c != target and c not in exclude]

    scaler = MinMaxScaler()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

    return df_scaled, feature_cols


# ------------------- MAIN PROGRAM -------------------

def main():
    print("=" * 60)
    print("2.1 DATASET LOADING AND INITIAL INSPECTION")
    print("=" * 60)
    df = load_and_inspect(DATA_FILE)

    print("\n" + "=" * 60)
    print("2.2 EXPLORATORY DATA ANALYSIS AND VISUALIZATION")
    print("=" * 60)

    heatmap_path = os.path.join(FIG_DIR, "correlation_heatmap.png")
    corr = plot_correlation_heatmap(df, heatmap_path)
    print(f"Correlation heatmap saved to '{heatmap_path}'.")

    best_feature, corr_value = find_feature_most_correlated_with_target(corr, TARGET)
    print(f"\nFeature most strongly correlated with {TARGET}: "
          f"'{best_feature}' (correlation = {corr_value:.4f})")

    scatter_path = os.path.join(FIG_DIR, f"scatter_{best_feature}_vs_{TARGET}.png")
    plot_scatter_vs_target(df, best_feature, TARGET, scatter_path)
    print(f"Scatter plot saved to '{scatter_path}'.")

    crim_hist_path = os.path.join(FIG_DIR, "crim_histogram.png")
    plot_crim_histogram(df, crim_hist_path)
    print(f"CRIM histogram saved to '{crim_hist_path}'.")

    print("\n" + "=" * 60)
    print("2.3 DATA PREPROCESSING - NORMALIZATION")
    print("=" * 60)

    df_scaled, feature_cols = normalize_features(df, TARGET)
    print(f"Normalized {len(feature_cols)} numerical feature columns to the [0, 1] range:")
    print(feature_cols)

    print("\nNormalized data (first 5 rows):")
    print(df_scaled.head())

    print("\nPost-normalization min/max check (should be 0.0 and 1.0):")
    print(df_scaled[feature_cols].agg(["min", "max"]))

    output_path = os.path.join(BASE_DIR, "boston_housing_normalized.csv")
    df_scaled.to_csv(output_path, index=False)
    print(f"\nNormalized dataset saved to '{output_path}'.")


if __name__ == "__main__":
    main()
