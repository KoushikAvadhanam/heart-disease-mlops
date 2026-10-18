
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "heart_disease_raw.csv"
    output_dir = project_root / "screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Load the original dataset
    df = pd.read_csv(data_path)

    # Convert the original 5-class target to binary
    df["target"] = (df["num"] > 0).astype(int)

    print("=" * 55)
    print("HEART DISEASE DATASET - EDA")
    print("=" * 55)

    print(f"\nDataset shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)

    print("\nStatistical summary:")
    print(df.describe().round(2))

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nBinary target distribution:")
    print(df["target"].value_counts().sort_index())
    print("\nTarget percentages:")
    print((df["target"].value_counts(normalize=True) * 100).round(2))

    # 1. Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    plt.figure(figsize=(8, 5))
    if not missing.empty:
        sns.barplot(x=missing.index, y=missing.values)
        plt.ylabel("Number of missing values")
        plt.xlabel("Feature")
        plt.title("Missing Values by Feature")
    else:
        plt.text(0.5, 0.5, "No missing values", ha="center")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "missing_values.png", dpi=150)
    plt.close()

    # 2. Target class balance
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="target")
    plt.title("Heart Disease Class Balance")
    plt.xlabel("Target (0 = absent, 1 = present)")
    plt.ylabel("Number of patients")
    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png", dpi=150)
    plt.close()

    # 3. Histograms of numeric columns
    numeric_columns = [
        "age", "trestbps", "chol", "thalach", "oldpeak"
    ]

    df[numeric_columns].hist(
        bins=15, figsize=(12, 8), edgecolor="black"
    )
    plt.suptitle("Distributions of Selected Patient Features")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_histograms.png", dpi=150)
    plt.close()

    # 4. Correlation heatmap
    # Includes the original target for exploratory analysis.
    # Correlations involving coded categories need careful interpretation.
    plt.figure(figsize=(12, 9))
    correlation = df.drop(columns=["num"]).corr(numeric_only=True)
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=False
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()

    # 5. Age distribution by binary target
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="age",
        hue="target",
        bins=15,
        multiple="layer",
        alpha=0.5
    )
    plt.title("Age Distribution by Heart Disease Class")
    plt.xlabel("Age")
    plt.ylabel("Number of patients")
    plt.tight_layout()
    plt.savefig(output_dir / "age_by_target.png", dpi=150)
    plt.close()

    print("\nEDA completed successfully.")
    print(f"Plots saved in: {output_dir}")


if __name__ == "__main__":
    main()
