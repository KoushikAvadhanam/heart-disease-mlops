
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def main():
    # Project root: the folder containing src/
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading the UCI Heart Disease dataset...")

    # Fetch dataset ID 45 from the UCI Machine Learning Repository
    heart_disease = fetch_ucirepo(id=45)

    # Get features and target
    features = heart_disease.data.features.copy()
    targets = heart_disease.data.targets.copy()

    # Combine features and target into one DataFrame
    df = pd.concat([features, targets], axis=1)

    # Save the original downloaded data before cleaning
    output_path = raw_dir / "heart_disease_raw.csv"
    df.to_csv(output_path, index=False)

    print("\nDownload successful!")
    print(f"Saved to: {output_path}")
    print(f"\nDataset shape: {df.shape}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nTarget distribution:")
    target_column = targets.columns[0]
    print(df[target_column].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
