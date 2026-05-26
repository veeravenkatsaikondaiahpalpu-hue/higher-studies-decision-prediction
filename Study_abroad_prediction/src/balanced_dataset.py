from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "study_abroad_dataset_with_india.csv"
OUTPUT_FILE = DATA_DIR / "study_abroad_dataset_balanced.csv"


def main():
    np.random.seed(42)

    df = pd.read_csv(INPUT_FILE)
    df["Country"] = df["Country"].astype(str).str.strip().str.title()
    df["Study_Destination"] = np.where(df["Country"] == "India", "India", "Abroad")

    print("Before balancing:")
    print(df["Study_Destination"].value_counts())

    df_india = df[df["Study_Destination"] == "India"].copy()
    df_abroad = df[df["Study_Destination"] == "Abroad"].copy()

    india_sample = df_india.sample(n=300, random_state=42, replace=False)
    # replace=True in case there are fewer than 700 abroad rows in the source
    abroad_sample = df_abroad.sample(n=700, random_state=42, replace=True)

    df_balanced = (
        pd.concat([india_sample, abroad_sample], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # Country column is always India because the source data context is India
    df_balanced["Country"] = "India"

    df_balanced.to_csv(OUTPUT_FILE, index=False)

    print("\nBalanced dataset created!")
    print(df_balanced["Study_Destination"].value_counts())
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
