from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "study_abroad_dataset_with_income.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "study_abroad_dataset_with_india.csv"


def add_india_students(df_abroad: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(df_abroad)

    df_india = pd.DataFrame({
        "CGPA": np.clip(rng.normal(7.2, 0.8, n), 5.0, 9.5).round(2),
        "GRE_Score": rng.integers(280, 315, n),
        "TOEFL_Score": rng.integers(75, 105, n),
        "Research_Experience": rng.choice([0, 1], n, p=[0.7, 0.3]),
        "Field_of_Study": rng.choice(df_abroad["Field_of_Study"].unique(), n),
        "Family_Income_LPA": np.clip(rng.normal(10, 3, n), 2.0, 20.0).round(2),
        "Budget_LPA": np.clip(rng.normal(5, 2, n), 1.0, 12.0).round(2),
        "Scholarship": rng.choice([0, 1], n, p=[0.85, 0.15]),
        "Study_Destination": "India",
        "Country": "India",
        "Indian_Students_Abroad": rng.integers(0, 500, n),
        "Country_Indian_Student_Percentage": rng.uniform(0.0, 0.5, n).round(4),
    })

    return pd.concat([df_abroad, df_india], ignore_index=True)


def main():
    df_abroad = pd.read_csv(INPUT_FILE)
    df_combined = add_india_students(df_abroad)
    df_combined.to_csv(OUTPUT_FILE, index=False)

    print("Combined dataset created!")
    print("Shape:", df_combined.shape)
    print(df_combined["Study_Destination"].value_counts())
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
