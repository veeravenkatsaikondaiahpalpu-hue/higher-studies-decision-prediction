from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "study_abroad_dataset_with_income.csv"


def generate(n_students: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    cgpa = rng.uniform(6, 10, n_students).round(2)
    gre = rng.integers(290, 340, n_students)
    toefl = rng.integers(90, 120, n_students)
    research = rng.choice([0, 1], n_students)
    field_of_study = rng.choice(["CS", "EE", "MBA", "ME", "CE"], n_students)

    # Family income drawn from a mixture of three income brackets
    bracket_ranges = {"Low": (3, 7), "Middle": (7, 20), "High": (20, 50)}
    income_cat = rng.choice(["Low", "Middle", "High"], n_students, p=[0.4, 0.4, 0.2])
    family_income = np.array([
        rng.uniform(*bracket_ranges[b]) for b in income_cat
    ]).round(2)

    budget = (family_income * rng.uniform(0.4, 0.8, n_students)).round(2)

    scholarship = []
    for i in range(n_students):
        merit_prob = 0.7 if (cgpa[i] > 8 and gre[i] > 310 and research[i] == 1) else 0.3
        need_prob = 0.5 if family_income[i] < 10 else 0.1
        prob = max(merit_prob, need_prob)
        scholarship.append(rng.choice([0, 1], p=[1 - prob, prob]))

    # Deterministic rule: students with strong academics and budget go abroad
    study_dest = [
        "Abroad" if (cgpa[i] > 8 and gre[i] > 310 and budget[i] > 15) else "India"
        for i in range(n_students)
    ]

    return pd.DataFrame({
        "CGPA": cgpa,
        "GRE_Score": gre,
        "TOEFL_Score": toefl,
        "Research_Experience": research,
        "Field_of_Study": field_of_study,
        "Family_Income_LPA": family_income,
        "Budget_LPA": budget,
        "Scholarship": scholarship,
        "Study_Destination": study_dest,
    })


def main():
    df = generate()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset created — shape: {df.shape}")
    print(df["Study_Destination"].value_counts())
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
