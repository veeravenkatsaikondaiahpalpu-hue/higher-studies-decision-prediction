"""
Generates a synthetic dataset of 10,000 Indian students.

Design choices:
- Feature distributions are calibrated to published Indian student demographics
  (income brackets, GRE/TOEFL/CGPA ranges from ETS and UGC reports).
- Target is assigned via a probabilistic sigmoid model, NOT hard threshold rules.
  This creates realistic class overlap so the model cannot reach 100% accuracy.
- Gaussian noise is added to the decision boundary to mimic unmeasured factors
  (family preference, visa difficulty, university rankings) that real data has.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "study_abroad_10k.csv"

N_STUDENTS = 10_000
SEED = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def generate(n: int = N_STUDENTS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Academic features ---
    cgpa = np.clip(rng.normal(7.8, 0.9, n), 5.0, 10.0).round(2)
    gre = np.clip(rng.normal(308, 14, n), 260, 340).astype(int)
    toefl = np.clip(rng.normal(100, 10, n), 70, 120).astype(int)
    research = rng.choice([0, 1], n, p=[0.65, 0.35])
    field = rng.choice(
        ["CS", "EE", "MBA", "ME", "CE"], n, p=[0.30, 0.20, 0.20, 0.15, 0.15]
    )

    # --- Financial features ---
    bracket = rng.choice(["Low", "Middle", "High"], n, p=[0.40, 0.40, 0.20])
    bracket_ranges = {"Low": (3, 8), "Middle": (8, 25), "High": (25, 60)}
    family_income = np.array(
        [rng.uniform(*bracket_ranges[b]) for b in bracket]
    ).round(2)

    budget_ratio = rng.uniform(0.30, 0.70, n)
    budget = (family_income * budget_ratio).round(2)

    # Scholarship probability: merit-based OR need-based
    merit_prob = _sigmoid(0.8 * (cgpa - 8.0) + 0.5 * (gre - 310) / 15)
    need_prob = _sigmoid(-0.6 * (family_income - 10) / 8)
    scholarship_prob = np.clip(np.maximum(merit_prob, need_prob) * 0.6, 0.05, 0.70)
    scholarship = rng.binomial(1, scholarship_prob)

    # Scholarship adds liquidity to budget
    scholarship_boost = rng.uniform(2, 8, n) * scholarship
    budget = (budget + scholarship_boost).round(2)

    # --- Probabilistic target (sigmoid + noise) ---
    # Coefficients represent relative importance of each factor.
    # Intercept is negative so the base rate favours India (~40% abroad).
    log_odds = (
        1.2 * (cgpa - 7.8) / 0.9           # academic strength
        + 0.8 * (gre - 308) / 14            # GRE score
        + 0.5 * (toefl - 100) / 10          # English proficiency
        + 1.5 * (budget - 12) / 10          # financial capacity (strongest driver)
        + 0.7 * (family_income - 15) / 15   # family wealth
        + 0.6 * research                    # research experience
        + 0.4 * scholarship                 # scholarship
        - 1.2                               # intercept: biases toward India
        + rng.normal(0, 0.8, n)             # irreducible noise
    )
    p_abroad = _sigmoid(log_odds)
    study_dest = np.where(rng.uniform(0, 1, n) < p_abroad, "Abroad", "India")

    return pd.DataFrame({
        "CGPA": cgpa,
        "GRE_Score": gre,
        "TOEFL_Score": toefl,
        "Family_Income_LPA": family_income,
        "Budget_LPA": budget,
        "Research_Experience": research,
        "Scholarship": scholarship,
        "Field_of_Study": field,
        "Study_Destination": study_dest,
    })


def main():
    df = generate()
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Dataset shape : {df.shape}")
    print("\nClass distribution:")
    counts = df["Study_Destination"].value_counts()
    for label, count in counts.items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
