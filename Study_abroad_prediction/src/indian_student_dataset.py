import pandas as pd
import numpy as np

# Load your abroad dataset
df_abroad = pd.read_csv("notebook/data/study_abroad_dataset_final.csv")

# --- Create synthetic India students ---
n = len(df_abroad)  # same number as abroad students

df_india = pd.DataFrame({
    "CGPA": np.round(np.random.normal(loc=7.2, scale=0.8, size=n), 2).clip(5.0, 9.5),
    "GRE_Score": np.random.randint(280, 315, n),
    "TOEFL_Score": np.random.randint(75, 105, n),
    "Research_Experience": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
    "Field_of_Study": np.random.choice(df_abroad["Field_of_Study"].unique(), size=n),
    "Family_Income_LPA": np.round(np.random.normal(loc=10, scale=3, size=n), 2).clip(2.0, 20.0),
    "Budget_LPA": np.round(np.random.normal(loc=5, scale=2, size=n), 2).clip(1.0, 12.0),
    "Scholarship": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
    "Study_Destination": "India",
    "Country": "India",
    "index": np.arange(len(df_abroad), len(df_abroad) + n),
    "Indian_Students_Abroad": np.random.randint(0, 500, n),
    "Country_Indian_Student_Percentage": np.random.uniform(0.0, 0.5, n)
})

# --- Combine both datasets ---
df_combined = pd.concat([df_abroad, df_india], ignore_index=True)

# --- Save the new dataset ---
df_combined.to_csv("study_abroad_dataset_with_india.csv", index=False)

print("✅ Added India-based student data successfully!")
print("New dataset shape:", df_combined.shape)
print(df_combined['Study_Destination'].value_counts())
print(df_combined.head())
