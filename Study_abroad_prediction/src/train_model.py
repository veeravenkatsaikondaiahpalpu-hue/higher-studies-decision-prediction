import pandas as pd

# Load your dataset
df = pd.read_csv("notebook/data/study_abroad_dataset_with_india.csv")

# Shuffle (randomize) all rows
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the randomized dataset
df_shuffled.to_csv("study_abroad_dataset_randomized.csv", index=False)

print("✅ Dataset rows randomized successfully!")
print("New dataset shape:", df_shuffled.shape)
print(df_shuffled.head())
