'''import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataframe (replace 'your_data.csv' with your actual file)
df = pd.read_csv('notebook/data/study_abroad_dataset_final.csv')

# Copy your dataframe
df_corr = df.copy()

# Select numeric features and outcome
numeric_features = ['CGPA', 'GRE_Score', 'TOEFL_Score', 'Family_Income_LPA']
target = 'Study_Destination'

# Compute correlation of numeric features with outcome
correlations = df_corr[numeric_features + [target]].corr()[target].sort_values(ascending=False)

print("Correlation of numeric features with Study_Destination:")
print(correlations)

# Optional: visualize with a heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df_corr[numeric_features + [target]].corr()[[target]], annot=True, cmap='coolwarm')
plt.title("Correlation with Study_Destination")
plt.show()'''
# ===============================================
# CORRELATION ANALYSIS WITH LABEL ENCODING
# ===============================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# --- 1. Load Dataset ---
df = pd.read_csv("notebook/data/study_abroad_dataset_final.csv")

# --- 2. Copy to avoid modifying the original ---
df_corr = df.copy()

# --- 3. Encode Study_Destination (convert countries to numeric) ---
le = LabelEncoder()
df_corr['Study_Destination'] = le.fit_transform(df_corr['Study_Destination'].astype(str))

# You can check the mapping:
country_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Country Encoding Mapping:")
for country, code in country_mapping.items():
    print(f"{country}: {code}")

# --- 4. Select numeric features ---
numeric_features = ['CGPA', 'GRE_Score', 'TOEFL_Score', 'Family_Income_LPA']
target = 'Study_Destination'

# --- 5. Compute correlations ---
correlations = df_corr[numeric_features + [target]].corr()[target].sort_values(ascending=False)

print("\nCorrelation of numeric features with Study_Destination:")
print(correlations)

# --- 6. Visualize correlations ---
plt.figure(figsize=(6, 4))
sns.heatmap(df_corr[numeric_features + [target]].corr()[[target]], annot=True, cmap='coolwarm')
plt.title("Correlation of Numeric Features with Study Destination")
plt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Encode categorical target
le = LabelEncoder()
df['Study_Destination'] = le.fit_transform(df['Study_Destination'].astype(str))

# Select numeric features
X = df[['CGPA', 'GRE_Score', 'TOEFL_Score', 'Family_Income_LPA']]
y = df['Study_Destination']

# Train model
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances)

# Plot
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,4))
sns.barplot(x=importances.values, y=importances.index, color="skyblue")
plt.title("Feature Importance (Random Forest)")
plt.show()
plt.figure(figsize=(7,4))
sns.barplot(x=importances.values, y=importances.index, color="#4C29CA", orient='h')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()



