# ==============================================
# STUDY ABROAD ANALYSIS + PREDICTION PIPELINE
# ==============================================

# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 2. Load Dataset ---
# Make sure your dataset is in the same directory
df = pd.read_csv("study_abroad_dataset_with_india.csv")

# --- 3. Clean & Prepare Columns ---
df.columns = df.columns.str.strip().str.replace(" ", "_")

print("✅ Columns validated:", df.columns.tolist())
print("\nSample of data:")
print(df.head())

# --- 4. Basic Info ---
print("\n--- DATASET INFO ---")
print(df.info())
print("\n--- SUMMARY STATISTICS ---")
print(df.describe(include='all'))

# --- 5. Encode Study Destination ---
# Convert "India" -> 0 and "Abroad" -> 1
df['Study_Destination'] = df['Study_Destination'].map({'India': 0, 'Abroad': 1})
if df['Study_Destination'].isnull().any():
    print("⚠️ Some Study_Destination values were invalid. Filling missing with 0 (India).")
    df['Study_Destination'] = df['Study_Destination'].fillna(0)

# --- 6. Exploratory Data Analysis (EDA) ---
sns.set(style="whitegrid", palette="muted")

# (a) Study Destination Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Study_Destination', data=df)
plt.title("Study Destination Distribution (0=India, 1=Abroad)")
plt.show()

# (b) CGPA vs Study Destination
plt.figure(figsize=(6,4))
sns.boxplot(x='Study_Destination', y='CGPA', data=df)
plt.title("CGPA vs Study Destination")
plt.show()

# (c) GRE vs Study Destination
plt.figure(figsize=(6,4))
sns.boxplot(x='Study_Destination', y='GRE_Score', data=df)
plt.title("GRE Score vs Study Destination")
plt.show()

# (d) Budget vs Study Destination
plt.figure(figsize=(6,4))
sns.boxplot(x='Study_Destination', y='Budget_LPA', data=df)
plt.title("Budget vs Study Destination")
plt.show()

# (e) Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# --- 7. Feature Encoding & Scaling ---
cat_cols = ['Field_of_Study', 'Country']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Define features (X) and target (y)
X = df.drop(['Study_Destination'], axis=1)
y = df['Study_Destination']

# --- 8. Train–Test Split ---
# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

# --- 9. Standardize Numeric Features ---
num_cols = [
    col for col in X_train.columns
    if X_train[col].dtype in [np.float64, np.int64] and col not in cat_cols
]

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# --- 10. Logistic Regression ---
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# --- 11. Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# --- 12. Confusion Matrix ---
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- 13. Feature Importance Plot ---
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

print("\n✅ Full analysis completed successfully!")
