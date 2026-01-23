import numpy as np
import pandas as pd

# Set seed
np.random.seed(42)

# Number of students
n_students = 500

# --- Academic Features (from original dataset style) ---
cgpa = np.round(np.random.uniform(6, 10, n_students), 2)
gre = np.random.randint(290, 340, n_students)
toefl = np.random.randint(90, 120, n_students)
research = np.random.choice([0,1], n_students)
fields = ['CS','EE','MBA','ME','CE']
field_of_study = np.random.choice(fields, n_students)

# --- Household Income (LPA) based on India distribution ---
income_brackets = ['Low','Middle','High']
bracket_ranges = {'Low': (3,7), 'Middle': (7,20), 'High': (20,50)}
prob = [0.4, 0.4, 0.2]  # 40% Low, 40% Middle, 20% High

income_cat = np.random.choice(income_brackets, n_students, p=prob)
family_income = [np.round(np.random.uniform(*bracket_ranges[b]),2) for b in income_cat]

# --- Budget (40-80% of family income) ---
budget = np.round(np.array(family_income) * np.random.uniform(0.4,0.8, n_students), 2)

# --- Scholarship (0=No,1=Yes) ---
scholarship = []
for i in range(n_students):
    merit_prob = 0.3
    if cgpa[i] > 8 and gre[i] > 310 and research[i]==1:
        merit_prob = 0.7
    need_prob = 0.1
    if family_income[i] < 10:
        need_prob = 0.5
    prob_scholarship = max(merit_prob, need_prob)
    scholarship.append(np.random.choice([0,1], p=[1-prob_scholarship, prob_scholarship]))

# --- Study Destination (Abroad/India) ---
study_dest = []
for i in range(n_students):
    if cgpa[i] > 8 and gre[i] > 310 and budget[i] > 15:  # tuition approx. 15 LPA
        study_dest.append('Abroad')
    else:
        study_dest.append('India')

# --- Create DataFrame ---
df = pd.DataFrame({
    'CGPA': cgpa,
    'GRE_Score': gre,
    'TOEFL_Score': toefl,
    'Research_Experience': research,
    'Field_of_Study': field_of_study,
    'Family_Income_LPA': family_income,
    'Budget_LPA': budget,
    'Scholarship': scholarship,
    'Study_Destination': study_dest
})

# --- Save CSV ---
df.to_csv('study_abroad_dataset_with_income.csv', index=False)
print("Dataset created! Shape:", df.shape)
df.head()
