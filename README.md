# Higher Studies Decision Prediction (India)

A machine learning project that predicts whether an Indian student is likely to pursue
higher education **abroad or in India**, based on academic performance, financial background,
and standardised test scores.

---

## Project Overview

Understanding what drives Indian students to study abroad is a meaningful question — this
project was built out of personal interest while learning end-to-end ML development.

The model analyses factors like CGPA, GRE/TOEFL scores, family income, budget, and research
experience to classify a student's likely study destination.

---

## Dataset

No publicly available dataset exists that contains individual Indian student records with
financial data and a binary "study abroad vs India" label at meaningful scale. The closest
public dataset (Graduate Admissions, Kaggle) has only ~500 rows and predicts admission
probability, not destination.

This project uses a **synthetically generated dataset of 10,000 students** built to reflect
realistic Indian student demographics:

- Feature distributions are calibrated to published data (ETS score reports, UGC income
  brackets, AISHE survey statistics)
- The study destination label is assigned via a **probabilistic sigmoid model**, not hard
  threshold rules — this creates realistic class overlap and makes the problem genuinely
  non-trivial
- Gaussian noise is added to simulate unmeasured real-world factors (visa difficulty,
  university rankings, personal preference)

**Result:** The model reaches ~80% accuracy (0.88 ROC-AUC), not 100% — by design.

| Feature | Description |
|---|---|
| CGPA | Undergraduate GPA (5.0 – 10.0) |
| GRE_Score | Graduate Record Exam (260 – 340) |
| TOEFL_Score | English proficiency (70 – 120) |
| Family_Income_LPA | Household income in Lakhs Per Annum |
| Budget_LPA | Available education budget in LPA |
| Research_Experience | Has research publication (0 / 1) |
| Scholarship | Has scholarship (0 / 1) |
| Field_of_Study | CS / EE / MBA / ME / CE |
| **Study_Destination** | **Target — India / Abroad** |

Class distribution: **65% India, 35% Abroad**

---

## ML Pipeline

All preprocessing is wrapped inside a scikit-learn `Pipeline` with a `ColumnTransformer`,
which ensures:

- `StandardScaler` and `OneHotEncoder` are fit **only on training data**
- The test set is never seen during model selection or hyperparameter tuning
- No data leakage between train and test

```
Data (10,000 rows)
    │
    ├── 80% Train set
    │       │
    │       └── 5-Fold StratifiedKFold CV → GridSearchCV
    │               │
    │               └── Best hyperparameters selected by ROC-AUC
    │
    └── 20% Test set (held out — evaluated once at the end)
```

---

## Model Results

GridSearchCV compared Logistic Regression and Random Forest across regularisation strength
and tree depth/leaf constraints.

| Model | CV AUC (train) | Test Accuracy | Test ROC-AUC |
|---|---|---|---|
| **Logistic Regression** | **0.8853** | **80.2%** | **0.8809** |
| Random Forest | 0.8761 | 79.8% | 0.8732 |

**Winner: Logistic Regression** — the linear model edges out the ensemble, suggesting the
features have a largely linear relationship with the outcome.

Detailed test-set metrics (Logistic Regression):

```
              precision    recall  f1-score   support
       India       0.88      0.81      0.84      1304
      Abroad       0.69      0.79      0.73       696
    accuracy                           0.80      2000
```

Best hyperparameters: `C=0.01` (strong L2 regularisation)

---

## Key Findings

- **Budget_LPA** is the strongest single predictor — financial capacity outweighs academic
  scores in determining who goes abroad
- **CGPA and GRE** are significant but secondary to financial factors
- **Research experience and scholarship** provide a moderate positive signal
- Logistic Regression outperforming Random Forest indicates the decision boundary is
  approximately linear in the standardised feature space

---

## Predictions

Model predictions on the held-out test set (2,000 students) are saved at:
`predictions/test_predictions.csv`

Columns: `CGPA`, `GRE_Score`, `Budget_LPA`, `actual_label`, `predicted_label`,
`probability_abroad`, `correct`

---

## Project Structure

```
Study_abroad_prediction/
├── data/
│   ├── study_abroad_10k.csv            ← main dataset (10k rows, synthetic)
│   ├── study_abroad_dataset_with_income.csv   ← early iteration (500 rows)
│   └── study_abroad_dataset_with_india.csv    ← early iteration (combined)
├── models/
│   └── best_model.pkl                  ← fitted pipeline (gitignored)
├── notebooks/
│   └── 01_eda.ipynb
├── predictions/
│   └── test_predictions.csv            ← test set predictions
└── src/
    ├── generate_dataset.py             ← generates study_abroad_10k.csv
    ├── train_model.py                  ← main training pipeline
    ├── model_comparison.py             ← 5-fold CV across 4 model types
    ├── model_validation.py             ← standalone cross-validation
    ├── stress_test_model.py            ← evaluation on borderline students
    ├── balanced_dataset.py             ← early dataset balancing (legacy)
    ├── dataset.py                      ← early data generation (legacy)
    └── indian_student_dataset.py       ← early India student generation (legacy)
```

---

## How to Run

```bash
# Install dependencies
pip install -r Study_abroad_prediction/requirements.txt

# Generate the dataset
python Study_abroad_prediction/src/generate_dataset.py

# Train the model (saves model + predictions)
python Study_abroad_prediction/src/train_model.py

# Compare all models with cross-validation
python Study_abroad_prediction/src/model_comparison.py
```

---

## Technologies

- Python 3.11
- pandas, NumPy
- scikit-learn (Pipeline, GridSearchCV, ColumnTransformer)
- Matplotlib, Seaborn

---

## Possible Next Steps

- Collect real survey data from Indian student communities
- Add SHAP values for individual prediction explanations
- Deploy as a Streamlit app with a student-facing prediction form
- Experiment with XGBoost and CalibratedClassifierCV

---

## Author

**Veera Venkat Sai Kondaiahpalpu**
M.Sc. Applied Computer Science, SRH University Stuttgart
