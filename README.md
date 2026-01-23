# Higher Studies Decision Prediction (India)

## 📌 Project Overview
This project analyzes and predicts whether Indian students are likely to pursue higher studies abroad or continue education in India.  
The prediction is based on academic performance, financial background, and standardized exam scores.

The goal is to understand **key decision factors** and build an interpretable machine learning model for educational outcome analysis.

---

## 🧠 Problem Statement
Students’ decisions to study abroad depend on multiple factors such as:
- CGPA
- GRE / TOEFL scores
- Family income and available budget
- Research experience
- Field of study
- Scholarship availability

This project explores these factors using **EDA, correlation analysis, and machine learning models**.

---

## 📊 Dataset
- Synthetic dataset inspired by real-world Indian student profiles
- Includes both **India-based** and **abroad-bound** students
- Financial features represented in **LPA (Lakhs Per Annum)**

Key features:
- CGPA
- GRE_Score
- TOEFL_Score
- Family_Income_LPA
- Budget_LPA
- Research_Experience
- Scholarship
- Study_Destination (Target variable)

---

## 🔍 Exploratory Data Analysis
- Distribution analysis of academic and financial features
- Correlation analysis with study destination
- Feature importance using Random Forest

Notebook:
- `notebooks/01_eda.ipynb`

---

## 🤖 Machine Learning Approach
- Label encoding for categorical targets
- Random Forest Classifier
- Feature importance analysis to identify key drivers

Important insights:
- Academic performance and budget are strong predictors
- Financial background plays a significant role in decision outcomes

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## 🚀 Future Improvements
- Add real-world datasets
- Try Logistic Regression and XGBoost
- Model performance comparison
- Deploy simple Streamlit app

---

## 👤 Author
**Veera Venkat Sai Kondaiahpalpu**  
M.Sc. Applied Computer Science, SRH University Stuttgart
