# 4_AI_in_Healthcare_Building_a_Life_Saving_Heart_Disease_Predictor
🩺 Heart Disease Prediction using Machine Learning
📘 Project Overview

This project aims to build and evaluate multiple machine learning models that predict whether a patient is likely to have heart disease based on various medical attributes such as age, chest pain type, cholesterol level, and maximum heart rate.

Heart disease prediction is a critical application of data science in healthcare, helping doctors make early and accurate diagnoses. This project demonstrates a full end-to-end classification pipeline, from Exploratory Data Analysis (EDA) to Model Evaluation and Feature Importance.

🎯 Objective

To develop and compare different machine learning classification algorithms that can accurately predict the presence of heart disease and identify the most significant factors influencing it.

🧠 Concepts Covered

Classification Fundamentals – Understanding supervised learning for categorical outcomes.

Exploratory Data Analysis (EDA) – Exploring relationships between medical features and heart disease occurrence.

Data Preprocessing – Handling missing data, encoding categorical features, and scaling numerical values.

Model Building – Implementing Logistic Regression, Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN).

Model Evaluation – Using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix for performance comparison.

Feature Importance – Identifying key medical attributes contributing to heart disease prediction.

📊 Dataset

Source: Heart Disease UCI Dataset (Kaggle)

Number of Records: ~1000 patients

Target Variable:

0 → No Heart Disease

1 → Has Heart Disease

Attributes Include:

Age, Sex, Chest Pain Type (cp), Resting Blood Pressure (trestbps), Cholesterol (chol), Max Heart Rate (thalach), Number of Major Vessels (ca), Thalassemia Type (thal), etc.

⚙️ Project Workflow
Step 1: Data Loading & Inspection

Imported dataset from Kaggle using kagglehub.

Inspected structure, data types, and missing values.

Step 2: Exploratory Data Analysis (EDA)

Visualized target variable distribution.

Analyzed key relationships (e.g., Age vs. Disease, Sex vs. Disease, Chest Pain Type vs. Disease).

Generated a correlation heatmap to find highly related medical factors.

Step 3: Data Preprocessing

Handled missing values using SimpleImputer.

Applied StandardScaler to numerical columns.

Used OneHotEncoder for categorical variables.

Combined transformations using ColumnTransformer and Pipeline.

Step 4: Model Training

Trained and compared four models:

Logistic Regression (Baseline Model)

Random Forest Classifier (Ensemble Model)

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Each model was trained both with and without Pipelines to demonstrate manual preprocessing and automated workflows.

Step 5: Model Evaluation

Evaluated models using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Visualized confusion matrices using Seaborn heatmaps.

Step 6: Feature Importance

Extracted feature importances from the Random Forest model.

Visualized the Top 10 Most Influential Features affecting heart disease prediction.

Step 7: Conclusion

Best Performing Model: Support Vector Machine (SVM) achieved the most balanced performance overall.

Key Predictive Factors: ca, thalach, thal, and cp were found to be the most significant indicators of heart disease.

Random Forest provided interpretability via feature importance, confirming medically relevant insights.

📈 Results Summary
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	~86%	Good	Moderate	0.85
Random Forest	~99%	Excellent	Excellent	0.99
SVM	~91%	High	High	0.90
KNN	~84%	Moderate	Moderate	0.83
🧩 Tools & Libraries Used

Languages: Python

Libraries:

Data Handling: pandas, numpy

Visualization: matplotlib, seaborn

Machine Learning: scikit-learn

Dataset Access: kagglehub

📦 Project Structure
Heart-Disease-Prediction/
│
├── heart_disease_prediction.ipynb     # Main Jupyter/Colab Notebook
├── README.md                          # Project Documentation
├── requirements.txt                    # Dependencies (optional)
└── dataset/                            # Heart Disease CSV file

✅ Key Learnings

Understanding and implementing classification techniques in healthcare.

Importance of proper preprocessing using Scikit-Learn Pipelines.

Interpreting model metrics beyond accuracy (Precision, Recall, F1-Score).

Visualizing and interpreting feature importance for actionable insights.

🚀 Future Improvements

Apply Hyperparameter Tuning using GridSearchCV or RandomizedSearchCV.

Implement Cross-Validation for robust evaluation.

Build a Streamlit or Flask Web App for real-time predictions.

Try Deep Learning Models (e.g., Neural Networks) for improved accuracy
