# IBM-AI-Course
The 'Labs' folder contains all of my labs from course 1 of my IBM AI Professional Certificate which served as practive for my final presentation

# 🌦️ Rain Prediction Classifier

This final presentation builds and evaluates machine learning models to predict **whether it will rain today** based on historical weather features. It includes full preprocessing, feature engineering (e.g., season extraction), hyperparameter tuning via grid search, and model evaluation using both **Random Forest** and **Logistic Regression** classifiers.

---

## 📊 Features & Workflow

- Converts date into **seasonal categories** (e.g., Summer, Winter)
- Preprocesses data using:
  - **Standard scaling** for numerical features
  - **One-hot encoding** for categorical features
- Implements a **ColumnTransformer + Pipeline** for clean feature handling
- Trains two models:
  - 🎯 **Random Forest Classifier** with grid search hyperparameter tuning
  - ⚖️ **Logistic Regression** with L1/L2 regularization
- Evaluates performance using:
  - Accuracy scores
  - Classification reports
  - Confusion matrices
  - Feature importance visualization

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (Pipeline, GridSearchCV, RandomForest, LogisticRegression)
- Seaborn & Matplotlib (for plots)

---

## 🔁 Model Training & Selection

The project uses **Stratified K-Fold Cross Validation** to tune hyperparameters and reduce overfitting. The models are evaluated based on accuracy, class balance, and visual insights.


