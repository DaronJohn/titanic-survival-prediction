# Titanic Survival Prediction

A machine learning project using the Titanic dataset to predict passenger survival. Built and executed using **Google Colab**.

This project demonstrates:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Logistic regression modeling
- Generating predictions on test data

---

## 📂 Dataset

- `train.csv` – for model training and validation
- `test.csv` – for final predictions

---

## 🔧 1. Data Preprocessing

### Train Set:
- Filled missing `Age` with the median.
- Dropped the `Cabin` column.
- Filled missing `Embarked` with the mode.
- Encoded:
  - `Sex`: `'male' → 0`, `'female' → 1`
  - `Embarked`: `'S' → 0`, `'C' → 1`, `'Q' → 2`

### Test Set:
- Filled missing `Age`, `Fare`, and `Embarked` similarly.
- Encoded categorical columns to match training set.

---

## 📊 2. Exploratory Data Analysis (EDA)

Using **Seaborn** and **Matplotlib**, visualizations included:

- Survival distribution
- Survival by gender
- Survival by class
- Age vs survival patterns by gender

### Insights:
- Females had a much higher survival rate.
- First-class passengers were more likely to survive.
- Most third-class males aged 20–40 did not survive.

---

## 🤖 3. Model Building

**Model Used**: Logistic Regression

### Features:
- `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`

### Training:
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

---

## 📈 4. Evaluation

```python
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)
```

- Accuracy: 0.80
- Confusion matrix and classification report showed good balance

---

## 🚀 5. Predicting on Test Set

- Cleaned test data
- Used trained model to predict survival
- Created final CSV output:

```python
submission.to_csv('titanic_submission.csv', index=False)
```

---

## 📁 Output Files

- `titanic_submission.csv` – for Kaggle or demo submission
- `titanic_survival_prediction.ipynb notebook` – your Google Colab file with full code
- `README.md` – this summary for documentation

---

## 🧠 Learnings

- Cleaned and visualized real-world datasets
- Practiced binary classification and model evaluation
- Understood logistic regression workflow end-to-end
- Worked in a notebook-based cloud environment (Google Colab)
