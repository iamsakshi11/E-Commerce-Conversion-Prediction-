# 🛒 E-Commerce Conversion Prediction Challenge
### Summer Analytics 2026 – Mini Hackathon Week 2

## 📌 Project Overview

This project was developed as part of the **Summer Analytics 2026 Mini Hackathon – Week 2**.

The objective was to build a machine learning model that predicts whether a user will convert based on demographic information, browsing behavior, engagement metrics, and marketing-related features.

### Target Variable

| Value | Meaning |
|---------|---------|
| 1 | Converted |
| 0 | Not Converted |

---

## 🎯 Problem Statement

Given user-level e-commerce data, predict whether a customer will convert.

The final evaluation metric for the competition is:

**F1 Score**

which balances Precision and Recall and is well-suited for classification problems.

---

## 📂 Dataset Description

The challenge provided the following datasets:

| File | Description |
|--------|------------|
| train.csv | Training dataset containing features and target labels |
| public_test.csv | Validation dataset containing features and target labels |
| private_test.csv | Unseen dataset used for final prediction |
| sample_submission.csv | Required submission format |

### Features

| Feature | Description |
|----------|------------|
| User_ID | Unique user identifier |
| Age | User age |
| Income | Estimated annual income |
| City_Tier | City classification |
| Device_Type | Device category |
| Traffic_Source | Source of website traffic |
| Pages_Viewed | Number of pages viewed |
| Products_Viewed | Number of products viewed |
| Time_On_Site | Session duration |
| Previous_Purchases | Historical purchases |
| Discount_Seen | Discount visibility indicator |
| Browser_Version | Browser version |
| Campaign_Code | Marketing campaign identifier |
| Converted | Target variable |

---

## 🔍 Exploratory Data Analysis (EDA)

The datasets were inspected to:

- Understand feature distributions
- Identify missing values
- Categorize variables into numerical and categorical features
- Explore relationships between user behavior and conversions

---

## ⚙️ Feature Engineering

To improve predictive performance, several new features were created.

### Engagement

```python
Engagement = Pages_Viewed + Products_Viewed
```

Measures overall user interaction with the website.

### Time Per Product

```python
Time_per_Product = Time_On_Site / (Products_Viewed + 1)
```

Represents the average time spent per viewed product.

### Income Per Page

```python
Income_per_Page = Income / (Pages_Viewed + 1)
```

Captures purchasing power relative to browsing activity.

### High Income Flag

```python
High_Income = Income > Median_Income
```

Identifies users with above-average income.

### Returns To Site

```python
Returns_to_Site = Previous_Purchases > 0
```

Indicates whether a user has previously made purchases.

---

## 🧹 Data Preprocessing

### Numerical Features

- Missing values handled using Median Imputation
- Standardized using StandardScaler

### Categorical Features

- Missing values replaced with "missing"
- One-Hot Encoding applied using OneHotEncoder

A ColumnTransformer and Scikit-Learn Pipeline were used to automate preprocessing and ensure consistency across training and prediction stages.

---

## 🤖 Machine Learning Model

### HistGradientBoostingClassifier

The primary model used in this project is:

```python
HistGradientBoostingClassifier(
    random_state=42,
    max_iter=300,
    learning_rate=0.1
)
```

### Why HistGradientBoosting?

- Strong performance on structured tabular data
- Handles non-linear relationships effectively
- Fast training and prediction
- Works well with engineered features
- Requires minimal parameter tuning

---

## 📊 Model Evaluation

### Evaluation Metric

```text
F1 Score
```

The F1 Score balances:

- Precision
- Recall

making it suitable for conversion prediction problems.

### Cross Validation

The model was evaluated using:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Benefits:

- Reliable performance estimation
- Maintains class balance across folds
- Reduces overfitting risk

---

## 🚀 Project Workflow

```text
Load Data
    ↓
Feature Engineering
    ↓
Data Preprocessing
    ↓
Cross Validation
    ↓
Model Training
    ↓
Public Test Evaluation
    ↓
Final Training on Combined Data
    ↓
Private Test Prediction
    ↓
Submission File Generation
```

---

## 📈 Final Training Strategy

1. Train model using train.csv
2. Evaluate using 5-Fold Cross Validation
3. Validate using public_test.csv
4. Combine train.csv and public_test.csv
5. Retrain model on all labeled data
6. Predict on private_test.csv
7. Generate submission.csv

---

## 📄 Submission Format

The final submission file contains:

```text
User_ID,Converted
103001,1
103002,0
103003,1
```

Output file:

```text
submission.csv
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn

---

## 📁 Project Structure

```text
E-Commerce-Conversion-Prediction/
│
├── train.csv
├── public_test.csv
├── private_test.csv
│
├── model.py
├── submission.csv
├── Methodology_Report.pdf
├── README.md
│
└── requirements.txt
```

---

## 📚 Key Learnings

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Classification Modeling
- Gradient Boosting
- Cross Validation
- F1 Score Evaluation
- End-to-End Machine Learning Pipeline

---

## 👩‍💻 Author

**Sakshi Khamkar**

Aspiring Data Analyst | Machine Learning Enthusiast

### Skills

- Python
- SQL
- Power BI
- Machine Learning
- Data Analytics

---

## 🏆 Challenge

Summer Analytics 2026 – Mini Hackathon Week 2

**E-Commerce Conversion Prediction Challenge**

This project demonstrates an end-to-end machine learning workflow for predicting customer conversion behavior using structured tabular data.
