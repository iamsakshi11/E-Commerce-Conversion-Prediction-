E-Commerce Conversion Prediction Challenge
Summer Analytics 2026 – Mini Hackathon Week 2
Project Overview
This project was developed as part of the Summer Analytics 2026 Mini Hackathon – Week 2.
The objective of the challenge was to predict whether a user would convert based on demographic information, browsing behavior, engagement metrics, and marketing-related features.
The target variable is:
•	Converted = 1 → User Converted
•	Converted = 0 → User Did Not Convert
The solution uses feature engineering, data preprocessing, cross-validation, and a Gradient Boosting Machine Learning model to generate predictions for unseen users.
________________________________________
Problem Statement
Given user-level e-commerce data, build a classification model that predicts whether a customer will convert.
The challenge evaluates submissions using the F1 Score, which balances Precision and Recall.
________________________________________
Dataset Description
The challenge provided the following datasets:
File	Description
train.csv	Training dataset containing features and target labels
public_test.csv	Validation dataset containing features and target labels
private_test.csv	Unseen dataset used for final prediction
sample_submission.csv	Required submission format
Features
Feature	Description
User_ID	Unique user identifier
Age	User age
Income	Estimated annual income
City_Tier	City classification
Device_Type	Device category
Traffic_Source	Source of website traffic
Pages_Viewed	Number of pages viewed
Products_Viewed	Number of products viewed
Time_On_Site	Session duration
Previous_Purchases	Historical purchases
Discount_Seen	Discount visibility indicator
Browser_Version	Browser version
Campaign_Code	Marketing campaign identifier
Converted	Target variable
________________________________________
Project Workflow
1. Data Loading
The datasets were loaded using Pandas and inspected to understand the available features and data types.
2. Feature Engineering
To improve model performance, additional behavioral features were created.
Engagement
Measures total website interaction.
Engagement = Pages_Viewed + Products_Viewed
Time per Product
Average time spent per viewed product.
Time_per_Product = Time_On_Site / (Products_Viewed + 1)
Income per Page
Income relative to browsing activity.
Income_per_Page = Income / (Pages_Viewed + 1)
High Income Flag
High_Income = Income > Median Income
Returns to Site
Returns_to_Site = Previous_Purchases > 0
These engineered features help capture user engagement and purchasing intent.
________________________________________
Data Preprocessing
Numerical Features
•	Missing values handled using Median Imputation
•	Standardized using StandardScaler
Categorical Features
•	Missing values replaced with "missing"
•	One-Hot Encoding applied using OneHotEncoder
A Scikit-Learn Pipeline and ColumnTransformer were used to automate preprocessing and ensure consistent transformations during training and prediction.
________________________________________
Machine Learning Model
HistGradientBoostingClassifier
The primary model used in this project is:
HistGradientBoostingClassifier(
    random_state=42,
    max_iter=300,
    learning_rate=0.1
)
Why This Model?
•	Performs exceptionally well on tabular datasets
•	Handles complex relationships between variables
•	Fast training and prediction
•	Strong predictive performance with minimal tuning
•	Suitable for structured business and customer analytics problems
________________________________________
Model Evaluation
The competition evaluation metric is:
F1 Score
F1 Score was chosen because it balances:
•	Precision
•	Recall
This makes it suitable for classification problems where both false positives and false negatives are important.
Cross Validation
The model was evaluated using:
StratifiedKFold(n_splits=5)
Benefits:
•	More reliable performance estimation
•	Preserves class distribution in every fold
•	Reduces overfitting risk
The trained model was also evaluated using the provided public test dataset before generating final predictions.
________________________________________
Final Training Strategy
1.	Train model on train.csv
2.	Evaluate using 5-Fold Cross Validation
3.	Validate using public_test.csv
4.	Combine train.csv and public_test.csv
5.	Retrain the final model using all available labeled data
6.	Generate predictions for private_test.csv
________________________________________
Submission Format
The final predictions were exported as:
submission.csv
Required format:
User_ID,Converted
103001,1
103002,0
103003,1
________________________________________
Technologies Used
•	Python
•	Pandas
•	Scikit-Learn
•	NumPy
________________________________________
Project Structure
├── train.csv
├── public_test.csv
├── private_test.csv
├── sample_submission.csv
├── model.py
├── submission.csv
├── Methodology_Report.pdf
└── README.md
________________________________________
Key Learnings
•	Exploratory Data Analysis (EDA)
•	Feature Engineering
•	Data Preprocessing
•	Classification Modeling
•	Cross Validation
•	F1 Score Evaluation
•	Machine Learning Pipelines
•	Customer Conversion Prediction
________________________________________

Author
Sakshi Khamkar
Aspiring Data Analyst | Machine Learning Enthusiast
Skills:
•	Python
•	SQL
•	Power BI
•	Machine Learning
•	Data Analytics
This project was completed as part of the Summer Analytics 2026 Mini Hackathon and demonstrates an end-to-end machine learning workflow for solving a real-world customer conversion prediction problem.

