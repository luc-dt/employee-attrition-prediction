# Employee Attrition Prediction

**Course:** MATH624 — Statistical Learning | Ball State University  
**Author:** Luc Nguyen | December 2025

## Overview

This project analyzes employee attrition using the IBM HR Analytics dataset. The goal is to predict which employees are more likely to leave and identify key business factors that influence attrition.

The project combines exploratory data analysis, employee segmentation, supervised machine learning, and business recommendations. The original analysis was completed in R Markdown, and additional Python and SQL files were added to make the project more relevant for data science job applications.

## Business Problem

Employee turnover can increase hiring costs, reduce productivity, and create knowledge loss. This project helps HR and business stakeholders answer three key questions:

1. Which employees are most likely to leave?
2. Which employee groups have the highest attrition risk?
3. What actions can help reduce employee attrition?

## Dataset

The project uses the IBM HR Analytics Employee Attrition dataset.

- **Rows:** 1,470 employees
- **Features:** 31 employee attributes
- **Target variable:** `Attrition`

Example features include:

- Age
- Department
- Job Role
- Monthly Income
- OverTime
- Business Travel
- Job Satisfaction
- Environment Satisfaction
- Work-Life Balance
- Years at Company

## Methods

### Exploratory Data Analysis

The analysis explores:

- Attrition class imbalance
- Missing values
- Summary statistics
- Numeric feature distributions
- Relationships between employee attributes and attrition
- Correlations among key variables

### Unsupervised Learning

The project uses unsupervised learning to identify employee segments:

- Principal Component Analysis, or PCA
- K-Means clustering
- Hierarchical clustering

These methods help identify workforce groups with different attrition patterns.

### Supervised Learning

The project compares six classification models:

| Model | Accuracy | Sensitivity |
| --- | ---: | ---: |
| Logistic Regression | 87.7% | 43.7% |
| LDA | 86.4% | 39.4% |
| QDA | 86.4% | 42.3% |
| k-NN | 84.8% | 16.9% |
| Random Forest | 85.7% | 15.5% |
| Gradient Boosting | 87.3% | 36.6% |

The best model was **Logistic Regression** because it provided strong accuracy, the best sensitivity among the tested models, and clear interpretability for business stakeholders.

## Key Findings

- **Cluster 3**, representing newer and early-career employees, had the highest attrition rate at **20.9%**.
- **OverTime** was the strongest predictor of attrition.
- **Frequent business travel** increased attrition risk.
- Higher **job satisfaction**, **environment satisfaction**, and **work-life balance** reduced attrition risk.
- Since only **16.1%** of employees left, sensitivity/recall is important and accuracy alone is not enough.

## Business Recommendations

Based on the analysis, HR teams should:

- Monitor overtime and workload distribution.
- Improve onboarding and mentoring for early-tenure employees.
- Track job satisfaction and environment satisfaction through regular surveys.
- Reduce unnecessary business travel when possible.
- Offer flexible work options for employees with long commutes.
- Monitor model performance over time if the model is used in practice.

## Python Machine Learning Pipeline

The file `src/attrition_model.py` adds a Python/scikit-learn workflow that complements the original R analysis.

The Python pipeline:

- Loads the HR attrition dataset
- Removes non-informative columns
- Encodes categorical variables
- Scales numeric variables
- Trains multiple classification models
- Evaluates attrition-focused metrics
- Saves the best model pipeline with `joblib`

Models included in the Python pipeline:

- Logistic Regression
- Random Forest
- Gradient Boosting

## SQL Business Analysis

The file `sql/attrition_business_questions.sql` includes SQL examples for answering stakeholder-focused business questions.

Example questions include:

- Which overtime groups have the highest attrition rate?
- Which departments have high early-tenure attrition?
- How do business travel and commute distance relate to attrition risk?

These SQL examples show how the project can support business analysis before or after machine learning modeling.

## Project Files

| File | Description |
| --- | --- |
| `LucNguyen_FinalProject.Rmd` | Original R Markdown analysis |
| `LucNguyen_FinalProject.html` | Rendered HTML report |
| `Math624_LucNguyen_Final_Project.pdf` | Final written report |
| `HR_Employee_Attrition.csv` | Employee attrition dataset |
| `src/attrition_model.py` | Python machine learning pipeline |
| `sql/attrition_business_questions.sql` | SQL analysis examples |
| `requirements.txt` | Python package requirements |
| `.gitignore` | Git ignore rules for private/local files |

## How to Run the R Analysis

Open `LucNguyen_FinalProject.Rmd` in RStudio and knit the file.

Install the required R packages:

```r
install.packages(c("tidyverse", "corrplot", "factoextra", "caret",
                   "MASS", "class", "randomForest", "gbm", "pROC"))
```

## How to Run the Python Pipeline

Create a virtual environment:

### Windows Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python src/attrition_model.py --data HR_Employee_Attrition.csv
```

The script saves the best trained model pipeline to:

```text
models/attrition_pipeline.joblib
```

## Tech Stack

- R
- R Markdown
- tidyverse
- ggplot2
- caret
- randomForest
- gbm
- Python
- pandas
- scikit-learn
- SQL

## Project Value

This project demonstrates the ability to:

- Translate a business problem into a data science workflow
- Clean and explore structured HR data
- Build and compare machine learning models
- Evaluate models using metrics relevant to an imbalanced classification problem
- Explain results clearly to business stakeholders
- Use R, Python, and SQL in one portfolio project