# Employee Attrition Prediction
**Course:** MATH624 — Statistical Learning | Ball State University  
**Author:** Luc Nguyen | December 2025
## Overview
Statistical learning project to predict employee attrition and identify at-risk workforce segments using the IBM HR Analytics dataset (1,470 employees, 31 features).
## Methods
**Unsupervised Learning**
- Principal Component Analysis (PCA) — dimensionality reduction across 23 numeric features
- K-Means Clustering (k=4) — employee segmentation
- Hierarchical Clustering — validation of K-means results

**Supervised Learning (6 models)**
| Model | Accuracy | Sensitivity |
|-------|----------|-------------|
| Logistic Regression | 87.7% | 43.7% |
| LDA | 86.4% | 39.4% |
| QDA | 86.4% | 42.3% |
| k-NN (k=5) | 84.8% | 16.9% |
| Random Forest | 85.7% | 15.5% |
| Gradient Boosting | 87.3% | 36.6% |

**Best model: Logistic Regression** (highest accuracy + sensitivity)
## Key Findings
- **Cluster 3** (new/early-career employees, ~49% of workforce) has the highest attrition rate at **20.9%**
- **OverTime** is the single strongest predictor of attrition (coefficient +1.912, p<0.001)
- **Frequent business travel** increases attrition odds by ~14x
- Higher **job satisfaction**, **environment satisfaction**, and **work-life balance** all significantly reduce attrition risk
## Files
| File | Description |
|------|-------------|
| `LucNguyen_FinalProject.Rmd` | Full R Markdown source code |
| `LucNguyen_FinalProject.html` | Rendered output with all plots |
| `Math624_LucNguyen_Final_Project.pdf` | Written report |
| `HR_Employee_Attrition.csv` | Dataset (IBM HR Analytics) |
## How to Run
```r
# Install required packages
install.packages(c("tidyverse", "corrplot", "factoextra", "caret", 
                   "MASS", "class", "randomForest", "gbm", "pROC"))
# Open LucNguyen_FinalProject.Rmd in RStudio and knit
```
## Tech Stack
R · tidyverse · ggplot2 · caret · randomForest · gbm · factoextra · pROC
