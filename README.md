# Elevate-Labs-AI-ML-Internship-Task-1

This repository contains a complete pipeline for cleaning and preparing the Titanic dataset for machine learning. It covers handling missing values, encoding, feature scaling, and outlier removal.

# Task 1: Data Cleaning & Preprocessing

## Objective
Clean and prepare the raw Titanic dataset for machine learning.  
- Handle missing values  
- Encode categorical features  
- Scale numerical features  
- Detect and remove outliers

## Dataset
- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)  
- Alternatively loaded via Seaborn for demonstration.

## Libraries / Tools
- Python 3.x  
- pandas, numpy  
- matplotlib  
- scikit-learn (StandardScaler)

## Steps Performed
1. **Imports & Data Loading**  
   - Loaded Titanic dataset (CSV or Seaborn).
2. **Initial Inspection**  
   - `head()`, `tail()`, `shape`, `info()`, `describe()`.
3. **Unique Values & Missing Counts**  
   - Identified categorical columns.  
   - Counted missing values per column.
4. **Missing Value Handling**  
   - `age` → fill with median.  
   - `embarked` → fill with mode.  
   - `embark_town` → fill with mode, then drop.  
   - Dropped `deck` (too many missing).  
   - Dropped derived columns: `who`, `adult_male`, `alive`, `alone`.
5. **Drop Redundant Columns**  
   - Dropped `embark_town`.  
   - Dropped `class` (duplicate of `pclass`).
6. **Categorical Encoding**  
   - `sex` → label encode (male=0, female=1).  
   - `embarked` → one-hot encode → `embarked_Q`, `embarked_S`.
7. **Feature Engineering**  
   - Created `family_size = sibsp + parch + 1`.  
   - Dropped `sibsp` and `parch`.
8. **Scaling**  
   - Standardized `age` and `fare` to mean≈0, std≈1.
9. **Outlier Removal** (IQR method on scaled `fare`)  
   - Computed Q1, Q3, IQR.  
   - Filtered all rows outside `[Q1 – 1.5·IQR, Q3 + 1.5·IQR]`.  
   - Visualized final boxplot.
10. **Final Preview & Save**  
    - Displayed cleaned DataFrame head & shape.  
    - Saved to `titanic_cleaned.csv` (optional).

## How to Reproduce
1. Clone this repo.  
2. Open `task1_notebook.ipynb` in Jupyter Notebook or JupyterLab.  
3. Run each cell in order.  
4. (Optional) The final cleaned CSV `titanic_cleaned.csv` is created at the end.
