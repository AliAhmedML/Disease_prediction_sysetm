# Disease Prediction System  

**A machine learning-based system for predicting multiple diseases using structured datasets.**  

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)

## 📌 Overview  

This project implements a **multi-disease prediction system** using machine learning. It includes data preprocessing, feature analysis, and model training to handle numerical healthcare datasets effectively.  

---

## 🛠️ Requirements  

- **Python 3.11 or later**  
- Recommended: **MiniConda** for environment management  

---

## ⚙️ Setup  

### Install Python via MiniConda  

1. Download and install [MiniConda](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).  
2. Create a dedicated environment:  

   ```bash  
   conda create -p venv_Disease python==3.11 

## 🛠️ Environment Setup

### Activate the Conda Environment

```bash
conda activate venv_Disease
```

## 🛠️ Installation

### Install Required Packages

Ensure all dependencies are installed by running:

```bash
pip install -r requirements.txt
```

Download and install [CUDA Toolkit 12.5.0](https://developer.nvidia.com/cuda-12-5-0-download-archive).  

> **Note:**  
> All preprocessing, training, and evaluation steps outlined in this guide will be applied to **three different datasets**:  
>
> - Diabetes  
> - Heart Disease  
> - Parkinson’s Disease  
> The same structured workflow is followed for each dataset, including data cleaning, transformation, model training, evaluation, and model saving.  
> Minor adjustments may be applied based on each dataset’s specific characteristics and requirements.

## 🔍 Data Preprocessing Guide

### Essential Preprocessing Steps

1. **Dataset Loading**
   - Load your dataset using appropriate methods
   - Verify the dataset structure and contents

2. **Missing Value Handling**
   - Identify and document missing values
   - Apply either removal or imputation strategies
   - Maintain records of all modifications

3. **Exploratory Visualization**
   - Generate distribution plots for numerical features
   - Create correlation visualizations
   - Examine feature relationships

4. **Outlier Management**
   - Detect outliers using IQR or Z-score methods
   - Handle them if they are likely to affect model performance or data quality
   - For small counts: remove rows with outliers
   - For larger counts: cap outliers using boundary values
   - Document all outlier handling steps clearlys

5. **Forming train and test datasets**
   - make X and Y variables
   - splitting data into train and test sets
   - standarize train data set

6. **Data Transformation**
   - Identify skewness using skewness score or visual inspection (histogram, boxplot)
   - Handle skewness if skewness score > 0.5 or < -0.5 (moderate) or > 1.0 or < -1.0 (strong)
   - For moderate skewness: apply log, square root, or Box-Cox transformation
   - For strong skewness: apply PowerTransformer or Yeo-Johnson transformation

7. **Feature Correlation**
   - Analyze multicollinearity using correlation matrix or Variance Inflation Factor (VIF)
   - Handle multicollinearity if correlation coefficient > 0.8 or VIF > 5
   - For high multicollinearity: remove one of the correlated features or apply dimensionality reduction (e.g., PCA)
   - Document correlation findings and actions taken

8. **Class Balance**
   - Evaluate target class distribution using value counts or class ratios
   - Handle imbalance if the minority class represents less than 20% of the total samples
   - For moderate imbalance: use class weighting during model training
   - For severe imbalance: apply resampling techniques such as SMOTE, ADASYN, or Random Over/Under Sampling

9. **Data Standarization**
   - Apply standardization when features have different units or scales  
   - Handle features with non-normal distributions carefully: standardize after skewness correction  
   - Use `StandardScaler` for models sensitive to feature scales (e.g., SVM, KNN, Logistic Regression, PCA)
   - Standardize after splitting into train-test sets to prevent data leakage  
   - Verify the standardized data by checking mean ≈ 0 and standard deviation ≈ 1

10. **Transforming Train and Test Data Sets into GPU**
    - Convert train and test data sets into GPU-supported data structures
    - Use libraries such as cuDF and cuML for GPU acceleration
    - Ensure compatibility between data format and GPU models

## 🎯 Training and Evaluating

1. **Model Training**
   - Train the following classifiers:
     - KNeighborsClassifier
     - LogisticRegression
     - RandomForestClassifier
     - SVC
     - XGBClassifier
   - Ensure all models are trained on the prepared and standardized dataset.

2. **Model Evaluation**
   - Evaluate each model using the following metrics:
     - F1 Score
     - Accuracy Score
     - ROC AUC Score
     - Log Loss
   - Document and compare performance across all metrics.

3. **Model Selection and Saving**
   - Select the model with the best performance according to project requirements.
   - Save the chosen model using appropriate serialization methods (pickle).
   - Prepare the saved model for integration into the prediction system.
