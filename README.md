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
   - Detect outliers using statistical methods
   - Apply clinically appropriate treatments
   - Document outlier handling procedures

5. **Forming train and test datasets**
   - make X and Y variables
   - splitting data into train and test sets
   - standarize train data set

6. **Data Transformation**
   - Identify and address feature skewness
   - Apply necessary normalization techniques
   - Verify transformations visually

7. **Feature Correlation**
   - Analyze multicollinearity between features
   - Plan dimensionality reduction if needed
   - Document correlation findings

8. **Class Balance**
   - Evaluate target class distribution
   - Address severe imbalances if present
   - Choose appropriate balancing methods

9. **Transforming Train and Test Data Sets into GPU**
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
