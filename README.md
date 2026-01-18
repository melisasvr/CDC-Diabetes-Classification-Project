# 🏥 CDC Diabetes Classification Project

- A comprehensive machine learning project demonstrating how data preprocessing, feature scaling, and hyperparameter tuning improve predictive performance for diabetes classification.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Visualizations](#visualizations)
- [Key Insights](#key-insights)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🎯 Overview

- This project builds and compares multiple classification models to predict diabetes risk based on health indicators and lifestyle factors. The analysis demonstrates:

- **Data Preprocessing**: Handling missing values, duplicates, and data splits
- **Feature Scaling**: Comparing StandardScaler, MinMaxScaler, and RobustScaler
- **Model Comparison**: Evaluating Logistic Regression, Random Forest, and Gradient Boosting
- **Hyperparameter Tuning**: Using GridSearchCV and RandomizedSearchCV
- **Performance Evaluation**: Multiple metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- **Comprehensive Visualizations**: 11 different charts and graphs

---

## 📊 Dataset

### CDC Diabetes Health Indicators Dataset

**Source**: [Kaggle - Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

**Description**: 
- Based on the CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2015
- Contains health indicators and lifestyle factors for predicting diabetes
- **253,680 survey responses** from adults across the United States
- **21 feature variables** + 1 target variable

### Features Include:
- **Health Indicators**: High Blood Pressure, High Cholesterol, BMI, Stroke History
- **Lifestyle Factors**: Physical Activity, Smoking, Alcohol Consumption, Diet
- **Healthcare Access**: Insurance Coverage, Cost Barriers
- **Demographics**: Age, Sex, Education, Income
- **Self-Reported Health**: General Health Status, Mental Health, Physical Health

### Target Variable:
- `Diabetes_binary`: 0 = No Diabetes, 1 = Diabetes

---

## ✨ Features

### 1️⃣ **Data Loading**
- Automatic download using `kagglehub` (if installed)
- Manual local file loading
- Fallback to realistic sample data for demonstration
- Validates dataset format and column names

### 2️⃣ **Data Preprocessing**
- Missing value detection and handling
- Duplicate row removal
- Stratified train-test split (80/20)
- Feature-target separation

### 3️⃣ **Feature Scaling Comparison**
Tests 4 different scaling methods:
- No Scaling (baseline)
- StandardScaler (mean=0, std=1)
- MinMaxScaler (range 0-1)
- RobustScaler (median-based, outlier-resistant)

### 4️⃣ **Model Training**

**Baseline Models** (default parameters):
- Logistic Regression
- Random Forest
- Gradient Boosting

**Tuned Models** (optimized hyperparameters):
- GridSearchCV for Logistic Regression
- RandomizedSearchCV for Random Forest and Gradient Boosting

### 5️⃣ **Evaluation Metrics**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

### 6️⃣ **Visualizations** (11 Total)
1. Class Distribution (Pie + Bar Chart)
2. Feature Distributions (Histograms)
3. Scaling Methods Comparison
4. Baseline Model Performance
5. ROC Curves - Baseline Models
6. Tuned Model Performance
7. ROC Curves - Tuned Models
8. Baseline vs Tuned Comparison
9. Confusion Matrix Heatmap
10. Feature Importance
11. Complete Performance Summary

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Optional (for automatic dataset download)

```bash
pip install kagglehub
```

### Clone or Download

```bash
# Clone the repository (if using Git)
git clone <your-repo-url>
cd diabetes-classification

# Or simply download the files to a folder
```

---

## 🚀 Usage

### Option 1: Automatic Download (Recommended)

```bash
# Install kagglehub
pip install kagglehub

# Run the script
python diabetes_classification.py
```

The script will automatically download the dataset and run the analysis.

### Option 2: Manual Download

1. **Download the dataset**:
   - Go to [Kaggle Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
   - Sign in (free account required)
   - Click "Download" and extract the ZIP file

2. **Place the CSV file**:
   - Copy `diabetes_binary_health_indicators_BRFSS2015.csv` to the project folder

3. **Run the script**:
   ```bash
   python diabetes_classification.py
   ```

### Option 3: Use Sample Data

Simply run the script without downloading the dataset:
```bash
python diabetes_classification.py
```

The code will use realistic sample data for demonstration (results won't be as meaningful).

---

## 📁 Project Structure

```
diabetes-classification/
│
├── diabetes_classification.py    # Main analysis script
├── README.md                      # This file
├── requirements.txt               # Python dependencies (optional)
│
└── diabetes_binary_health_indicators_BRFSS2015.csv  # Dataset (after download)
```

---

## 📈 Results

### Expected Performance (with Real Data)

**Scaling Comparison**:
- All scalers perform similarly (ROC-AUC: 0.80-0.81)
- StandardScaler or RobustScaler typically best

**Model Performance**:
- Logistic Regression: ROC-AUC ~0.80
- Random Forest: ROC-AUC ~0.81
- Gradient Boosting: ROC-AUC ~0.81-0.82

**Improvement from Tuning**:
- Average improvement: 0.5-2% in ROC-AUC score
- Best model typically: Gradient Boosting

**Training Time** (with 183K samples):
- Logistic Regression tuning: ~30 seconds
- Random Forest tuning: ~2-5 minutes
- Gradient Boosting tuning: ~2-5 minutes
- **Total runtime: ~5-10 minutes**

### Important Note on Class Imbalance

The dataset has **class imbalance**:
- ~84% No Diabetes (213,703 cases)
- ~16% Diabetes (39,977 cases)

This is realistic (reflects real population) but can lead to:
- High accuracy (83-84%) but moderate recall for diabetes class
- Models may slightly favor predicting "No Diabetes"
- ROC-AUC is a better metric than accuracy for this dataset
- Precision and Recall show the true prediction quality for diabetes detection

---

## 📊 Visualizations

The script generates **11 visualizations**:

### Data Exploration
1. **Class Distribution**: Shows the imbalance between diabetes/non-diabetes cases
2. **Feature Distributions**: Compares key features between groups

### Scaling Analysis
3. **Scaling Methods Comparison**: Bar chart comparing all scaling approaches

### Model Performance
4. **Baseline Performance**: All metrics for non-tuned models
5. **Baseline ROC Curves**: Visual comparison of model discrimination
6. **Tuned Performance**: All metrics after hyperparameter optimization
7. **Tuned ROC Curves**: Improved model discrimination

### Comparison Analysis
8. **Baseline vs Tuned**: Side-by-side comparison showing improvements
9. **Confusion Matrix**: True/False Positives and Negatives for best model
10. **Feature Importance**: Most influential health indicators
11. **Complete Summary**: Comprehensive performance overview

All graphs display automatically and can be saved manually when they appear.

---

## 🔑 Key Insights

### 1. Feature Scaling Matters
- Standardization (StandardScaler) generally performs best
- Critical for distance-based algorithms
- Minimal impact on tree-based models

### 2. Hyperparameter Tuning Improves Performance
- Consistent 2-5% improvement in ROC-AUC
- More significant for complex models (Random Forest, Gradient Boosting)
- GridSearchCV effective for small parameter spaces
- RandomizedSearchCV better for larger parameter spaces

### 3. Model Selection
- Tree-based models (Random Forest, Gradient Boosting) typically outperform Logistic Regression
- Gradient Boosting often achieves best performance
- Logistic Regression provides good interpretability

### 4. Important Health Indicators
Top predictors typically include:
- General Health Status
- High Blood Pressure
- BMI (Body Mass Index)
- Age
- High Cholesterol

### 5. Class Imbalance Challenge
- Dataset reflects real-world diabetes prevalence (~15%)
- Accuracy can be misleading
- ROC-AUC is more reliable metric
- Precision-Recall trade-off is important

---

## ⚠️ Limitations

### 1. Class Imbalance
- Dataset has 85% non-diabetes, 15% diabetes cases
- Models may struggle to identify diabetes cases
- High accuracy doesn't mean good diabetes detection

### 2. Sample Data Performance
- If using sample data (not real CDC data), results are for demonstration only
- Sample data won't show realistic performance
- Always use real dataset for meaningful analysis

### 3. Computational Requirements
- Real dataset (253K samples, ~184K after deduplication) requires decent computing power
- Hyperparameter tuning takes approximately 5-10 minutes total
- Recommended: Multi-core CPU (uses all cores with n_jobs=-1)
- Memory: At least 4GB RAM recommended
- SVM removed due to excessive training time (would take 30+ minutes)

### 4. Model Interpretability
- Random Forest and Gradient Boosting are "black box" models
- Feature importance helps but doesn't show exact relationships
- Logistic Regression more interpretable but lower performance

---

## 🚀 Future Improvements

### Potential Enhancements:

1. **Handle Class Imbalance**:
   - Implement SMOTE (Synthetic Minority Oversampling)
   - Add class weights to models
   - Try threshold adjustment

2. **Add More Models**:
   - XGBoost
   - LightGBM
   - Neural Networks

3. **Feature Engineering**:
   - Create interaction features (e.g., BMI × Age)
   - Polynomial features
   - Feature selection techniques

4. **Cross-Validation**:
   - Stratified K-Fold for more robust evaluation
   - Learning curves to detect overfitting

5. **Ensemble Methods**:
   - Voting Classifier
   - Stacking
   - Blending

6. **Explainability**:
   - SHAP values
   - LIME explanations
   - Partial Dependence Plots

7. **Deployment**:
   - Save best model with joblib/pickle
   - Create web interface (Streamlit/Flask)
   - API for predictions

8. **Save Outputs**:
   - Export all visualizations as PNG/PDF
   - Save results to CSV
   - Generate HTML report

---

## 📝 Technical Details

### Hyperparameter Spaces (Optimized for Speed)

**Logistic Regression**:
- C: [0.001, 0.01, 0.1, 1, 10, 100]
- Penalty: ['l1', 'l2']
- Solver: ['liblinear', 'saga']
- Method: GridSearchCV (exhaustive search)

**Random Forest**:
- n_estimators: [50, 100]
- max_depth: [10, 20]
- min_samples_split: [5, 10]
- min_samples_leaf: [2, 4]
- Method: RandomizedSearchCV (6 iterations)

**Gradient Boosting**:
- n_estimators: [50, 100]
- learning_rate: [0.05, 0.1]
- max_depth: [3, 5]
- subsample: [0.8, 1.0]
- Method: RandomizedSearchCV (6 iterations)

**Note**: Hyperparameter spaces are optimized to balance performance with training time on large datasets (183K+ samples).

### Train-Test Split
- 80% Training (~183,769 samples after removing duplicates)
- 20% Testing (~45,943 samples)
- Stratified to maintain class distribution
- ~24,000 duplicate rows removed from original dataset

---

## 🤝 Contributing
- Contributions are welcome! Some ideas:
- Add more models or techniques
- Improve visualizations
- Add feature engineering
- Implement class imbalance solutions
- Add model explainability
- Create interactive dashboard

---

## 📄 License

- This project is for educational purposes. 

**Dataset License**: CDC BRFSS data is public domain.

**Code**: Feel free to use and modify for educational and research purposes.

---

## 📧 Contact
- For questions, suggestions, or issues:
- Create an issue in the repository

---

## 🙏 Acknowledgments

- **CDC BRFSS**: For providing the public health data
- **Kaggle**: For hosting the dataset
- **Scikit-learn**: For machine learning tools
- **Python Community**: For excellent data science libraries

---

## 📚 References

1. CDC Behavioral Risk Factor Surveillance System (BRFSS): https://www.cdc.gov/brfss/
2. Kaggle Dataset: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
3. Scikit-learn Documentation: https://scikit-learn.org/
4. CDC Diabetes Statistics: https://www.cdc.gov/diabetes/data/

---

## ⭐ Quick Start Summary

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub

# 2. Run the script
python diabetes_classification.py

# 3. View results
# - Console output shows metrics
# - 11 graphs display automatically
# - Close each graph to see the next one
```

**That's it! Enjoy exploring diabetes prediction with machine learning! 🎉**
