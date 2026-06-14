# OIBSIP - OASIS INFOBYTE Data Science Internship

**Author**: Pratham Bhat  
**Repository**: siem-assistant-isro  
**Program**: OASIS INFOBYTE Summer Internship Program (OIBSIP)  
**Duration**: Data Science Specialization  
**Status**: Project in Progress (Datasets pending upload)

---

## Overview

This repository contains 5 comprehensive Data Science projects completed as part of the OASIS INFOBYTE Summer Internship Program. Each task focuses on different machine learning and data analysis techniques using Python, demonstrating practical skills in supervised learning, classification, regression, and NLP.

---

## Project Structure

```
OIBSIP/
├── Task1/          # Iris Flower Classification
├── Task2/          # Unemployment Analysis
├── Task3/          # Car Price Prediction
├── Task4/          # Email Spam Detection
├── Task5/          # Sales Prediction
└── README.md       # This file
```

---

## Tasks Summary

### **Task 1: Iris Flower Classification** ✅
**Status**: Completed & Tested  
**Location**: `Task1/`

Classify iris flowers into three species using Random Forest Classifier.

- **Dataset**: Built-in sklearn iris dataset (no CSV needed)
- **Model**: Random Forest Classifier (100 estimators)
- **Train-Test Split**: 80-20 (stratified)
- **Metrics**: Accuracy, Confusion Matrix, Feature Importance
- **Visualizations**: 6 PNG plots
- **Key Result**: ~90% Test Accuracy

**Files**:
- `PrathamBhat_Task1.py` - Standalone executable script
- `PrathamBhat_Task1.ipynb` - Jupyter notebook with markdown documentation
- `README.md` - Task-specific documentation

---

### **Task 2: Unemployment Analysis with Python** 📊
**Status**: Code Complete (Dataset Required)  
**Location**: `Task2/`

Analyze unemployment trends across regions and time periods. Compare pre-COVID vs post-COVID unemployment rates.

- **Dataset**: `unemployment.csv` (user-provided from Kaggle)
- **Expected Columns**: Region, Date, Estimated Unemployment Rate
- **Analysis**: Regional comparison, COVID impact, trend analysis
- **Visualizations**: 6 PNG plots including boxplots, heatmaps, trend charts
- **Features**: Boxplot comparison, Top 10 affected regions, Time series analysis

**Files**:
- `PrathamBhat_Task2.py` - Standalone script (requires CSV)
- `PrathamBhat_Task2.ipynb` - Jupyter notebook
- `README.md` - Task documentation

**To Run**:
1. Download `unemployment.csv` from Kaggle
2. Place it in `Task2/` folder
3. Run: `python PrathamBhat_Task2.py`

---

### **Task 3: Car Price Prediction with ML** 🚗
**Status**: Code Complete (Dataset Required)  
**Location**: `Task3/`

Build ML models to predict car selling prices. Compare Linear Regression vs Random Forest.

- **Dataset**: `car_data.csv` (user-provided from Kaggle)
- **Expected Columns**: selling_price, present_price, kms_driven, fuel_type, seller_type, transmission, owner, year
- **Models**: Linear Regression, Random Forest Regressor
- **Metrics**: MAE, R² Score, RMSE, Feature Importance
- **Visualizations**: 6 PNG plots including correlation, model comparison, feature importance

**Files**:
- `PrathamBhat_Task3.py` - Standalone script (requires CSV)
- `PrathamBhat_Task3.ipynb` - Jupyter notebook
- `README.md` - Task documentation

**To Run**:
1. Download `car_data.csv` from Kaggle
2. Place it in `Task3/` folder
3. Run: `python PrathamBhat_Task3.py`

---

### **Task 4: Email Spam Detection with ML** 📧
**Status**: Code Complete (Dataset Required)  
**Location**: `Task4/`

Detect spam emails using TF-IDF vectorization and Multinomial Naive Bayes classifier.

- **Dataset**: `spam.csv` (user-provided from Kaggle)
- **Expected Columns**: label (spam/ham), message (email text)
- **Technique**: TF-IDF vectorization, Multinomial Naive Bayes
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Features**: Text classification, feature extraction, confusion matrix, ROC curve

**Files**:
- `PrathamBhat_Task4.py` - Standalone script (requires CSV)
- `PrathamBhat_Task4.ipynb` - Jupyter notebook
- `README.md` - Task documentation

**To Run**:
1. Download `spam.csv` from Kaggle
2. Place it in `Task4/` folder
3. Run: `python PrathamBhat_Task4.py`

---

### **Task 5: Sales Prediction using Python** 💰
**Status**: Code Complete (Dataset Required)  
**Location**: `Task5/`

Predict product sales using Linear Regression based on advertising spend across TV, Radio, and Newspaper.

- **Dataset**: `advertising.csv` (user-provided from Kaggle)
- **Expected Columns**: TV, Radio, Newspaper, Sales (all in $1000s)
- **Model**: Linear Regression
- **Metrics**: R² Score, MAE, RMSE, Feature Coefficients
- **Visualizations**: 6 PNG plots including channel analysis, actual vs predicted, residuals

**Files**:
- `PrathamBhat_Task5.py` - Standalone script (requires CSV)
- `PrathamBhat_Task5.ipynb` - Jupyter notebook
- `README.md` - Task documentation

**To Run**:
1. Download `advertising.csv` from Kaggle
2. Place it in `Task5/` folder
3. Run: `python PrathamBhat_Task5.py`

---

## Dataset Instructions

### Download and Setup

**Important**: Tasks 2-5 require user-provided datasets from Kaggle. Task 1 uses sklearn's built-in dataset (no download needed).

**Step 1: Download Datasets**
- Task 2: [Unemployment Dataset](https://www.kaggle.com/datasets/unemployment/)
- Task 3: [Car Price Dataset](https://www.kaggle.com/datasets/cardataset/)
- Task 4: [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/)
- Task 5: [Advertising Dataset](https://www.kaggle.com/datasets/sazidthe1/advertising/)

**Step 2: Place CSV Files**
```
OIBSIP/
├── Task2/
│   ├── unemployment.csv         ← Place downloaded CSV here
│   ├── PrathamBhat_Task2.py
│   └── ...
├── Task3/
│   ├── car_data.csv             ← Place downloaded CSV here
│   ├── PrathamBhat_Task3.py
│   └── ...
├── Task4/
│   ├── spam.csv                 ← Place downloaded CSV here
│   ├── PrathamBhat_Task4.py
│   └── ...
└── Task5/
    ├── advertising.csv          ← Place downloaded CSV here
    ├── PrathamBhat_Task5.py
    └── ...
```

### Expected Dataset Filenames
| Task | Filename | Purpose |
|------|----------|---------|
| 1 | N/A (built-in) | Iris classification |
| 2 | `unemployment.csv` | Unemployment analysis |
| 3 | `car_data.csv` | Car price prediction |
| 4 | `spam.csv` | Email spam detection |
| 5 | `advertising.csv` | Sales prediction |

---

## Tech Stack

### Languages & Libraries
- **Python 3.7+**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Text Processing**: scikit-learn (TF-IDF)
- **Notebooks**: Jupyter

### Installation

```bash
# Install required dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter scipy

# Or use requirements file (if created)
pip install -r requirements.txt
```

---

## Usage

### Running Python Scripts
```bash
cd TaskX
python PrathamBhat_TaskX.py
```

**Output**: Console summary + PNG visualization files in the same folder

### Running Jupyter Notebooks
```bash
cd TaskX
jupyter notebook PrathamBhat_TaskX.ipynb
```

---

## Code Features

### Consistent Across All Tasks
✅ **Clean, beginner-friendly code** with professional structure  
✅ **Comprehensive comments** explaining key concepts  
✅ **Column name configuration section** for easy CSV adaptation  
✅ **Automatic PNG export** of all visualizations  
✅ **Detailed console output** with analysis summary  
✅ **Markdown cells in notebooks** with:
   - Objective statements
   - EDA sections
   - Model training details
   - Evaluation results
   - Summary insights

### Visualizations
Each task generates 6 PNG plots saved in its folder:
1. **Distribution/Correlation Analysis**
2. **Feature Relationships**
3. **Statistical Summaries**
4. **Model Performance Metrics**
5. **Prediction/Classification Results**
6. **Feature Importance/Impact Analysis**

---

## Results Summary

| Task | Status | Model | Primary Metric | Output |
|------|--------|-------|-----------------|--------|
| 1 | ✅ Complete | Random Forest | 90% Accuracy | 6 PNGs |
| 2 | 🔄 Pending CSV | Analysis | Visualizations | 6 PNGs |
| 3 | 🔄 Pending CSV | Regression | R² Score | 6 PNGs |
| 4 | 🔄 Pending CSV | Naive Bayes | Accuracy | 6 PNGs |
| 5 | 🔄 Pending CSV | Linear Reg | R² Score | 6 PNGs |

---

## File Organization

Each task follows this structure:

```
TaskX/
├── PrathamBhat_TaskX.py          # Main Python script
├── PrathamBhat_TaskX.ipynb       # Jupyter Notebook
├── README.md                      # Task documentation
├── 01_*.png                       # Visualization plots (generated)
├── 02_*.png
├── 03_*.png
├── 04_*.png
├── 05_*.png
└── 06_*.png
```

---

## Key Learnings

### Task 1: Supervised Learning (Classification)
- Random Forest for multiclass classification
- Train-test split and stratification
- Confusion matrix interpretation
- Feature importance analysis

### Task 2: Data Analysis & Visualization
- Time series analysis
- Boxplot comparisons
- Heatmap generation
- Regional trend identification

### Task 3: Regression Modeling
- Linear Regression vs ensemble methods
- Feature encoding (categorical variables)
- Model comparison (MAE, R²)
- Feature importance in regression

### Task 4: NLP & Text Classification
- TF-IDF vectorization
- Naive Bayes for text classification
- ROC-AUC evaluation
- Text preprocessing

### Task 5: Multivariate Regression
- Multiple linear regression
- Correlation analysis
- Residuals analysis
- Feature coefficient interpretation

---

## How to Complete This Project

1. **Clone/Download Repository**
   ```bash
   git clone <repo-url>
   cd OIBSIP
   ```

2. **Install Dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter scipy
   ```

3. **Download Datasets** (Tasks 2-5 only)
   - Visit each Kaggle link
   - Download CSV files
   - Place in respective Task folders

4. **Run Task 1** (immediately, no CSV needed)
   ```bash
   cd Task1
   python PrathamBhat_Task1.py
   ```

5. **Run Other Tasks** (after uploading CSVs)
   ```bash
   cd Task2
   python PrathamBhat_Task2.py
   # Repeat for Tasks 3, 4, 5
   ```

6. **View Results**
   - Check console output for analysis summary
   - View PNG files for visualizations
   - Open .ipynb files in Jupyter for detailed analysis

---

## Notes

- **Task 1** is self-contained and ready to run immediately
- **Tasks 2-5** require dataset CSVs from Kaggle
- All scripts have a "COLUMN NAME CONFIG" section for easy customization
- PNG files are saved in the same folder as the script
- Notebooks can be run in Jupyter for interactive analysis
- Code is commented for learning purposes

---

## Author

**Pratham Bhat**  
OASIS INFOBYTE Data Science Internship (OIBSIP)  
2024

---

## License

This project is part of the OASIS INFOBYTE internship program.

---

## Next Steps

- [ ] Download and place datasets for Tasks 2-5
- [ ] Run all task scripts to generate visualizations
- [ ] Review PNG outputs and console analysis
- [ ] Explore Jupyter notebooks for detailed explanations
- [ ] Customize column names if your CSV files differ

---

**Status**: Project structure complete. Awaiting dataset uploads for Tasks 2-5.
