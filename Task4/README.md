# Task 4: Email Spam Detection with ML

## Objective
Build a machine learning model to detect spam emails using Natural Language Processing and Naive Bayes classification. Classify emails as spam or legitimate (ham) using TF-IDF vectorization and achieve high accuracy with interpretable results.

## Dataset
- **Source**: Kaggle - SMS Spam Collection Dataset (user-provided CSV)
- **Expected Filename**: `spam.csv`
- **Expected Columns**:
  - label (or v1) - "spam" or "ham"
  - message (or v2) - email/SMS text content
- **Note**: This is a user-provided dataset. Place the CSV file in this folder before running the script.

## Approach
1. **Data Loading & Preprocessing**:
   - Load spam dataset from CSV
   - Handle categorical labels (spam/ham)
   - Remove missing values
   - Calculate message statistics (length, word count)

2. **Feature Engineering**:
   - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
   - Extract top 5000 features
   - Remove English stop words

3. **Exploratory Data Analysis**:
   - Class distribution analysis
   - Message length comparison
   - Word count analysis
   - Top spam vs ham indicators

4. **Model Training**:
   - Multinomial Naive Bayes classifier
   - 80-20 train-test split

5. **Evaluation Metrics**:
   - Accuracy Score
   - Precision, Recall, F1-Score
   - Confusion Matrix
   - ROC-AUC Curve
   - Feature importance (top spam/ham keywords)

6. **Visualizations**:
   - Data distribution plots
   - Confusion matrix heatmap
   - Performance metrics bar chart
   - Top spam and ham indicators
   - ROC curve

## Results
| Metric | Value |
|--------|-------|
| Accuracy | Dataset dependent |
| Precision | Dataset dependent |
| Recall | Dataset dependent |
| F1-Score | Dataset dependent |
| ROC-AUC | Dataset dependent |

*Note: Run the script to see actual results and visualizations*

## Files
- `PrathamBhat_Task4.py` - Main Python script (generates 6 PNG plots)
- `PrathamBhat_Task4.ipynb` - Jupyter notebook with documentation
- `README.md` - This file

## Visualizations Generated
1. `01_Data_Distribution.png` - Email type distribution and message statistics
2. `02_Feature_Comparison.png` - Message length and word count comparison
3. `03_Confusion_Matrix.png` - Classification confusion matrix
4. `04_Performance_Metrics.png` - Accuracy, Precision, Recall, F1-Score
5. `05_Top_Features.png` - Top spam and ham indicator words
6. `06_ROC_Curve.png` - ROC curve and AUC score

## How to Run

### Prerequisites
1. Download the spam dataset from Kaggle
2. Place the CSV file in this folder as `spam.csv`
3. Verify column names match your CSV (or edit the COLUMN NAME CONFIG section)

### Option 1: Run Python Script
```bash
cd Task4
python PrathamBhat_Task4.py
```

**Output**: Console model performance + 6 PNG visualization files

### Option 2: Run Jupyter Notebook
```bash
cd Task4
jupyter notebook PrathamBhat_Task4.ipynb
```

### Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy

### Install Dependencies (if needed)
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

## Column Name Configuration
If your CSV has different column names, edit these lines in the Python script:
```python
LABEL_COLUMN = 'label'  # or 'v1' - column with spam/ham labels
MESSAGE_COLUMN = 'message'  # or 'v2' - column with email text
```

## Key Findings & Interpretation
- Demonstrates text classification with TF-IDF and Naive Bayes
- Identifies characteristic spam vs ham keywords
- High-accuracy spam detection possible with simple NLP
- ROC-AUC shows model's ability to distinguish classes

## Author
**Pratham Bhat** - OASIS INFOBYTE Data Science Internship (OIBSIP)
