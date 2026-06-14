# Task 5: Sales Prediction using Python

## Objective
Build a Linear Regression model to predict product sales based on advertising spend across different channels (TV, Radio, Newspaper). Analyze the impact of each advertising channel on sales and create visualizations showing relationships and model performance.

## Dataset
- **Source**: Kaggle - Advertising Dataset (user-provided CSV)
- **Expected Filename**: `advertising.csv`
- **Expected Columns**:
  - TV (advertising spend in TV, in $1000s)
  - Radio (advertising spend in Radio, in $1000s)
  - Newspaper (advertising spend in Newspaper, in $1000s)
  - Sales (product sales, in $1000s)
- **Note**: This is a user-provided dataset. Place the CSV file in this folder before running the script.

## Approach
1. **Data Loading & Preprocessing**:
   - Load advertising data from CSV
   - Remove any unnamed index columns
   - Check for missing values

2. **Exploratory Data Analysis**:
   - Correlation analysis between channels and sales
   - Distribution analysis for all variables
   - Relationship visualization (scatter plots)

3. **Model Development**:
   - Linear Regression model training
   - Feature: TV, Radio, Newspaper spend
   - Target: Sales

4. **Evaluation Metrics**:
   - R² Score (coefficient of determination)
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Feature coefficients (impact analysis)

5. **Visualizations**:
   - Correlation heatmap
   - Individual channel analysis (scatter + regression line)
   - Distribution analysis
   - Actual vs Predicted plot
   - Residuals analysis
   - Feature coefficients visualization

## Results
| Metric | Value |
|--------|-------|
| R² Score | Dataset dependent |
| MAE | Dataset dependent |
| RMSE | Dataset dependent |
| Model Equation | Y = a + b₁X₁ + b₂X₂ + b₃X₃ |

*Note: Run the script to see actual results and model coefficients*

## Files
- `PrathamBhat_Task5.py` - Main Python script (generates 6 PNG plots)
- `PrathamBhat_Task5.ipynb` - Jupyter notebook with documentation
- `README.md` - This file

## Visualizations Generated
1. `01_Correlation_Heatmap.png` - Feature correlation matrix
2. `02_Channel_Analysis.png` - Sales vs TV/Radio/Newspaper with regression lines
3. `03_Distribution_Analysis.png` - Distributions of all variables
4. `04_Actual_vs_Predicted.png` - Model prediction accuracy scatter plot
5. `05_Residuals_Analysis.png` - Residuals visualization and analysis
6. `06_Feature_Coefficients.png` - Impact of each advertising channel on sales

## How to Run

### Prerequisites
1. Download the advertising dataset from Kaggle
2. Place the CSV file in this folder as `advertising.csv`
3. Verify column names match your CSV (or edit the COLUMN NAME CONFIG section)

### Option 1: Run Python Script
```bash
cd Task5
python PrathamBhat_Task5.py
```

**Output**: Console model analysis + 6 PNG visualization files

### Option 2: Run Jupyter Notebook
```bash
cd Task5
jupyter notebook PrathamBhat_Task5.ipynb
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
TV_COLUMN = 'TV'
RADIO_COLUMN = 'Radio'
NEWSPAPER_COLUMN = 'Newspaper'
SALES_COLUMN = 'Sales'
```

## Interpretation of Results

### Model Equation
The Linear Regression model produces an equation: 
**Sales = Intercept + (TV_Coef × TV) + (Radio_Coef × Radio) + (Newspaper_Coef × Newspaper)**

### Feature Coefficients Meaning
- Higher coefficient = stronger impact on sales
- Positive coefficient = increase in spend → increase in sales
- Magnitude shows relative importance of each channel

### R² Score
- Ranges from 0 to 1
- Higher value (closer to 1) = better model fit
- Indicates % of sales variance explained by the model

### Residuals Analysis
- Should be randomly distributed around 0
- Shows prediction errors
- Helps identify model assumptions violations

## Key Findings
- Analyzes multi-feature regression modeling
- Compares relative advertising channel effectiveness
- Demonstrates Linear Regression assumptions and limitations
- Shows how to interpret model coefficients

## Author
**Pratham Bhat** - OASIS INFOBYTE Data Science Internship (OIBSIP)
