# Task 3: Car Price Prediction with ML

## Objective
Build machine learning models to predict car selling prices based on vehicle attributes. Compare Linear Regression and Random Forest Regressor to identify the best predictive model. Analyze feature importance and model performance.

## Dataset
- **Source**: Kaggle - Car Price Prediction Dataset (user-provided CSV)
- **Expected Filename**: `car_data.csv`
- **Expected Columns**:
  - selling_price (target variable)
  - present_price
  - kms_driven
  - fuel_type
  - seller_type
  - transmission
  - owner
  - year
- **Note**: This is a user-provided dataset. Place the CSV file in this folder before running the script.

## Approach
1. **Data Loading & Preprocessing**:
   - Load car data from CSV
   - Handle missing values
   - Encode categorical variables (fuel type, seller type, transmission)

2. **Exploratory Data Analysis**:
   - Correlation analysis between features and price
   - Feature distribution plots
   - Scatter plots showing relationships

3. **Model Development**:
   - Linear Regression model
   - Random Forest Regressor model
   - 80-20 train-test split

4. **Evaluation Metrics**:
   - Mean Absolute Error (MAE)
   - R² Score
   - Root Mean Squared Error (RMSE)
   - Actual vs Predicted plots
   - Feature Importance ranking

5. **Visualizations**:
   - Correlation heatmap
   - Feature analysis scatter plots
   - Model comparison charts
   - Actual vs Predicted plots
   - Feature importance ranking

## Results
| Metric | Linear Regression | Random Forest |
|--------|-------------------|---------------|
| MAE | Dataset dependent | Dataset dependent |
| R² Score | Dataset dependent | Dataset dependent |
| RMSE | Dataset dependent | Dataset dependent |

*Note: Run the script to see actual results and model comparisons*

## Files
- `PrathamBhat_Task3.py` - Main Python script (generates 6 PNG plots)
- `PrathamBhat_Task3.ipynb` - Jupyter notebook with documentation
- `README.md` - This file

## Visualizations Generated
1. `01_Correlation_Heatmap.png` - Feature correlation matrix
2. `02_Feature_Analysis.png` - Scatter plots of key features vs price
3. `03_Price_Distribution.png` - Price distribution and boxplot
4. `04_Model_Comparison.png` - Linear Regression vs Random Forest metrics
5. `05_Actual_vs_Predicted.png` - Prediction accuracy for both models
6. `06_Feature_Importance.png` - Feature importance ranking (Random Forest)

## How to Run

### Prerequisites
1. Download the car price dataset from Kaggle
2. Place the CSV file in this folder as `car_data.csv`
3. Verify column names match your CSV (or edit the COLUMN NAME CONFIG section)

### Option 1: Run Python Script
```bash
cd Task3
python PrathamBhat_Task3.py
```

**Output**: Console model performance + 6 PNG visualization files

### Option 2: Run Jupyter Notebook
```bash
cd Task3
jupyter notebook PrathamBhat_Task3.ipynb
```

### Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### Install Dependencies (if needed)
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Column Name Configuration
If your CSV has different column names, edit these lines in the Python script:
```python
SELLING_PRICE_COLUMN = 'selling_price'
PRESENT_PRICE_COLUMN = 'present_price'
KMS_DRIVEN_COLUMN = 'kms_driven'
FUEL_TYPE_COLUMN = 'fuel_type'
SELLER_TYPE_COLUMN = 'seller_type'
TRANSMISSION_COLUMN = 'transmission'
OWNER_COLUMN = 'owner'
YEAR_COLUMN = 'year'
```

## Key Findings & Interpretation
- Compares Linear vs Random Forest regression approaches
- Identifies most influential features in price prediction
- Evaluates model generalization capability
- Highlights advantages of ensemble methods

## Author
**Pratham Bhat** - OASIS INFOBYTE Data Science Internship (OIBSIP)
