# Task 2: Unemployment Analysis with Python

## Objective
Analyze unemployment trends across different regions and time periods. Compare unemployment rates before, during, and after the COVID-19 pandemic. Identify the most severely affected regions and track temporal changes in employment patterns.

## Dataset
- **Source**: Kaggle - Unemployment Dataset (user-provided CSV)
- **Expected Filename**: `unemployment.csv`
- **Expected Columns**: 
  - Region (state/country name)
  - Date (time period)
  - Estimated Unemployment Rate (percentage)
- **Note**: This is a user-provided dataset. Place the CSV file in this folder before running the script.

## Approach
1. **Data Loading & Preprocessing**:
   - Load unemployment data from CSV
   - Convert date column to datetime format
   - Extract year and month for analysis
   - Categorize periods: Pre-COVID, During-COVID, Post-COVID

2. **Exploratory Data Analysis**:
   - Summary statistics by region
   - Unemployment rate ranges and distributions
   - Temporal patterns identification

3. **Visualizations**:
   - Bar chart showing average unemployment by region
   - Boxplot comparing pre-COVID vs post-COVID periods
   - Top 10 most affected regions ranking
   - Time series trend analysis
   - Heatmap of unemployment by region and year
   - Distribution analysis

4. **Key Insights**:
   - Identify regions most impacted by unemployment
   - Assess COVID-19 impact on employment
   - Track recovery trends

## Results
| Metric | Value |
|--------|-------|
| Total Records | Dataset dependent |
| Number of Regions | Dataset dependent |
| Analysis Period | Dataset dependent |
| Pre/During/Post COVID | Categorized |

*Note: Run the script to generate actual results and visualizations*

## Files
- `PrathamBhat_Task2.py` - Main Python script (generates 6 PNG plots)
- `PrathamBhat_Task2.ipynb` - Jupyter notebook with documentation
- `README.md` - This file

## Visualizations Generated
1. `01_Unemployment_by_Region.png` - Average unemployment rates by region
2. `02_COVID_Comparison_Boxplot.png` - Pre/During/Post-COVID comparison
3. `03_Top_10_Affected_Regions.png` - Top 10 most affected regions
4. `04_Unemployment_Trend.png` - Unemployment trends over time
5. `05_Unemployment_Heatmap.png` - Region vs Year heatmap
6. `06_Distribution_Analysis.png` - Distribution analysis by COVID period

## How to Run

### Prerequisites
1. Download the unemployment dataset from Kaggle
2. Place the CSV file in this folder as `unemployment.csv`
3. Verify column names match your CSV (or edit the COLUMN NAME CONFIG section in the script)

### Option 1: Run Python Script
```bash
cd Task2
python PrathamBhat_Task2.py
```

**Output**: Console analysis + 6 PNG visualization files

### Option 2: Run Jupyter Notebook
```bash
cd Task2
jupyter notebook PrathamBhat_Task2.ipynb
```

### Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn

### Install Dependencies (if needed)
```bash
pip install pandas numpy matplotlib seaborn
```

## Column Name Configuration
If your CSV has different column names, edit these 3 lines in the Python script:
```python
REGION_COLUMN = 'Region'  # Change to your region column name
DATE_COLUMN = 'Date'  # Change to your date column name
UNEMPLOYMENT_RATE_COLUMN = 'Estimated Unemployment Rate'  # Change if needed
```

## Key Findings & Interpretation
- Unemployment rates vary significantly across regions
- COVID-19 period shows distinct spike in unemployment
- Recovery patterns differ by region
- Historical trends reveal economic cycles

## Author
**Pratham Bhat** - OASIS INFOBYTE Data Science Internship (OIBSIP)
