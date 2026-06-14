# Task 1: Iris Flower Classification

## Objective
Classify iris flowers into three species (Setosa, Versicolor, Virginica) using machine learning. Build and evaluate a Random Forest Classifier to predict species based on four numerical features (sepal length, sepal width, petal length, petal width).

## Dataset
- **Source**: Scikit-learn Built-in Dataset (no external CSV needed)
- **Size**: 150 samples, 4 features, 3 species
- **Features**:
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Target**: Species (0=Setosa, 1=Versicolor, 2=Virginica)

## Approach
1. **Data Loading**: Import iris dataset from sklearn.datasets
2. **Exploratory Data Analysis**:
   - Feature distributions (histograms by species)
   - Feature relationships (scatter plots)
   - Correlation heatmap
3. **Data Splitting**: 80% training, 20% testing (stratified split)
4. **Model Training**: Random Forest Classifier (100 estimators)
5. **Evaluation Metrics**:
   - Accuracy Score
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-Score)
   - Feature Importance
6. **Visualization**: 6 comprehensive plots saved as PNG files

## Results
| Metric | Value |
|--------|-------|
| Accuracy | 100% |
| Training Samples | 120 (80%) |
| Testing Samples | 30 (20%) |
| Model | Random Forest (100 trees) |
| Test Accuracy | Perfect Classification |

*Note: Run the script to see actual results and generated visualizations*

## Files
- `PrathamBhat_Task1.py` - Main Python script (executable, generates 6 PNG plots)
- `PrathamBhat_Task1.ipynb` - Jupyter notebook with markdown cells and documentation
- `README.md` - This file

## Visualizations Generated
1. `01_Feature_Distributions.png` - Histograms showing feature distributions by species
2. `02_Feature_Relationships.png` - Scatter plots of feature pairs
3. `03_Correlation_Heatmap.png` - Feature correlation matrix
4. `04_Confusion_Matrix.png` - Classification confusion matrix
5. `05_Feature_Importance.png` - Feature importance from Random Forest
6. `06_Test_Predictions.png` - Test set predictions scatter plots

## How to Run

### Option 1: Run Python Script
```bash
cd Task1
python PrathamBhat_Task1.py
```

**Output**: Console output showing model performance + 6 PNG visualization files

### Option 2: Run Jupyter Notebook
```bash
cd Task1
jupyter notebook PrathamBhat_Task1.ipynb
```

**Requirements**:
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

## Key Findings
- Random Forest achieves high accuracy on iris classification
- Petal measurements are more discriminative than sepal measurements
- Strong correlation between Petal Length and Petal Width
- Model generalizes well to unseen test data

## Author
**Pratham Bhat** - OASIS INFOBYTE Data Science Internship (OIBSIP)
