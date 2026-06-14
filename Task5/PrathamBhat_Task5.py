"""
Task 5: Sales Prediction using Python
Author: Pratham Bhat
Objective: Predict sales based on advertising spend using Linear Regression
Dataset: advertising.csv (user-provided)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# COLUMN NAME CONFIG - Edit these if your CSV has different column names
# ============================================================================
TV_COLUMN = 'TV'
RADIO_COLUMN = 'Radio'
NEWSPAPER_COLUMN = 'Newspaper'
SALES_COLUMN = 'Sales'
# ============================================================================

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the advertising dataset
# Place the dataset CSV in this folder before running
print("Loading Advertising Dataset...")
try:
    df = pd.read_csv('advertising.csv')
except FileNotFoundError:
    print("ERROR: advertising.csv not found in this folder!")
    print("Please download the dataset from Kaggle and place it here.")
    exit(1)

# Remove unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nDataset Info:")
print(df.info())

# ============================================================================
# DATA PREPROCESSING & EDA
# ============================================================================
print("\n" + "="*70)
print("DATA PREPROCESSING & EXPLORATORY DATA ANALYSIS")
print("="*70)

print(f"\nStatistical Summary:")
print(df.describe())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# ============================================================================
# VISUALIZATION 1: Correlation Heatmap
# ============================================================================
print("\nGenerating visualizations...")

fig, ax = plt.subplots(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax.set_title('Advertising Dataset - Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('01_Correlation_Heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_Correlation_Heatmap.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Scatter Plots - Individual Channel Effects
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Sales vs Advertising Channels', fontsize=14, fontweight='bold')

channels = [(TV_COLUMN, 'TV', 'steelblue'),
            (RADIO_COLUMN, 'Radio', 'coral'),
            (NEWSPAPER_COLUMN, 'Newspaper', 'mediumseagreen')]

for idx, (col, label, color) in enumerate(channels):
    ax = axes[idx]
    ax.scatter(df[col], df[SALES_COLUMN], alpha=0.6, s=60, color=color, edgecolors='black', linewidth=0.5)
    
    # Add regression line
    z = np.polyfit(df[col], df[SALES_COLUMN], 1)
    p = np.poly1d(z)
    ax.plot(df[col], p(df[col]), "r--", linewidth=2, label=f'Linear fit')
    
    # Calculate correlation
    corr = df[col].corr(df[SALES_COLUMN])
    ax.set_xlabel(f'{label} Spend ($1000s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sales ($1000s)', fontsize=11, fontweight='bold')
    ax.set_title(f'{label} (Correlation: {corr:.3f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('02_Channel_Analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_Channel_Analysis.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: Distribution Analysis
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Distribution Analysis', fontsize=14, fontweight='bold')

# TV distribution
axes[0, 0].hist(df[TV_COLUMN], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('TV Spend ($1000s)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0, 0].set_title('TV Spend Distribution', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Radio distribution
axes[0, 1].hist(df[RADIO_COLUMN], bins=30, color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Radio Spend ($1000s)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Radio Spend Distribution', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Newspaper distribution
axes[1, 0].hist(df[NEWSPAPER_COLUMN], bins=30, color='mediumseagreen', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Newspaper Spend ($1000s)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Newspaper Spend Distribution', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Sales distribution
axes[1, 1].hist(df[SALES_COLUMN], bins=30, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Sales ($1000s)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Sales Distribution', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('03_Distribution_Analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_Distribution_Analysis.png")
plt.close()

# ============================================================================
# MODEL TRAINING
# ============================================================================
print("\n" + "="*70)
print("LINEAR REGRESSION MODEL TRAINING")
print("="*70)

# Prepare features and target
X = df[[TV_COLUMN, RADIO_COLUMN, NEWSPAPER_COLUMN]]
y = df[SALES_COLUMN]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X, y)

# Make predictions
y_pred = lr_model.predict(X)

# Calculate metrics
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"\n✓ Model training completed")
print(f"\nModel Coefficients:")
print(f"  TV: {lr_model.coef_[0]:.6f}")
print(f"  Radio: {lr_model.coef_[1]:.6f}")
print(f"  Newspaper: {lr_model.coef_[2]:.6f}")
print(f"  Intercept: {lr_model.intercept_:.6f}")

print(f"\nModel Performance:")
print(f"  R² Score: {r2:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")

# ============================================================================
# VISUALIZATION 4: Actual vs Predicted
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y, y_pred, alpha=0.6, s=60, color='steelblue', edgecolors='navy', linewidth=0.5)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Sales ($1000s)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Sales ($1000s)', fontsize=12, fontweight='bold')
ax.set_title(f'Linear Regression: Actual vs Predicted Sales\n(R² = {r2:.4f}, MAE = {mae:.4f})', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('04_Actual_vs_Predicted.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 04_Actual_vs_Predicted.png")
plt.close()

# ============================================================================
# VISUALIZATION 5: Residuals Analysis
# ============================================================================

residuals = y - y_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals vs Predicted
axes[0].scatter(y_pred, residuals, alpha=0.6, s=60, color='steelblue', edgecolors='navy', linewidth=0.5)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted Sales ($1000s)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Residuals ($1000s)', fontsize=11, fontweight='bold')
axes[0].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Residuals distribution
axes[1].hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Residuals ($1000s)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1].set_title(f'Residuals Distribution (Mean: {residuals.mean():.4f})', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('05_Residuals_Analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_Residuals_Analysis.png")
plt.close()

# ============================================================================
# VISUALIZATION 6: Feature Coefficients
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))
channels_names = ['TV', 'Radio', 'Newspaper']
coefficients = lr_model.coef_
colors = ['steelblue', 'coral', 'mediumseagreen']

bars = ax.barh(channels_names, coefficients, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax.set_title('Feature Coefficients - Impact on Sales', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, coef in zip(bars, coefficients):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{coef:.4f}', ha='left' if coef > 0 else 'right', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('06_Feature_Coefficients.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_Feature_Coefficients.png")
plt.close()

# ============================================================================
# SUMMARY AND INTERPRETATION
# ============================================================================
print("\n" + "="*70)
print("TASK COMPLETION SUMMARY")
print("="*70)
print(f"\n✓ All visualizations saved successfully!")
print(f"\nDataset Summary:")
print(f"  Total records: {len(df)}")
print(f"  Features: TV, Radio, Newspaper advertising spend")
print(f"  Target: Sales")
print(f"\nLinear Regression Model:")
print(f"  Equation: Sales = {lr_model.intercept_:.4f} + {lr_model.coef_[0]:.4f}*TV + {lr_model.coef_[1]:.4f}*Radio + {lr_model.coef_[2]:.4f}*Newspaper")
print(f"\nModel Performance:")
print(f"  R² Score: {r2:.4f} (explains {r2*100:.2f}% of variance)")
print(f"  MAE: {mae:.4f} (average error: $${mae*1000:.0f})")
print(f"  RMSE: {rmse:.4f}")
print(f"\nFeature Impact (Coefficient):")
print(f"  TV has coefficient: {lr_model.coef_[0]:.4f} - strongest impact on sales")
print(f"  Radio has coefficient: {lr_model.coef_[1]:.4f}")
print(f"  Newspaper has coefficient: {lr_model.coef_[2]:.4f} - weakest impact on sales")
print(f"\nGenerated Visualizations:")
print("  1. 01_Correlation_Heatmap.png - Feature correlation matrix")
print("  2. 02_Channel_Analysis.png - Sales vs individual advertising channels")
print("  3. 03_Distribution_Analysis.png - Distributions of all variables")
print("  4. 04_Actual_vs_Predicted.png - Model prediction accuracy")
print("  5. 05_Residuals_Analysis.png - Residuals analysis and distribution")
print("  6. 06_Feature_Coefficients.png - Feature impact on sales")
print("\n" + "="*70)
