"""
Task 3: Car Price Prediction with ML
Author: Pratham Bhat
Objective: Predict car selling prices using Linear Regression and Random Forest
Dataset: car_data.csv (user-provided)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# COLUMN NAME CONFIG - Edit these if your CSV has different column names
# ============================================================================
SELLING_PRICE_COLUMN = 'selling_price'
PRESENT_PRICE_COLUMN = 'present_price'
KMS_DRIVEN_COLUMN = 'kms_driven'
FUEL_TYPE_COLUMN = 'fuel_type'
SELLER_TYPE_COLUMN = 'seller_type'
TRANSMISSION_COLUMN = 'transmission'
OWNER_COLUMN = 'owner'
YEAR_COLUMN = 'year'
# ============================================================================

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the car price dataset
# Place the dataset CSV in this folder before running
print("Loading Car Price Dataset...")
try:
    df = pd.read_csv('car_data.csv')
except FileNotFoundError:
    print("ERROR: car_data.csv not found in this folder!")
    print("Please download the dataset from Kaggle and place it here.")
    exit(1)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nDataset Info:")
print(df.info())

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

# Drop rows with missing values in key columns
df = df.dropna(subset=[SELLING_PRICE_COLUMN, PRESENT_PRICE_COLUMN, KMS_DRIVEN_COLUMN])

# Encode categorical variables
df_encoded = df.copy()
categorical_cols = [FUEL_TYPE_COLUMN, SELLER_TYPE_COLUMN, TRANSMISSION_COLUMN]
for col in categorical_cols:
    if col in df_encoded.columns:
        df_encoded[col] = pd.Categorical(df_encoded[col]).codes

print(f"Preprocessed dataset shape: {df_encoded.shape}")
print(f"\nPrice statistics (Selling Price):")
print(df[SELLING_PRICE_COLUMN].describe())

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

# Select numeric columns for correlation
numeric_cols = [SELLING_PRICE_COLUMN, PRESENT_PRICE_COLUMN, KMS_DRIVEN_COLUMN, YEAR_COLUMN]
numeric_cols = [col for col in numeric_cols if col in df.columns]

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax.set_title('Car Features - Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('01_Correlation_Heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_Correlation_Heatmap.png")
plt.close()

# Scatter plots: Selling Price vs Key Features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Selling Price vs Car Features', fontsize=14, fontweight='bold')

feature_pairs = [(PRESENT_PRICE_COLUMN, 'Present Price'),
                 (KMS_DRIVEN_COLUMN, 'KMs Driven'),
                 (YEAR_COLUMN, 'Year'),
                 ('owner', 'Owner')]

for idx, (feature, label) in enumerate(feature_pairs):
    if feature in df.columns:
        ax = axes[idx // 2, idx % 2]
        ax.scatter(df[feature], df[SELLING_PRICE_COLUMN], alpha=0.5, s=50, color='steelblue', edgecolors='navy')
        ax.set_xlabel(label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Selling Price', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_Feature_Analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_Feature_Analysis.png")
plt.close()

# Distribution of selling price
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df[SELLING_PRICE_COLUMN], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Selling Price', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title('Distribution of Selling Price', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

df[SELLING_PRICE_COLUMN].plot(kind='box', ax=axes[1])
axes[1].set_ylabel('Selling Price', fontsize=11, fontweight='bold')
axes[1].set_title('Boxplot of Selling Price', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_Price_Distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_Price_Distribution.png")
plt.close()

# ============================================================================
# MODEL TRAINING
# ============================================================================
print("\n" + "="*70)
print("MODEL TRAINING")
print("="*70)

# Prepare features and target
feature_cols = [PRESENT_PRICE_COLUMN, KMS_DRIVEN_COLUMN, FUEL_TYPE_COLUMN, 
                SELLER_TYPE_COLUMN, TRANSMISSION_COLUMN, OWNER_COLUMN, YEAR_COLUMN]
feature_cols = [col for col in feature_cols if col in df_encoded.columns]

X = df_encoded[feature_cols]
y = df_encoded[SELLING_PRICE_COLUMN]

# Split data (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print(f"\n✓ Linear Regression Model Trained")
print(f"  MAE: {lr_mae:.4f}")
print(f"  R²: {lr_r2:.4f}")
print(f"  RMSE: {lr_rmse:.4f}")

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"\n✓ Random Forest Regressor Model Trained")
print(f"  MAE: {rf_mae:.4f}")
print(f"  R²: {rf_r2:.4f}")
print(f"  RMSE: {rf_rmse:.4f}")

# ============================================================================
# MODEL EVALUATION
# ============================================================================
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

# Model comparison
fig, ax = plt.subplots(figsize=(10, 6))
models = ['Linear Regression', 'Random Forest']
mae_scores = [lr_mae, rf_mae]
r2_scores = [lr_r2, rf_r2]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, mae_scores, width, label='MAE', color='steelblue', alpha=0.8)
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, r2_scores, width, label='R²', color='coral', alpha=0.8)

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('MAE', fontsize=11, fontweight='bold', color='steelblue')
ax2.set_ylabel('R² Score', fontsize=11, fontweight='bold', color='coral')
ax.set_title('Model Comparison: Linear Regression vs Random Forest', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.tick_params(axis='y', labelcolor='steelblue')
ax2.tick_params(axis='y', labelcolor='coral')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

fig.legend(handles=bars1+bars2, loc='upper left', bbox_to_anchor=(0.12, 0.95))
plt.tight_layout()
plt.savefig('04_Model_Comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_Model_Comparison.png")
plt.close()

# Actual vs Predicted - Linear Regression
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, y_pred_lr, alpha=0.6, s=50, color='steelblue', edgecolors='navy')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[0].set_xlabel('Actual Price', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Predicted Price', fontsize=11, fontweight='bold')
axes[0].set_title(f'Linear Regression (R²={lr_r2:.4f})', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Actual vs Predicted - Random Forest
axes[1].scatter(y_test, y_pred_rf, alpha=0.6, s=50, color='coral', edgecolors='darkred')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[1].set_xlabel('Actual Price', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Predicted Price', fontsize=11, fontweight='bold')
axes[1].set_title(f'Random Forest (R²={rf_r2:.4f})', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_Actual_vs_Predicted.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_Actual_vs_Predicted.png")
plt.close()

# Feature Importance (Random Forest)
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], 
               color='steelblue', alpha=0.8, edgecolor='navy')
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance - Random Forest Regressor', fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.4f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('06_Feature_Importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_Feature_Importance.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TASK COMPLETION SUMMARY")
print("="*70)
print(f"\n✓ All visualizations saved successfully!")
print(f"\nDataset Summary:")
print(f"  Total samples: {len(df)}")
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")
print(f"  Number of features: {len(feature_cols)}")
print(f"\nBest Model: {'Random Forest' if rf_r2 > lr_r2 else 'Linear Regression'}")
print(f"  R² Score: {max(rf_r2, lr_r2):.4f}")
print(f"  MAE: {min(rf_mae, lr_mae):.4f}")
print(f"\nGenerated Visualizations:")
print("  1. 01_Correlation_Heatmap.png - Feature correlation matrix")
print("  2. 02_Feature_Analysis.png - Scatter plots of key features")
print("  3. 03_Price_Distribution.png - Distribution analysis")
print("  4. 04_Model_Comparison.png - Linear vs Random Forest")
print("  5. 05_Actual_vs_Predicted.png - Prediction accuracy visualization")
print("  6. 06_Feature_Importance.png - Feature importance ranking")
print("\n" + "="*70)
