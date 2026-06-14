"""
Task 1: Iris Flower Classification
Author: Pratham Bhat
Objective: Classify iris flowers into three species using Random Forest Classifier
Dataset: Built-in sklearn iris dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the iris dataset
print("Loading Iris Dataset...")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for EDA
df = pd.DataFrame(X, columns=feature_names)
df['Species'] = [target_names[i] for i in y]

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nDataset info:\n{df.info()}")
print(f"\nStatistical summary:\n{df.describe()}")

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

# Create EDA plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Iris Dataset - Feature Distributions', fontsize=16, fontweight='bold')

for idx, feature in enumerate(feature_names):
    ax = axes[idx // 2, idx % 2]
    for species in target_names:
        data = df[df['Species'] == species][feature]
        ax.hist(data, alpha=0.6, label=species, bins=15)
    ax.set_xlabel(feature, fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_Feature_Distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_Feature_Distributions.png")
plt.close()

# Pairplot
fig = plt.figure(figsize=(12, 10))
# Manual pairplot
feature_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}

for idx, (i, j) in enumerate(feature_pairs, 1):
    ax = fig.add_subplot(2, 3, idx)
    for species in target_names:
        mask = df['Species'] == species
        ax.scatter(df[mask][feature_names[i]], df[mask][feature_names[j]], 
                  label=species, alpha=0.7, s=100, color=colors[species])
    ax.set_xlabel(feature_names[i], fontsize=10, fontweight='bold')
    ax.set_ylabel(feature_names[j], fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if idx == 1:
        ax.legend(fontsize=9)

fig.suptitle('Iris Dataset - Feature Relationships', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('02_Feature_Relationships.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_Feature_Relationships.png")
plt.close()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
numeric_df = df.iloc[:, :4]
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax.set_title('Iris Features - Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('03_Correlation_Heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_Correlation_Heatmap.png")
plt.close()

# ============================================================================
# MODEL TRAINING
# ============================================================================
print("\n" + "="*70)
print("MODEL TRAINING: RANDOM FOREST CLASSIFIER")
print("="*70)

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Testing set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("\n✓ Model training completed")

# ============================================================================
# MODEL EVALUATION
# ============================================================================
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

# Predictions
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={"shrink": 0.8}, linewidths=2, linecolor='black')
ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.4f})', 
             fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('04_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_Confusion_Matrix.png")
plt.close()

# Feature Importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(importances)), importances[indices], color='steelblue', alpha=0.8)
ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
ax.set_ylabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance - Random Forest Classifier', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(importances)))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('05_Feature_Importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_Feature_Importance.png")
plt.close()

# Test Set Predictions - Scatter plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Model Predictions on Test Set', fontsize=16, fontweight='bold')

test_df = pd.DataFrame(X_test, columns=feature_names)
test_df['True_Species'] = [target_names[i] for i in y_test]
test_df['Pred_Species'] = [target_names[i] for i in y_pred]
test_df['Correct'] = y_test == y_pred

feature_pairs = [(0, 1), (0, 2), (2, 3), (1, 3)]
for idx, (i, j) in enumerate(feature_pairs):
    ax = axes[idx // 2, idx % 2]
    
    # Correct predictions in green, incorrect in red
    correct_mask = test_df['Correct']
    incorrect_mask = ~test_df['Correct']
    
    ax.scatter(test_df[correct_mask][feature_names[i]], 
              test_df[correct_mask][feature_names[j]], 
              c='green', marker='o', s=100, alpha=0.7, label='Correct', edgecolors='darkgreen')
    ax.scatter(test_df[incorrect_mask][feature_names[i]], 
              test_df[incorrect_mask][feature_names[j]], 
              c='red', marker='X', s=150, alpha=0.8, label='Incorrect', edgecolors='darkred')
    
    ax.set_xlabel(feature_names[i], fontsize=11, fontweight='bold')
    ax.set_ylabel(feature_names[j], fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('06_Test_Predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_Test_Predictions.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TASK COMPLETION SUMMARY")
print("="*70)
print(f"\n✓ All plots saved successfully!")
print(f"✓ Model Accuracy: {accuracy*100:.2f}%")
print(f"✓ Total Samples: {len(X)}")
print(f"✓ Training Samples: {len(X_train)}")
print(f"✓ Testing Samples: {len(X_test)}")
print("\nGenerated Visualizations:")
print("  1. 01_Feature_Distributions.png - Histogram of all features by species")
print("  2. 02_Feature_Relationships.png - Scatter plots of feature pairs")
print("  3. 03_Correlation_Heatmap.png - Correlation between features")
print("  4. 04_Confusion_Matrix.png - Classification confusion matrix")
print("  5. 05_Feature_Importance.png - Random Forest feature importances")
print("  6. 06_Test_Predictions.png - Test set predictions visualization")
print("\n" + "="*70)
