"""
Task 4: Email Spam Detection with ML
Author: Pratham Bhat
Objective: Detect spam emails using TF-IDF and Naive Bayes
Dataset: spam.csv (user-provided)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# COLUMN NAME CONFIG - Edit these if your CSV has different column names
# ============================================================================
LABEL_COLUMN = 'label'  # or 'v1' - column with spam/ham labels
MESSAGE_COLUMN = 'message'  # or 'v2' - column with email text
# ============================================================================

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the spam dataset
# Place the dataset CSV in this folder before running
print("Loading Email Spam Dataset...")
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
except FileNotFoundError:
    print("ERROR: spam.csv not found in this folder!")
    print("Please download the dataset from Kaggle and place it here.")
    exit(1)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset Info:")
print(df.info())

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

# Select relevant columns
if LABEL_COLUMN in df.columns and MESSAGE_COLUMN in df.columns:
    df = df[[LABEL_COLUMN, MESSAGE_COLUMN]].copy()
else:
    print("ERROR: Column names not found. Please check your CSV column names:")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

# Convert labels to binary (spam=1, ham=0)
df[LABEL_COLUMN] = df[LABEL_COLUMN].map({'spam': 1, 'ham': 0})
if df[LABEL_COLUMN].isna().any():
    # Try alternative mapping
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.lower().map({'spam': 1, 'ham': 0})

# Remove rows with missing values
df = df.dropna()

print(f"Preprocessed dataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df[LABEL_COLUMN].value_counts())
print(f"\nClass percentages:")
print(df[LABEL_COLUMN].value_counts(normalize=True) * 100)

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

# Message length analysis
df['message_length'] = df[MESSAGE_COLUMN].apply(len)
df['word_count'] = df[MESSAGE_COLUMN].apply(lambda x: len(str(x).split()))

print(f"\nMessage Statistics:")
print(f"  Mean length: {df['message_length'].mean():.2f} characters")
print(f"  Mean word count: {df['word_count'].mean():.2f} words")
print(f"  Max length: {df['message_length'].max()} characters")
print(f"  Min length: {df['message_length'].min()} characters")

# Visualization: Class distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Class count
labels_text = ['Ham', 'Spam']
counts = df[LABEL_COLUMN].value_counts().sort_index().values
colors = ['green', 'red']
axes[0].bar(labels_text, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[0].set_title('Email Distribution', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, count in enumerate(counts):
    axes[0].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')

# Message length distribution
for label, color in [(0, 'green'), (1, 'red')]:
    axes[1].hist(df[df[LABEL_COLUMN] == label]['message_length'], 
                alpha=0.6, label=labels_text[label], color=color, bins=30)
axes[1].set_xlabel('Message Length (characters)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1].set_title('Message Length Distribution', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

# Word count distribution
for label, color in [(0, 'green'), (1, 'red')]:
    axes[2].hist(df[df[LABEL_COLUMN] == label]['word_count'], 
                alpha=0.6, label=labels_text[label], color=color, bins=30)
axes[2].set_xlabel('Word Count', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[2].set_title('Word Count Distribution', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('01_Data_Distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 01_Data_Distribution.png")
plt.close()

# Boxplot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df.boxplot(column='message_length', by=LABEL_COLUMN, ax=axes[0])
axes[0].set_xticklabels(labels_text)
axes[0].set_xlabel('Email Type', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Message Length (characters)', fontsize=11, fontweight='bold')
axes[0].set_title('Message Length Comparison', fontsize=12, fontweight='bold')
plt.sca(axes[0])
plt.xticks([1, 2], labels_text)

df.boxplot(column='word_count', by=LABEL_COLUMN, ax=axes[1])
axes[1].set_xticklabels(labels_text)
axes[1].set_xlabel('Email Type', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Word Count', fontsize=11, fontweight='bold')
axes[1].set_title('Word Count Comparison', fontsize=12, fontweight='bold')
plt.sca(axes[1])
plt.xticks([1, 2], labels_text)

plt.tight_layout()
plt.savefig('02_Feature_Comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_Feature_Comparison.png")
plt.close()

# ============================================================================
# TEXT VECTORIZATION AND MODEL TRAINING
# ============================================================================
print("\n" + "="*70)
print("TEXT VECTORIZATION AND MODEL TRAINING")
print("="*70)

# TF-IDF Vectorization
print("Vectorizing text data using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', lowercase=True)
X = tfidf.fit_transform(df[MESSAGE_COLUMN])
y = df[LABEL_COLUMN].values

print(f"TF-IDF matrix shape: {X.shape}")
print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Train Multinomial Naive Bayes
print("\nTraining Multinomial Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Make predictions
y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✓ Model training completed")
print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ============================================================================
# MODEL EVALUATION
# ============================================================================
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels_text))

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Visualization: Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=labels_text, yticklabels=labels_text,
            cbar_kws={"shrink": 0.8}, linewidths=2, linecolor='black')
ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.4f})', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('03_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 03_Confusion_Matrix.png")
plt.close()

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Visualization: Performance Metrics
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax.bar(metrics, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2., value,
            f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('04_Performance_Metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_Performance_Metrics.png")
plt.close()

# Top features for spam detection
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Get feature names
feature_names = np.array(tfidf.get_feature_names_out())

# Top spam features (positive coefficients)
spam_coef = nb_model.feature_log_prob_[1]
top_spam_idx = spam_coef.argsort()[-15:][::-1]
axes[0].barh(range(len(top_spam_idx)), spam_coef[top_spam_idx], color='red', alpha=0.7, edgecolor='darkred')
axes[0].set_yticks(range(len(top_spam_idx)))
axes[0].set_yticklabels(feature_names[top_spam_idx])
axes[0].set_xlabel('Log Probability', fontsize=11, fontweight='bold')
axes[0].set_title('Top 15 Spam Indicators', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# Top ham features
ham_coef = nb_model.feature_log_prob_[0]
top_ham_idx = ham_coef.argsort()[-15:][::-1]
axes[1].barh(range(len(top_ham_idx)), ham_coef[top_ham_idx], color='green', alpha=0.7, edgecolor='darkgreen')
axes[1].set_yticks(range(len(top_ham_idx)))
axes[1].set_yticklabels(feature_names[top_ham_idx])
axes[1].set_xlabel('Log Probability', fontsize=11, fontweight='bold')
axes[1].set_title('Top 15 Ham Indicators', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('05_Top_Features.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_Top_Features.png")
plt.close()

# ROC Curve Analysis
from sklearn.metrics import roc_curve, auc
y_pred_proba = nb_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve - Spam Detection', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('06_ROC_Curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_ROC_Curve.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TASK COMPLETION SUMMARY")
print("="*70)
print(f"\n✓ All visualizations saved successfully!")
print(f"\nDataset Summary:")
print(f"  Total emails: {len(df)}")
print(f"  Ham emails: {(y==0).sum()} ({(y==0).sum()/len(df)*100:.1f}%)")
print(f"  Spam emails: {(y==1).sum()} ({(y==1).sum()/len(df)*100:.1f}%)")
print(f"\nModel Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")
print(f"  ROC-AUC: {roc_auc:.4f}")
print(f"\nFeature Engineering:")
print(f"  TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")
print(f"  Features used: 5000 (top features)")
print(f"\nGenerated Visualizations:")
print("  1. 01_Data_Distribution.png - Email distribution analysis")
print("  2. 02_Feature_Comparison.png - Message length and word count comparison")
print("  3. 03_Confusion_Matrix.png - Classification confusion matrix")
print("  4. 04_Performance_Metrics.png - Accuracy, Precision, Recall, F1-Score")
print("  5. 05_Top_Features.png - Top spam and ham indicators")
print("  6. 06_ROC_Curve.png - ROC curve and AUC score")
print("\n" + "="*70)
