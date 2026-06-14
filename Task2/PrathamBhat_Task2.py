"""
Task 2: Unemployment Analysis with Python
Author: Pratham Bhat
Objective: Analyze unemployment data across regions and time periods
Dataset: unemployment.csv (user-provided)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# COLUMN NAME CONFIG - Edit these 3 lines if your CSV has different column names
# ============================================================================
REGION_COLUMN = 'Region'
DATE_COLUMN = 'Date'
UNEMPLOYMENT_RATE_COLUMN = 'Estimated Unemployment Rate'
# ============================================================================

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the unemployment dataset
# Place the dataset CSV in this folder before running
print("Loading Unemployment Dataset...")
try:
    df = pd.read_csv('unemployment.csv')
except FileNotFoundError:
    print("ERROR: unemployment.csv not found in this folder!")
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

# Convert date column to datetime if it's a string
if df[DATE_COLUMN].dtype == 'object':
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

# Sort by date and region
df = df.sort_values([REGION_COLUMN, DATE_COLUMN]).reset_index(drop=True)

# Extract year and month for analysis
df['Year'] = df[DATE_COLUMN].dt.year
df['Month'] = df[DATE_COLUMN].dt.month

print(f"Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")
print(f"Regions: {df[REGION_COLUMN].nunique()}")
print(f"\nRegions list:")
print(df[REGION_COLUMN].unique())

# Identify pre-COVID and post-COVID periods
# Typically: Pre-COVID = before March 2020, Post-COVID = March 2020 onwards
df['COVID_Period'] = df['Year'].apply(lambda x: 'Pre-COVID' if x < 2020 else ('During-COVID' if x < 2021 else 'Post-COVID'))

print(f"\nCOVID Period Distribution:")
print(df['COVID_Period'].value_counts())

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

# Summary statistics
print(f"\nUnemployment Rate Summary:")
print(df[UNEMPLOYMENT_RATE_COLUMN].describe())

print(f"\nUnemployment Rate by Region:")
region_stats = df.groupby(REGION_COLUMN)[UNEMPLOYMENT_RATE_COLUMN].agg(['mean', 'min', 'max', 'std'])
print(region_stats.sort_values('mean', ascending=False))

# ============================================================================
# VISUALIZATION 1: Bar Chart by Region (Average Unemployment)
# ============================================================================
print("\nGenerating visualizations...")

fig, ax = plt.subplots(figsize=(14, 6))
region_means = df.groupby(REGION_COLUMN)[UNEMPLOYMENT_RATE_COLUMN].mean().sort_values(ascending=False)
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(region_means)))
bars = ax.bar(range(len(region_means)), region_means.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Region', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Average Unemployment Rate by Region', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(region_means)))
ax.set_xticklabels(region_means.index, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('01_Unemployment_by_Region.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_Unemployment_by_Region.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Pre/Post-COVID Boxplot Comparison
# ============================================================================

# Filter data for pre and post COVID comparison
pre_covid = df[df['COVID_Period'].isin(['Pre-COVID', 'During-COVID', 'Post-COVID'])]

fig, ax = plt.subplots(figsize=(12, 6))
covid_order = ['Pre-COVID', 'During-COVID', 'Post-COVID']
sns.boxplot(data=pre_covid, x='COVID_Period', y=UNEMPLOYMENT_RATE_COLUMN, 
            order=covid_order, ax=ax, palette='Set2', width=0.6)
ax.set_xlabel('COVID Period', fontsize=12, fontweight='bold')
ax.set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Unemployment Rate: Pre-COVID vs During-COVID vs Post-COVID', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add mean markers
means = pre_covid.groupby('COVID_Period')[UNEMPLOYMENT_RATE_COLUMN].mean().reindex(covid_order)
positions = range(len(covid_order))
ax.plot(positions, means.values, 'ro-', linewidth=2, markersize=8, label='Mean', zorder=3)
ax.legend()

plt.tight_layout()
plt.savefig('02_COVID_Comparison_Boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_COVID_Comparison_Boxplot.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: Top 10 Most Affected Regions (Highest Avg Unemployment)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))
top_10 = df.groupby(REGION_COLUMN)[UNEMPLOYMENT_RATE_COLUMN].mean().nlargest(10)
colors_top10 = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_10)))
bars = ax.barh(range(len(top_10)), top_10.values, color=colors_top10, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Region', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Most Affected Regions by Unemployment', fontsize=14, fontweight='bold')
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10.index)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.2f}%', ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('03_Top_10_Affected_Regions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_Top_10_Affected_Regions.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: Unemployment Trend Over Time
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 6))
# Get top 5 regions by unemployment
top_5_regions = df.groupby(REGION_COLUMN)[UNEMPLOYMENT_RATE_COLUMN].mean().nlargest(5).index
for region in top_5_regions:
    region_data = df[df[REGION_COLUMN] == region].sort_values(DATE_COLUMN)
    ax.plot(region_data[DATE_COLUMN], region_data[UNEMPLOYMENT_RATE_COLUMN], 
            marker='o', label=region, linewidth=2, alpha=0.7)

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Unemployment Trend Over Time - Top 5 Affected Regions', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('04_Unemployment_Trend.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_Unemployment_Trend.png")
plt.close()

# ============================================================================
# VISUALIZATION 5: Heatmap - Unemployment by Region and Year
# ============================================================================

pivot_data = df.pivot_table(values=UNEMPLOYMENT_RATE_COLUMN, 
                             index=REGION_COLUMN, 
                             columns='Year', 
                             aggfunc='mean')

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r', 
            ax=ax, cbar_kws={"shrink": 0.8}, linewidths=0.5)
ax.set_title('Unemployment Rate Heatmap: Region vs Year', fontsize=14, fontweight='bold')
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Region', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('05_Unemployment_Heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_Unemployment_Heatmap.png")
plt.close()

# ============================================================================
# VISUALIZATION 6: Distribution of Unemployment Rates
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df[UNEMPLOYMENT_RATE_COLUMN], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Unemployment Rate (%)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title('Distribution of Unemployment Rates', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# KDE plot by COVID period
for period in covid_order:
    period_data = pre_covid[pre_covid['COVID_Period'] == period][UNEMPLOYMENT_RATE_COLUMN]
    period_data.plot(kind='kde', ax=axes[1], label=period, linewidth=2)

axes[1].set_xlabel('Unemployment Rate (%)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[1].set_title('Distribution by COVID Period', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('06_Distribution_Analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_Distribution_Analysis.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("TASK COMPLETION SUMMARY")
print("="*70)
print(f"\n✓ All visualizations saved successfully!")
print(f"\nDataset Overview:")
print(f"  Total records: {len(df)}")
print(f"  Date range: {df[DATE_COLUMN].min().date()} to {df[DATE_COLUMN].max().date()}")
print(f"  Number of regions: {df[REGION_COLUMN].nunique()}")
print(f"\nUnemployment Statistics:")
print(f"  Mean: {df[UNEMPLOYMENT_RATE_COLUMN].mean():.2f}%")
print(f"  Median: {df[UNEMPLOYMENT_RATE_COLUMN].median():.2f}%")
print(f"  Min: {df[UNEMPLOYMENT_RATE_COLUMN].min():.2f}%")
print(f"  Max: {df[UNEMPLOYMENT_RATE_COLUMN].max():.2f}%")
print(f"\nGenerated Visualizations:")
print("  1. 01_Unemployment_by_Region.png - Average unemployment by region")
print("  2. 02_COVID_Comparison_Boxplot.png - Pre/During/Post-COVID comparison")
print("  3. 03_Top_10_Affected_Regions.png - Top 10 regions by unemployment")
print("  4. 04_Unemployment_Trend.png - Unemployment trend over time")
print("  5. 05_Unemployment_Heatmap.png - Region vs Year heatmap")
print("  6. 06_Distribution_Analysis.png - Distribution analysis")
print("\n" + "="*70)
