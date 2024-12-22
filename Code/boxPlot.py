import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv('Files/dataset.csv')

# Checking for numeric columns for further analysis
numeric_columns = [col for col in ['Quantity', 'UnitPrice', 'Sales'] if col in df.columns]

# Plotting box plots for numeric columns
if numeric_columns:
    plt.figure(figsize=(14, 8))
    palette = sns.color_palette("coolwarm", len(numeric_columns))

    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(1, len(numeric_columns), i)
        ax = sns.boxplot(y=df[column], color=palette[i - 1])
        
        # Add gridlines for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Annotate median values on the boxplot
        median = df[column].median()
        plt.text(0.1, median, f'Median: {median:.2f}', 
                 horizontalalignment='center', verticalalignment='center', 
                 color='black', fontsize=8, backgroundcolor='white')
        
        # Set plot title and labels
        plt.title(f'Box Plot of {column}', fontsize=14, fontweight='bold')
        plt.ylabel(column, fontsize=8)
        plt.xlabel('')

    plt.tight_layout()
    plt.show()
else:
    print("No valid numeric columns found for box plots.")
