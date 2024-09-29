import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('csv/technologies.csv', sep=';')

# Select numerical columns for correlation analysis
numerical_cols = ['TRL', 'PriceMin', 'PriceMax', 'PriceAvg']

# Calculate the correlation matrix
correlation_matrix = df[numerical_cols].corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()