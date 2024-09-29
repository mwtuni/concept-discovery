import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('csv/technologies.csv', sep=';')

# Create a scatter plot using Seaborn
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='TRL', y='PriceAvg', hue='Category', palette='viridis', alpha=0.7)
plt.title('Technology Readiness Level vs. Average Price')
plt.xlabel('TRL (Technology Readiness Level)')
plt.ylabel('Average Price')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()