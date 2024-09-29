import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('csv/technologies.csv', sep=';')

# Create a scatter plot using Seaborn
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='TRL', y='PriceAvg', hue='Category', size='Use', sizes=(50, 500), palette='viridis', alpha=0.7)
plt.title('Technology Analysis')
plt.xlabel('TRL (Technology Readiness Level)')
plt.ylabel('Average Price')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()