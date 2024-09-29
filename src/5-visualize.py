import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('csv/technologies.csv', sep=';')

# Calculate average price for each category
df['PriceAvg'] = (df['PriceMin'] + df['PriceMax']) / 2

# Create a bar plot using Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Category', y='PriceAvg', palette='viridis')
plt.title('Average Price of Technologies by Category')
plt.xlabel('Category')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.show()