import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load exosuits data
exosuits_df = pd.read_csv('csv/exosuits.csv')

# Drop 'PID' column as it's not needed for this analysis
exosuits_df.drop(columns=['PID'], inplace=True)

# Convert 'T1' to 'T124' columns to binary format (0 or 1)
binary_matrix = exosuits_df.applymap(lambda x: 1 if x == 1 else 0)

# Calculate co-occurrence matrix
co_occurrence_matrix = binary_matrix.T.dot(binary_matrix)

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(co_occurrence_matrix, cmap='viridis', annot=True, fmt='g')
plt.title('Co-occurrence of Technologies Across Exosuits')
plt.xlabel('Technology')
plt.ylabel('Technology')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()
