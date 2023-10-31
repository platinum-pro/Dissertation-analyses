import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('Study 1c.csv')

# Create df_wide
df_wide = df.drop_duplicates(subset='ResponseId', keep='first').reset_index(drop=True)
# df_wide = df.groupby('ResponseId').mean().reset_index()
# print(df_wide)

# Calculate IQR for Qm
Q1 = df_wide['Qm'].quantile(0.25)
Q3 = df_wide['Qm'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Flag outliers in a new column
df_wide['is_outlier'] = (df_wide['Qm'] < lower_bound) | (df_wide['Qm'] > upper_bound)

# Filter out outliers for plotting
df_filtered = df_wide[~df_wide['is_outlier']]

# Calculate correlation for filtered data
correlation_Qo_Qm = df_filtered['Qo'].corr(df_filtered['Qm'])

# Calculate counts for used and dropped data points
N_used = len(df_filtered)
N_dropped = len(df_wide) - N_used

# Plot scatterplot for Qo vs Qm with regression line for the filtered data
plt.figure(figsize=(10, 6))
sns.regplot(x='Qo', y='Qm', data=df_filtered, line_kws={"color":"red"})
plt.title(f'Correlation between Qo and Qm without outliers\nr = {correlation_Qo_Qm:.2f}\nUsed: {N_used} | Dropped: {N_dropped}')
plt.xlabel('Qo')
plt.ylabel('Qm')
plt.show()


# Identify 'ResponseId's that produced outliers
# outlier_ids = df_wide[df_wide['is_outlier']]['ResponseId'].tolist()
# print(f"ResponseId's that produced outliers: {outlier_ids}")

