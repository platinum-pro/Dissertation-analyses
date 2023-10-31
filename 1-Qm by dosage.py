import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd

# Load the data
df_wide= pd.read_csv('Study 1d.csv')

# Create df
df = df_wide.drop_duplicates(subset='ResponseId', keep='first').reset_index(drop=True)

# Ensure 'alpha' values are positive before taking the logarithm
df = df[df['alpha'] > 0]

# Convert 'alpha' to log10 units
df['log_alpha'] = np.log10(df['alpha'])

# Filter dataframe for the columns of interest
df_interest = df[['ResponseId', 'log_alpha', 'Qo', 'Qm', 'Dosage_ann', 'Dosage_one', 'Dosage_two', 'Mode_muc', 'Mode_oral', 'Mode_sub']]

# Count the data points before outlier removal
initial_count = df_interest.shape[0]

# Remove outliers for Qm
Q1 = df_interest['Qo'].quantile(0.25)
Q3 = df_interest['Qo'].quantile(0.75)
IQR = Q3 - Q1

filter = (df_interest['Qo'] >= Q1 - 1.5 * IQR) & (df_interest['Qo'] <= Q3 + 1.5 * IQR)
df_interest_filtered = df_interest[filter]

# Count the data points after outlier removal
filtered_count = df_interest_filtered.shape[0]

# Calculate counts for used and dropped data points
used_data_points = filtered_count
dropped_data_points = initial_count - filtered_count


# Create a new column to represent the group
# df_interest_filtered['group'] = np.where(df_interest_filtered['Dosage_ann'], 'Dosage_ann',
#                                         np.where(df_interest_filtered['Dosage_one'], 'Dosage_one', 'Dosage_two'))

# Create a new column to represent the group
df_interest_filtered['group'] = np.where(df_interest_filtered['Mode_muc'], 'Mode_muc',
                                         np.where(df_interest_filtered['Mode_oral'], 'Mode_oral', 'Mode_sub'))

# Specify a list of colors or use a predefined palette
colors = ['blue', 'green', 'red']

# Plot box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='group', y='Qo', data=df_interest_filtered, palette=colors, width=0.5)

title_str = (f"Average for each Mode Group\n"
             f"Used Data Points: {used_data_points} | Dropped Data Points: {dropped_data_points}")

plt.title(title_str)

# Log scale for the y-axis
# plt.yscale('log')

# Custom tick labels
# ticks = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5,]
# plt.yticks(ticks, ticks)

plt.ylabel('Qo')
plt.xlabel('Mode of Administration Group')
plt.show()
