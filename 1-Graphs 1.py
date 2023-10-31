import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd

# Assuming you have the dataframes data_1c and data_1e loaded already
data_1c = pd.read_csv('Study 1c.csv')
data_1e = pd.read_csv('Study 1f.csv')

# Replace infinities with NaN
data_1c = data_1c.replace([np.inf, -np.inf], np.nan)
data_1e = data_1e.replace([np.inf, -np.inf], np.nan)

# Model function
def model(x, Qo, k, alpha):
    return Qo * 10 ** (k * np.exp(-alpha * Qo * x) - 1)

# Plot Settings
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Plot Individual Data Points
for _, individual_data in data_1c.groupby('ResponseId'):  # Assuming there's an 'ID' column to differentiate individuals
    try:
        params, _ = curve_fit(model, individual_data['x'], individual_data['y'])
        
        # Generate dense x-values for smoother curves
        x_dense = np.linspace(min(individual_data['x']), max(individual_data['x']), 1000)
        y_fitted = model(x_dense, *params)
        
        plt.plot(x_dense, y_fitted, color='grey', lw=0.5, alpha=0.2)
    except:
        continue  # Skip individuals that cause errors in curve fitting

# Define a color map for each condition
colors = ['red', 'blue', 'green']

# Aggregate Data with Fitted Model
for index, condition in enumerate(['Dosage_ann', 'Dosage_one', 'Dosage_two']):
# for index, condition in enumerate(['Mode_muc', 'Mode_oral', 'Mode_sub']):
    try:
        params, _ = curve_fit(model, data_1e['x'], data_1e[condition])
        y_fitted = model(data_1e['x'], *params)
        plt.plot(data_1e['x'], y_fitted, label=condition, linestyle='--', color=colors[index])
    except:
        continue # Skip conditions that cause errors in curve fitting

# Graph Details
# plt.xscale('log')
plt.yscale('log')

# Set the tick labels for x-axis and y-axis
plt.gca().set_xticks([1, 10, 100, 1000, 2000])
plt.gca().set_yticks([1, 10, 100])

# Set the tick labels to be displayed in integer format
plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
plt.gca().get_yaxis().set_major_formatter(plt.ScalarFormatter())

# Ensure the minor ticks are turned off to avoid clutter
# plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
# plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())

plt.xlabel('Price ($USD)')
plt.ylabel('Likelihood of Consumption (%)')
plt.legend()
plt.title('Demand for HIV Vaccines based on Dosage')
# plt.title('Demand for HIV Vaccines based on Mode of Administration')
plt.show()
