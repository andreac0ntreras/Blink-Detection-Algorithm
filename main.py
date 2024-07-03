from utils import processing_utils, plotting_utils, sourcing_utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM

# Define folder paths
folder_path = 'data/raw/XDF Files'
output_folder = 'output/interim'

# Process the XDF files
sourcing_utils.process_xdf_files(folder_path, output_folder)

# Create a new CSV file with all participants, days, blink rates, and other features
compiled_df = processing_utils.process_csv_files(output_folder)

# Plot pupil size data over time with blink annotations for all CSV files in the specified folder.
plotting_utils.plot_all_time_v_pupil_size_csv_files_in_directory(output_folder)

# Transform the data to long format if needed
blink_rate_data_long = compiled_df.pivot(index='subject', columns='day',
                                         values='concat_average_blink_rate').reset_index()
blink_rate_data_long.columns = ['subject', 'day1', 'day2', 'day3']

# Transform the data to long format if needed
missing_data_long = (compiled_df.pivot(index='subject', columns='day',
                                       values='left_missing_data_percentage_excluding_blinks_and_min_time_range').
                     reset_index())
missing_data_long.columns = ['subject', 'day1', 'day2', 'day3']

# Melt the dataframe back to long format for ANOVA
blink_rate_data_melted = pd.melt(blink_rate_data_long, id_vars=['subject'], value_vars=['day1', 'day2', 'day3'],
                                 var_name='day', value_name='concat_average_blink_rate')

missing_data_melted = pd.melt(missing_data_long, id_vars=['subject'], value_vars=['day1', 'day2', 'day3'],
                              var_name='day',
                              value_name='left_missing_data_percentage_excluding_blinks_and_min_time_range')

# Fit the ANOVA model
aovrm = AnovaRM(blink_rate_data_melted, 'concat_average_blink_rate', 'subject', within=['day'])
res = aovrm.fit()

# Print the summary of the ANOVA
print(res.summary())

# Line plot
sns.lineplot(x='day', y='concat_average_blink_rate', data=blink_rate_data_melted, hue='subject', marker='o')
plt.title('Blink Rate Across Different Days for Each Subject')
plt.xlabel('Day')
plt.ylabel('Average Blink Rate')
plt.show()

sns.lineplot(x='day', y='left_missing_data_percentage_excluding_blinks_and_min_time_range', data=missing_data_melted,
             hue='subject', marker='o')
plt.title('Missing Data Across Different Days for Each Subject')
plt.xlabel('Day')
plt.ylabel('Percentage of Missing Data')
plt.show()

"""
             Anova
================================
    F Value Num DF Den DF Pr > F
--------------------------------
day  0.9166 2.0000 8.0000 0.4381
================================


There is no statistically significant difference in the average blink rate across the three days for each subject.
The p-value of 0.4381 indicates that any observed differences in blink rate are likely due to random variation
rather than an effect of the day.
"""
