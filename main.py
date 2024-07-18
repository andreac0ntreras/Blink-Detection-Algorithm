import blinkdetection as bd

# Define folder paths
folder_path = 'data/raw/XDF Files'
interim_folder = 'output/interim'

# Process the XDF files
# bd.process_xdf_files(folder_path, interim_folder)

# Create a new CSV file with all participants, days, blink features, pupil features, and quality insights
# CSV is saved to 'output/features/compiled_feature_extraction.csv'
compiled_feature_df, compiled_blink_df = bd.process_csv_files(interim_folder)

# Plot pupil size data over time with blink annotations for all CSV files in the specified folder.
# bd.plot_all_time_v_pupil_size_csv_files_in_directory(interim_folder, show=True)

# Plot missing data and blink rate over time
# bd.plot_feature_over_three_days(compiled_df, show=True)

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
