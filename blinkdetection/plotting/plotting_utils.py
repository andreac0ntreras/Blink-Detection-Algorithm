import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from blinkdetection.blinkdetection import blink_detection_utils
import numpy as np


def plot_pupil_size_v_time(csv_file, show=False):
    """
    This function reads a CSV file containing pupil size data, detects blinks,
    and plots pupil size (left and right eye) vs time.

    Parameters:
        csv_file (str): Path to the CSV file containing pupil size data.
        show (bool): indication of whether to display the plot
    """

    # Read the CSV file using pandas
    dataframe = pd.read_csv(csv_file)

    # Extract columns by header name
    timestamps = dataframe['timestamps']
    pupil_size_left = dataframe['pupil_size_left']
    pupil_size_right = dataframe['pupil_size_right']

    # Detect blinks using a separate function (not included here)
    blinks = blink_detection_utils.calculate_total_blinks_and_missing_data(dataframe, 600)

    # Plot pupil size vs time for left and right eye
    plt.plot(timestamps, pupil_size_left, label='Left Size')
    plt.plot(timestamps, pupil_size_right, label='Right Size', color='darkorange')

    # Mark blink onsets and offsets with vertical lines
    for blink_onset in blinks["blink_onset"]:
        plt.axvline(blink_onset, color='green')
    for blink_offset in blinks["blink_offset"]:
        plt.axvline(blink_offset, color='pink')

    # Label axes and set title
    plt.xlabel('Time (s)')
    plt.ylabel('Size (AU)')
    subject_id = csv_file.split('_')[0].split("/")[-1]
    day_number = csv_file.split('_')[4].split('.')[0]
    plt.title(f'{subject_id}_{day_number} Pupil Size')
    plt.legend()

    # Create output directory if it doesn't exist
    output_dir = '../../output/plots'
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename and save the plot
    plot_filename = os.path.join(output_dir, f"{os.path.basename(csv_file)}_pupil_size_vs_time.png")
    plt.savefig(plot_filename, dpi=1000)

    if show is True:
        plt.show()

    plt.clf()


def interval_based_plot_pupil_size_v_time(csv_file, show=False):
    """
    This function reads a CSV file containing pupil size data, detects blinks,
    and plots pupil size (left and right eye) vs time.

    Parameters:
        csv_file (str): Path to the CSV file containing pupil size data.
        show (bool): indication of whether to display the plot
    """

    # Read the CSV file using pandas
    dataframe = pd.read_csv(csv_file)

    # Extract columns by header name
    timestamps = dataframe['timestamps']
    pupil_size_left = dataframe['pupil_size_left']
    pupil_size_right = dataframe['pupil_size_right']

    # Detect blinks using the updated function
    blinks = blink_detection_utils.calculate_total_blinks_and_missing_data(dataframe, 600)

    # Initialize lists to store plot data for each stream
    left_plot_data = {'timestamps': [], 'pupil_size': []}
    right_plot_data = {'timestamps': [], 'pupil_size': []}

    # Separate the data based on the used stream
    for i, stream in enumerate(blinks["used_stream"]):
        if stream == 'left':
            left_plot_data['timestamps'].append(timestamps[i])
            left_plot_data['pupil_size'].append(pupil_size_left[i])
            right_plot_data['timestamps'].append(np.nan)  # Add NaN for the unused stream
            right_plot_data['pupil_size'].append(np.nan)
        else:
            right_plot_data['timestamps'].append(timestamps[i])
            right_plot_data['pupil_size'].append(pupil_size_right[i])
            left_plot_data['timestamps'].append(np.nan)  # Add NaN for the unused stream
            left_plot_data['pupil_size'].append(np.nan)

    # Plot pupil size vs time for left and right eye
    plt.plot(left_plot_data['timestamps'], left_plot_data['pupil_size'], label='Left Size', color='lightblue')
    plt.plot(right_plot_data['timestamps'], right_plot_data['pupil_size'], label='Right Size', color='darkorange')

    # Mark blink onsets and offsets with vertical lines
    for blink_onset in blinks["blink_onset"]:
        plt.axvline(blink_onset, color='green')
    for blink_offset in blinks["blink_offset"]:
        plt.axvline(blink_offset, color='pink')

    # Label axes and set title
    plt.xlabel('Time (s)')
    plt.ylabel('Size (AU)')
    subject_id = csv_file.split('_')[0].split("/")[-1]
    day_number = csv_file.split('_')[4].split('.')[0]
    plt.title(f'{subject_id}_{day_number} Pupil Size')
    plt.legend()

    # Create output directory if it doesn't exist
    output_dir = '../../output/plots'
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename and save the plot
    plot_filename = os.path.join(output_dir, f"{os.path.basename(csv_file)}_pupil_size_vs_time.png")
    plt.savefig(plot_filename, dpi=1000)

    if show is True:
        plt.show()

    plt.clf()


def plot_pupil_size_v_time_w_both_eyes_blink_onset_and_offsets(csv_file, show=False):
    """
    This function reads a CSV file containing pupil size data, detects blinks,
    and plots pupil size (left and right eye) vs time.

    Parameters:
        csv_file (str): Path to the CSV file containing pupil size data.
        show (bool): indication of whether to display the plot
    """

    # Read the CSV file using pandas
    dataframe = pd.read_csv(csv_file)

    # Extract columns by header name
    timestamps = dataframe['timestamps']
    pupil_size_left = dataframe['pupil_size_left']
    pupil_size_right = dataframe['pupil_size_right']

    # Detect blinks using a separate function (not included here)
    left_blinks = blink_detection_utils.calculate_total_blinks_and_missing_data(dataframe, 600)
    right_blinks = blink_detection_utils.single_pupil_blink_detection(pupil_size_right, 600, timestamps)

    # Plot pupil size vs time for left and right eye
    plt.plot(timestamps, pupil_size_left, label='Left Size')
    plt.plot(timestamps, pupil_size_right, label='Right Size')

    # Mark blink onsets and offsets with vertical lines
    for blink_onset in left_blinks["blink_onset"]:
        plt.axvline(blink_onset, color='green')
    for blink_offset in left_blinks["blink_offset"]:
        plt.axvline(blink_offset, color='pink')
    for blink_onset in right_blinks["blink_onset"]:
        plt.axvline(blink_onset, color='yellow')
    for blink_offset in right_blinks["blink_offset"]:
        plt.axvline(blink_offset, color='red')

    # Label axes and set title
    plt.xlabel('Time (s)')
    plt.ylabel('Size (AU)')
    subject_id = csv_file.split('_')[0].split("/")[-1]
    day_number = csv_file.split('_')[4].split('.')[0]
    plt.title(f'{subject_id}_{day_number} Pupil Size')
    plt.legend()

    if show is True:
        plt.show()

    # Create output directory if it doesn't exist
    output_dir = '../../output/plots'
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename and save the plot
    plot_filename = os.path.join(output_dir, f"{os.path.basename(csv_file)}_pupil_size_vs_time.png")
    plt.savefig(plot_filename, dpi=1000)
    plt.clf()


def plot_all_time_v_pupil_size_csv_files_in_directory(folder, show=False):
    """
    Plot pupil size data over time with blink annotations for all CSV files in the specified folder.

    Parameters:
        folder (str): Path to the folder containing CSV files.
        show (bool): indication of whether to display the plot.
    """
    # Get a list of all files in the specified folder
    files = sorted(os.listdir(folder))

    # Filter the list to include only CSV files
    csv_files = [f for f in files if f.endswith('.csv')]

    # Loop through each CSV file and plot the pupil size data
    for csv_file in csv_files:
        # Construct the full file path
        file_path = os.path.join(folder, csv_file)

        # Plot the pupil size data for the current CSV file
        plot_pupil_size_v_time_w_both_eyes_blink_onset_and_offsets(file_path, show=show)
        interval_based_plot_pupil_size_v_time(file_path, show=show)

    plt.clf()


def plot_feature_over_three_days(compiled_df, show=False):
    """
    Plot blink rate and missing data percentage across three days for each subject.

    Parameters:
        compiled_df (pd.DataFrame): The return of process_individual_csv, which is a DataFrame containing columns
                                    corresponding to different features.
        show (bool): indication of whether to display the plot

    The function performs the following steps:
        1. Transforms the blink rate data to a wide format where each day is a separate column.
        2. Transforms the missing data percentage to a wide format where each day is a separate column.
        3. Melts the wide-format blink rate data back to a long format suitable for plotting and ANOVA.
        4. Melts the wide-format missing data percentage back to a long format suitable for plotting and ANOVA.
        5. Creates a line plot for blink rate across different days for each subject.
        6. Optionally displays the blink rate plot if `show` is True.
        7. Creates a line plot for missing data percentage across different days for each subject.
        8. Optionally displays the missing data plot if `show` is True.
    """

    # Transform the data to long format if needed
    blink_rate_data_long = compiled_df.pivot(index='subject', columns='day',
                                             values='ebr').reset_index()
    blink_rate_data_long.columns = ['subject', 'day1', 'day2', 'day3']

    # Transform the data to long format if needed
    missing_data_long = (compiled_df.pivot(index='subject', columns='day',
                                           values= 'left_missing_exb').
                         reset_index())
    missing_data_long.columns = ['subject', 'day1', 'day2', 'day3']

    # Melt the dataframe back to long format for ANOVA
    blink_rate_data_melted = pd.melt(blink_rate_data_long, id_vars=['subject'], value_vars=['day1', 'day2', 'day3'],
                                     var_name='day', value_name='average_blink_rate')

    missing_data_melted = pd.melt(missing_data_long, id_vars=['subject'], value_vars=['day1', 'day2', 'day3'],
                                  var_name='day',
                                  value_name='left_missing_data_percentage_excluding_blinks')

    # Line plot
    sns.lineplot(x='day', y='average_blink_rate', data=blink_rate_data_melted, hue='subject', marker='o')
    plt.title('Blink Rate Across Different Days for Each Subject')
    plt.xlabel('Day')
    plt.ylabel('Average Blink Rate')

    if show is True:
        plt.show()

    sns.lineplot(x='day', y='left_missing_data_percentage_excluding_blinks',
                 data=missing_data_melted,
                 hue='subject', marker='o')
    plt.title('Missing Data Across Different Days for Each Subject')
    plt.xlabel('Day')
    plt.ylabel('Percentage of Missing Data')

    if show is True:
        plt.show()
    plt.clf()


def plot_left_vs_right_ebr_agreement(df):
    """
    Plots the agreement between left and right Eye Blink Rate (EBR) with color-coded points
    based on the amount of missing data not accounted for by blinks. Points are darker if
    the average missing data for that participant is greater than the dataset's average missing data.

    Parameters:
    df (pd.DataFrame): DataFrame containing the columns 'left_ebr', 'right_ebr',
                       'left_missing_exb', and 'right_missing_exb'
    """
    plt.figure(figsize=(10, 6))

    # Calculate average of left_missing_exb and right_missing_exb
    # This will give us an average of the amount of missing data not accounted by blinks for each participant
    # and each day
    avg_missing_exb = (df['left_missing_exb'] + df['right_missing_exb']) / 2

    # Color of the points representing the participants is light blue if the average missing data for that
    # participant is less than the average missing dta in the entire dataset. Color the point dark blue if
    # the average missing data is greater than the average missing data in the entire dataset
    for index, row in df.iterrows():
        color = 'blue' if avg_missing_exb.iloc[index] > df['left_missing_exb'].mean() else 'lightblue'
        plt.scatter(row['left_ebr'], row['right_ebr'], alpha=0.5, color=color)

    # Line with slope 1 (ideal relationship)
    max_value = max(df['left_ebr'].max(), df['right_ebr'].max())
    plt.plot([0, max_value], [0, max_value], color='red', linestyle='-', linewidth=2)

    # Dashed lines showing deviation
    for left_ebr, right_ebr in zip(df['left_ebr'], df['right_ebr']):
        plt.plot([left_ebr, left_ebr], [left_ebr, right_ebr], color='grey', linestyle='--', linewidth=0.5)

    plt.xlabel('Left EBR')
    plt.ylabel('Right EBR')
    plt.title('Left vs. Right EBR')
    plt.show()


def plot_left_vs_right_bd_agreement(df):
    """
    Plots the agreement between left and right Blink Duration (BD) with color-coded points
    based on the amount of missing data not accounted for by blinks. Points are darker if
    the average missing data for that participant is greater than the dataset's average missing data.

    Parameters:
    df (pd.DataFrame): DataFrame containing the columns 'left_bd', 'right_bd',
                       'left_missing_exb', and 'right_missing_exb'
    """
    plt.figure(figsize=(10, 6))

    # Calculate average of left_missing_exb and right_missing_exb
    avg_missing_exb = (df['left_missing_exb'] + df['right_missing_exb']) / 2

    # Get the mean of the average missing data
    mean_avg_missing_exb = avg_missing_exb.mean()

    # Plot each participant's data point
    for index, row in df.iterrows():
        color = 'blue' if avg_missing_exb.iloc[index] > mean_avg_missing_exb else 'lightblue'
        plt.scatter(row['left_bd'], row['right_bd'], alpha=0.5, color=color)

    # Line with slope 1 (ideal relationship)
    max_value = max(df['left_bd'].max(), df['right_bd'].max())
    plt.plot([0, max_value], [0, max_value], color='red', linestyle='-', linewidth=2)

    # Dashed lines showing deviation
    for left_bd, right_bd in zip(df['left_bd'], df['right_bd']):
        plt.plot([left_bd, left_bd], [left_bd, right_bd], color='grey', linestyle='--', linewidth=0.5)

    # Set the limits of the plot to ensure the y=x line is well visible
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)

    plt.xlabel('Left Blink Duration')
    plt.ylabel('Right Blink Duration')
    plt.title('Left vs. Right Blink Duration')
    plt.show()


def plot_ibi_variability_boxplot(df):
    """
    Plots a boxplot of IBI (Inter-Blink Interval) variability for left eye, right eye,
    and concatenated IBI data across all subjects.

    Parameters:
    df (pd.DataFrame): DataFrame containing the columns 'left_ibi_var', 'right_ibi_var',
                       and 'concat_ibi_var'
    """
    plt.figure(figsize=(10, 6))

    # Create a boxplot for the specified columns in the DataFrame
    df.boxplot(column=['left_ibi_var', 'right_ibi_var', 'ibi_var'])

    plt.xlabel('Eye')
    plt.ylabel('IBI Variability')
    plt.title('IBI Variability Across Subjects')
    plt.show()
